# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════
#  NANO-GPT — GPT с нуля на чистом NumPy. Без PyTorch, без фреймворков.
#  Decoder-only transformer: токен+позиционные эмбеддинги, N блоков
#  (LayerNorm → каузальный Multi-Head Self-Attention → residual,
#   LayerNorm → MLP(GELU) → residual), финальный LN, weight tying.
#  Обучение: Adam + кросс-энтропия. Токенизация: посимвольная.
# ═══════════════════════════════════════════════════════════════
import numpy as np

def gelu(x):
    return 0.5*x*(1.0+np.tanh(0.7978845608*(x+0.044715*x**3)))

def gelu_grad(x, gy):
    u = 0.7978845608*(x+0.044715*x**3)
    t = np.tanh(u)
    du = 0.7978845608*(1+3*0.044715*x**2)
    return gy*(0.5*(1+t) + 0.5*x*(1-t**2)*du)

def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e/e.sum(-1, keepdims=True)

def ln_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0/np.sqrt(var+eps)
    xh = (x-mu)*inv
    return xh*g + b, xh, inv

def ln_bwd(xh, inv, gy, g):
    # градиенты: dg, db, gx
    dg = (gy*xh).sum((0,1)); db = gy.sum((0,1))
    gxh = gy*g
    B,T,D = xh.shape
    gx = (gxh - gxh.mean(-1, keepdims=True) - xh*(gxh*xh).mean(-1, keepdims=True))*inv
    return dg, db, gx

class NanoGPT:
    def __init__(self, vocab, d_model=96, n_layer=3, n_head=4, ctx=128, seed=42):
        rng = np.random.default_rng(seed)
        self.V, self.D, self.L, self.H = len(vocab), d_model, n_layer, n_head
        self.T, self.hs = ctx, d_model//n_head
        self.vocab = list(vocab)
        self.stoi = {c:i for i,c in enumerate(self.vocab)}
        self.p = {}
        def unif(k, shape, scale): self.p[k] = rng.uniform(-scale, scale, shape)
        unif('we', (self.V, self.D), 3/np.sqrt(self.D))
        unif('wpe',(self.T, self.D), 0.02)
        for l in range(self.L):
            s = np.sqrt(2.0/self.D)
            unif(f'{l}q',(self.D,self.D), s); unif(f'{l}k',(self.D,self.D), s)
            unif(f'{l}v',(self.D,self.D), s); unif(f'{l}o',(self.D,self.D), s)
            s2 = np.sqrt(2.0/(4*self.D))
            unif(f'{l}fc',(self.D,4*self.D), s2); unif(f'{l}fc2',(4*self.D,self.D), s2)
            self.p[f'{l}ln1g']=np.ones(self.D);  self.p[f'{l}ln1b']=np.zeros(self.D)
            self.p[f'{l}ln2g']=np.ones(self.D);  self.p[f'{l}ln2b']=np.zeros(self.D)
        self.p['lnfg']=np.ones(self.D); self.p['lnfb']=np.zeros(self.D)
        self.opt = {k:{'m':np.zeros_like(v),'v':np.zeros_like(v)} for k,v in self.p.items()}
        self.step = 0

    # ─────────── forward: logits + кэш для backward ───────────
    def forward(self, idx):
        p, L, H, hs = self.p, self.L, self.H, self.hs
        B, T = idx.shape
        x = p['we'][idx] + p['wpe'][:T]
        c = {'emb_idx': idx, 'layers': []}
        mask = np.triu(np.full((T,T), -1e10, dtype=np.float32), 1)
        for l in range(L):
            lc = {}
            h1, xh1, inv1 = ln_fwd(x, p[f'{l}ln1g'], p[f'{l}ln1b'])
            q = (h1 @ p[f'{l}q']).reshape(B,T,H,hs).transpose(0,2,1,3)
            k = (h1 @ p[f'{l}k']).reshape(B,T,H,hs).transpose(0,2,1,3)
            v = (h1 @ p[f'{l}v']).reshape(B,T,H,hs).transpose(0,2,1,3)
            att = np.matmul(q, k.transpose(0,1,3,2))/np.sqrt(hs) + mask
            a = softmax(att)
            o = np.matmul(a, v).transpose(0,2,1,3).reshape(B,T,self.D)
            o = o @ p[f'{l}o']
            x = x + o
            h2, xh2, inv2 = ln_fwd(x, p[f'{l}ln2g'], p[f'{l}ln2b'])
            pre = h2 @ p[f'{l}fc']
            act = gelu(pre)
            m = act @ p[f'{l}fc2']
            x = x + m
            lc.update(xh1=xh1, inv1=inv1, h1=h1, q=q, k=k, v=v, a=a, o=o,
                      xh2=xh2, inv2=inv2, h2=h2, pre=pre, act=act, x_res=None)
            c['layers'].append(lc)
        xf, xhf, invf = ln_fwd(x, p['lnfg'], p['lnfb'])
        logits = xf @ p['we'].T
        c.update(xf=xf, xhf=xhf, invf=invf, x=x)
        return logits, c

    # ─────────── loss ───────────
    def loss(self, logits, targets):
        B,T,V = logits.shape
        pr = softmax(logits)
        n = B*T
        pl = pr.reshape(n, V)[np.arange(n), targets.reshape(-1)]
        ce = -np.log(np.clip(pl, 1e-9, None)).mean()
        return ce, pr

    # ─────────── backward: dLoss/dпараметры ───────────
    def backward(self, targets, pr, c):
        p, L, H, hs, D, V = self.p, self.L, self.H, self.hs, self.D, self.V
        g = {k: np.zeros_like(v) for k, v in p.items()}
        B, T = targets.shape
        # 1) голова: dlogits
        dl = pr.copy()
        dl.reshape(-1, V)[np.arange(B*T), targets.reshape(-1)] -= 1.0
        dl /= B*T
        # weight tying: dwe от головы
        g['we'] += dl.reshape(-1, V).T @ c['xhf'].reshape(-1, D)   # (V,D)
        dxf = dl @ p['we']                                          # (B,T,D)
        # финальный LN
        g['lnfg'], g['lnfb'], dx = ln_bwd(c['xhf'], c['invf'], dxf, p['lnfg'])
        for l in reversed(range(L)):
            lc = c['layers'][l]
            # MLP branch
            g[f'{l}fc2'] += lc['act'].reshape(-1, 4*D).T @ dx.reshape(-1, D)
            dact = dx @ p[f'{l}fc2'].T
            dpre = gelu_grad(lc['pre'], dact)
            g[f'{l}fc'] += lc['h2'].reshape(-1, D).T @ dpre.reshape(-1, 4*D)
            dh2 = dpre @ p[f'{l}fc'].T
            # LN2
            g[f'{l}ln2g'], g[f'{l}ln2b'], dmlp = ln_bwd(lc['xh2'], lc['inv2'], dh2, p[f'{l}ln2g'])
            dx = dx + dmlp        # residual: поток идёт в обе ветки, но mlp-часть уже учтена через x выше
            # вниманиe: dx на входе — градиент после слоя x = x_res + o
            # (порядок: x_out = x_res + o; затем LN2+MLP добавили m)
            # аккуратно: x после блока = x_res + o + m; dx уже включает всё
            do = dx.copy()         # градиент по o
            dres = dx.copy()        # градиент по x_res (residual)
            # о = (o_heads) @ Wo
            oh = lc['o']            # это уже после Wo? — нет: кэш o сохранён ДО Wo? нет, после.
            # исправление: посчитаем oh повторно из a,v
            B_,T_,D_ = do.shape
            oh = np.matmul(lc['a'], lc['v']).transpose(0,2,1,3).reshape(B_,T_,D_)
            g[f'{l}o'] += oh.reshape(-1, D).T @ do.reshape(-1, D)
            doh = do @ p[f'{l}o'].T                                # (B,T,D)
            doh = doh.reshape(B_,T_,H,hs).transpose(0,2,1,3)       # (B,H,T,hs)
            da = np.matmul(doh, lc['v'].transpose(0,1,3,2))        # (B,H,T,T)
            dv = np.matmul(lc['a'].transpose(0,1,3,2), doh)       # (B,H,T,hs)
            # softmax backward: datt = a*(da - sum(da*a))
            s = (da*lc['a']).sum(-1, keepdims=True)
            datt = lc['a']*(da - s)
            datt /= np.sqrt(hs)                                    # /sqrt уже в forward: att=(qk)/sqrt
            dq = np.matmul(datt, lc['k']) / 1.0
            dk = np.matmul(datt.transpose(0,1,3,2), lc['q'])
            dq2 = dq.reshape(B_,T_,D_) if dq.shape[1]==H else dq
            # собираем (B,H,T,hs)->(B,T,D)
            dqv = dq.transpose(0,2,1,3).reshape(B_,T_,D_)
            dkv = dk.transpose(0,2,1,3).reshape(B_,T_,D_)
            dvv = dv.transpose(0,2,1,3).reshape(B_,T_,D_)
            g[f'{l}q'] += lc['h1'].reshape(-1, D).T @ dqv.reshape(-1, D)
            g[f'{l}k'] += lc['h1'].reshape(-1, D).T @ dkv.reshape(-1, D)
            g[f'{l}v'] += lc['h1'].reshape(-1, D).T @ dvv.reshape(-1, D)
            dh1 = dqv @ p[f'{l}q'].T + dkv @ p[f'{l}k'].T + dvv @ p[f'{l}v'].T
            g[f'{l}ln1g'], g[f'{l}ln1b'], dx = ln_bwd(lc['xh1'], lc['inv1'], dh1, p[f'{l}ln1g'])
            dx = dx + dres
        # эмбеддинги
        np.add.at(g['we'], c['emb_idx'], dx)
        g['wpe'][:T] += dx.sum(0)
        return g

    # ─────────── Adam ───────────
    def update(self, grads, lr, b1=0.9, b2=0.95, wd=0.01):
        self.step += 1
        bc1 = 1-b1**self.step; bc2 = 1-b2**self.step
        for k, gr in grads.items():
            m, v = self.opt[k]['m'], self.opt[k]['v']
            m[:] = b1*m + (1-b1)*gr
            v[:] = b2*v + (1-b2)*(gr*gr)
            self.p[k] -= lr*(m/bc1)/(np.sqrt(v/bc2)+1e-8)

    # ─────────── генерация ───────────
    def encode(self, s): return [self.stoi.get(ch, 0) for ch in s]
    def generate(self, prompt, n=60, temp=0.8, topk=6, stop='\n\n'):
        ids = self.encode(prompt)[-self.T:]
        out = []
        for _ in range(n):
            ctx = np.array(ids[-self.T:])[None, :]
            logits, _ = self.forward(ctx)
            lg = logits[0, -1]/max(temp, 1e-3)
            lg = lg - lg.max()
            pr = np.exp(lg); pr /= pr.sum()
            if topk and topk < len(pr):
                thr = np.sort(pr)[-topk]
                pr[pr < thr] = 0; pr /= pr.sum()
            ch = np.random.choice(len(pr), p=pr)
            ids.append(int(ch)); out.append(self.vocab[int(ch)])
            s = ''.join(out)
            if stop and s.endswith(stop): break
        return ''.join(out)
