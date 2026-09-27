# -*- coding: utf-8 -*-
# Собирает демо 50M: index.html (движок+UI) + w50.json (int8-квантованные веса)
import numpy as np, json, base64, os, re, sys

META = os.environ.get('META', 'meta_50m.json')
WFILE = os.environ.get('WFILE', 'weights_big.npz')

meta = json.load(open(META, encoding='utf-8'))
c = meta['cfg']
cfg = dict(V=len(meta['vocab']), D=c['d_model'], L=c['n_layer'], H=c['n_head'], hs=c['d_model']//c['n_head'], T=c['ctx'])

z = np.load(WFILE)
W = {}
total = 0
for k in z.files:
    a = np.ascontiguousarray(z[k], dtype='<f4')
    total += a.size
    sc = float(np.abs(a).max()) / 127.0
    q = np.clip(np.round(a / sc), -127, 127).astype(np.int8) if sc > 0 else np.zeros(a.shape, np.int8)
    W[k] = {'s': list(a.shape), 'q': 1, 'sc': sc,
            'b': base64.b64encode(q.tobytes()).decode()}
print('квантовано %d параметров' % total)

w50 = {'cfg': cfg, 'vocab': meta['vocab'], 'W': W}
with open('w50.json', 'w', encoding='utf-8') as f:
    json.dump(w50, f, ensure_ascii=False, separators=(',', ':'))
print('w50.json:', os.path.getsize('w50.json'), 'bytes')

# ── движок из старого index.html + патч загрузчика под int8 ──
src_html = open('index.html', encoding='utf-8').read()
engine = re.search(r'const W=.*?;\n(.*?)const model=mkEngine', src_html, re.S).group(1)
old_load = "for(const k in W){if(!W[k].b)continue;P[k]={a:f32(W[k].b),s:W[k].s};}"
new_load = """for(const k in W){if(!W[k].b)continue;
  if(W[k].q){const bin=atob(W[k].b),n=bin.length,u8=new Uint8Array(n);
    for(let i=0;i<n;i++)u8[i]=bin.charCodeAt(i);
    const i8=new Int8Array(u8.buffer),sc=W[k].sc,out=new Float32Array(n);
    for(let i=0;i<n;i++)out[i]=i8[i]*sc;
    P[k]={a:out,s:W[k].s};}
  else P[k]={a:f32(W[k].b),s:W[k].s};}"""
assert old_load in engine, 'loader patch point not found'
engine = engine.replace(old_load, new_load)

html = """<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NANO-GPT 50M</title><style>
body{margin:0;background:#0b0e14;color:#cfe;font-family:system-ui,sans-serif}
#chat{max-width:640px;margin:0 auto;padding:12px;height:calc(100vh - 90px);overflow-y:auto}
.m{padding:10px 14px;border-radius:14px;margin:8px 0;max-width:85%;white-space:pre-wrap;line-height:1.4}
.u{background:#1c2b4a;margin-left:auto}.n{background:#152c2a;margin-right:auto}
.hdr{padding:10px;text-align:center;color:#4fd;font-size:14px}
#inp{position:fixed;bottom:0;left:0;right:0;display:flex;gap:8px;padding:10px;background:#0b0e14}
#inp input{flex:1;background:#141a26;color:#cfe;border:1px solid #234;border-radius:10px;padding:12px;font-size:16px}
#inp button{background:#0a5;border:0;color:#021;padding:0 18px;border-radius:10px;font-size:16px}
#load{padding:40px;text-align:center;color:#4fd}
</style></head><body>
<div class="hdr">NANO-GPT 50M — 49.6M параметров, обучена с нуля, int8</div>
<div id="chat"><div id="load">загружаю мозги (~50 МБ)... подожди</div></div>
<div id="inp"><input id="q" placeholder="напиши что-нибудь..." autocomplete="off" disabled><button onclick="send()" disabled>➤</button></div>
<script>
const CFG={V:%%V%%,D:%%D%%,L:%%L%%,H:%%H%%,hs:%%HS%%,T:%%T%%};
%%ENGINE%%
const chat=document.getElementById('chat');let hist='';
function add(cls,txt){const d=document.createElement('div');d.className='m '+cls;d.textContent=txt;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;}
let model=null;
fetch('w50.json').then(r=>r.json()).then(W=>{
  model=mkEngine(W.W,CFG);
  document.getElementById('load').remove();
  document.getElementById('q').disabled=false;
  document.querySelector('#inp button').disabled=false;
  add('n','привет! я нано-гпт на 49.6 млн параметров, обучена с нуля. спроси: как дела, кто ты, расскажи о себе...');
}).catch(e=>add('n','не загрузилось: '+e));
function send(){const inp=document.getElementById('q'),q=inp.value.trim().toLowerCase();if(!q||!model)return;
add('u',q);inp.value='';
const think=add('n','...думаю...');
setTimeout(()=>{const ans=model.generate(hist+'В: '+q+'\\nО:',50,0.7,5).split('\\n')[0].trim();
think.textContent=ans||'...';hist+='В: '+q+'\\nО: '+ans+'\\n\\n';},50);}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
</script></body></html>"""
html = html.replace('%%ENGINE%%', engine)
for k, v in cfg.items():
    html = html.replace('%%' + k.upper() + '%%', str(v))
os.makedirs('web50', exist_ok=True)
open('web50/index.html', 'w', encoding='utf-8').write(html)
open('web50/w50.json', encoding='utf-8').write('')  # потом скопируем
print('web50/index.html:', os.path.getsize('web50/index.html'), 'bytes')
