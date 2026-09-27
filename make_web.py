# -*- coding: utf-8 -*-
# Собирает standalone index.html: движок + веса (base64) + чат-UI
import numpy as np, json, base64, os
from nanogpt import NanoGPT
meta = json.load(open('meta.json', encoding='utf-8'))
c = meta['cfg']
cfg = dict(V=len(meta['vocab']), D=c['d_model'], L=c['n_layer'], H=c['n_head'], hs=c['d_model']//c['n_head'], T=c['ctx'])
WFILE = os.environ.get('WFILE', 'weights.npz')
z = np.load(WFILE)
W = {}
for k in z.files:
    a = np.ascontiguousarray(z[k], dtype='<f4')
    W[k] = {'s': list(a.shape), 'b': base64.b64encode(a.tobytes()).decode()}
W['__vocab'] = meta['vocab']
engine = open('engine.js', encoding='utf-8').read()
html = """<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NANO-GPT</title><style>
body{margin:0;background:#0b0e14;color:#cfe;font-family:system-ui,sans-serif}
#chat{max-width:640px;margin:0 auto;padding:12px;height:calc(100vh - 90px);overflow-y:auto}
.m{padding:10px 14px;border-radius:14px;margin:8px 0;max-width:85%;white-space:pre-wrap;line-height:1.4}
.u{background:#1c2b4a;margin-left:auto}.n{background:#152c2a;margin-right:auto}
.hdr{padding:10px;text-align:center;color:#4fd;font-size:14px}
#inp{position:fixed;bottom:0;left:0;right:0;display:flex;gap:8px;padding:10px;background:#0b0e14}
#inp input{flex:1;background:#141a26;color:#cfe;border:1px solid #234;border-radius:10px;padding:12px;font-size:16px}
#inp button{background:#0a5;border:0;color:#021;padding:0 18px;border-radius:10px;font-size:16px}
</style></head><body>
<div class="hdr">NANO-GPT — GPT с нуля, char-level, %%HDR%%</div>
<div id="chat"></div>
<div id="inp"><input id="q" placeholder="напиши что-нибудь..." autocomplete="off"><button onclick="send()">➤</button></div>
<script>
const W=%%WEIGHTS%%;
%%ENGINE%%
const model=mkEngine(W,{V:%%V%%,D:%%D%%,L:%%L%%,H:%%H%%,hs:%%HS%%,T:%%T%%});
const chat=document.getElementById('chat');let hist='';
function add(cls,txt){const d=document.createElement('div');d.className='m '+cls;d.textContent=txt;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;}
add('n','привет! я нано-гпт, обучен с нуля. задай вопрос из простых: как дела, кто ты, расскажи шутку, пока...');
function send(){const inp=document.getElementById('q'),q=inp.value.trim().toLowerCase();if(!q)return;
add('u',q);inp.value='';
setTimeout(()=>{const ans=model.generate(hist+'В: '+q+'\\nО:',60,0.7,5).split('\\n')[0].trim();
add('n',ans||'...');hist+='В: '+q+'\\nО: '+ans+'\\n\\n';},50);}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
</script></body></html>"""
html = html.replace('%%WEIGHTS%%', json.dumps(W, ensure_ascii=False, separators=(',',':')))
html = html.replace('%%ENGINE%%', engine)
html = html.replace('%%HDR%%', os.environ.get('HDR', '%d параметров' % sum(int(np.prod(z[k].shape)) for k in z.files if not k.startswith('__'))))
for k,v in cfg.items(): html = html.replace('%%'+k+'%%', str(v)); html = html.replace('%%'+k.upper()+'%%', str(v))
os.makedirs('web', exist_ok=True)
open('web/index.html','w',encoding='utf-8').write(html)
print('web/index.html:', os.path.getsize('web/index.html'), 'bytes')
