# -*- coding: utf-8 -*-
# Конвейер 700M: шард Либрусека -> токены -> GitHub-релиз corpus-700m.
# Откато-устойчив: источник истины — ассеты релиза (.done маркеры).
# Заливка потоковая (-T), при пропуске готовой части курсор сдвигается.
import os, sys, json, time, struct, gzip, subprocess

try:
    import zstandard as zstd
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zstandard", "tokenizers"])
    import zstandard as zstd
from tokenizers import Tokenizer

D = os.path.dirname(os.path.abspath(__file__))
UP = os.path.dirname(D)
TOK = os.path.join(D, "bpe_tokenizer_700.json")
GT = os.environ.get("GITHUB_TOKEN_2") or os.environ.get("GITHUB_TOKEN")
REPO = "lucifermornngstar52-cell/nano-gpt"
TAG = "corpus-700m"
BASE = "https://huggingface.co/datasets/IlyaGusev/librusec_full/resolve/main/"
SHARDS = [
    "000024-030559.jsonl.zst", "030560-060423.jsonl.zst",
    "060424-074391.jsonl.zst", "074392-091839.jsonl.zst",
    "091841-104214.jsonl.zst", "104215-113436.jsonl.zst",
    "113437-132107.jsonl.zst", "132108-147517.jsonl.zst",
]
PART = 1900_000_000  # 1.9 ГБ на ассет (лимит GH 2 ГиБ)

LOG = open(os.path.join(D, "pipe.log"), "a", encoding="utf-8")
def log(m):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()

def gh(path, data=None):
    cmd = ["curl", "-sS", "-m", "60", "-H", "Authorization: token " + GT]
    if data is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(data)]
    cmd.append("https://api.github.com/repos/%s/%s" % (REPO, path))
    return json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout or "{}")

def rid():
    r = gh("releases/tags/" + TAG)
    if r.get("id"):
        return r["id"]
    r = gh("releases", {"tag_name": TAG, "name": "corpus-700m tokens",
                        "body": "Токенизированный корпус для 700M-1B (uint16, словарь 8192)"})
    return r["id"]

def assets(R):
    return {a["name"]: a for a in gh("releases/%d/assets" % R)}

def upload(R, name, path):
    url = ("https://uploads.github.com/repos/%s/releases/%d/assets?name=%s"
           % (REPO, R, name))
    r = subprocess.run(["curl", "-sS", "-m", "7200", "-X", "POST",
        "-H", "Authorization: token " + GT,
        "-H", "Content-Type: application/octet-stream",
        "-T", path, url], capture_output=True, text=True)
    try:
        j = json.loads(r.stdout or "{}")
        return bool(j.get("id")) or "already_exists" in (r.stdout or "")
    except Exception:
        return False

def clean(text):
    text = text.replace("\r", "").replace("\t", " ")
    out, blank = [], 0
    for line in text.split("\n"):
        line = " ".join(line.split())
        if not line:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(line)
    return "\n".join(out)

def ensure_books(i):
    tpath = os.path.join(D, "books_%02d.txt" % i)
    if os.path.exists(tpath) and os.path.getsize(tpath) > 100e6:
        return tpath
    shard = SHARDS[i]; zpath = os.path.join(D, shard)
    for attempt in range(1, 7):
        r = subprocess.run(["curl", "-sL", "-C", "-", "--retry", "3",
                           "--retry-all-errors", "-o", zpath, BASE + shard])
        if r.returncode == 0:
            break
        log("шард %d: curl код %d, попытка %d" % (i, r.returncode, attempt))
        time.sleep(10)
    else:
        raise RuntimeError("шард %d не скачался" % i)
    dctx = zstd.ZstdDecompressor()
    books = 0
    with open(zpath, "rb") as f, open(tpath, "w", encoding="utf-8") as out:
        reader = dctx.stream_reader(f)
        buf = b""
        while True:
            chunk = reader.read(1 << 24)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                ln = ln.strip()
                if not ln or len(ln) < 2000:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                if obj.get("lang") not in (None, "ru", ""):
                    continue
                text = "\n\n".join(s for s in (obj.get("sections") or []) if isinstance(s, str))
                if len(text) < 5000:
                    continue
                text = clean(text)
                if len(text) < 3000:
                    continue
                out.write(text + "\n\n")
                books += 1
    os.remove(zpath)
    log("шард %d: %d книг распаковано" % (i, books))
    return tpath

tk = None
def tokenizer():
    global tk
    if tk is None:
        tk = Tokenizer.from_file(TOK)
    return tk

def ids2bytes(ids):
    return struct.pack('<%dH' % len(ids), *ids)

def tokenize_file(src, dst):
    t0 = time.time(); toks = chars = 0
    T = tokenizer()
    tmp = dst + ".tmp"
    with open(src, "r", encoding="utf-8") as f, open(tmp, "wb") as out:
        buf = []
        blen = 0
        def flush():
            enc = T.encode_batch(buf)
            out.write(b''.join(ids2bytes(e.ids) for e in enc))
            return sum(len(e.ids) for e in enc)
        for line in f:
            buf.append(line); blen += len(line); chars += len(line)
            if blen >= 8_000_000:
                toks += flush(); buf.clear(); blen = 0
        if buf:
            toks += flush()
    os.rename(tmp, dst)
    log("%s: %.0fM симв -> %.0fM ток (%.2f с/т), %.0f с" % (
        os.path.basename(dst), chars/1e6, toks/1e6, chars/max(toks,1), time.time()-t0))
    return toks

def ship_bin(R, binpath, name):
    """Режет bin на части, заливает (потоково), грузит маркер, чистит локально."""
    size = os.path.getsize(binpath)
    n = max(1, -(-size // PART))
    have = assets(R)
    ok = True
    with open(binpath, "rb") as f:
        for p in range(n):
            pn = "%s.p%d" % (name, p)
            pp = os.path.join(D, pn)
            if pn in have:
                log("  %s уже на GitHub" % pn)
                f.seek(min(PART, size - f.tell()), 1)  # курсор вперёд на размер части
                continue
            with open(pp, "wb") as o:
                left = PART
                while left > 0:
                    b = f.read(min(1 << 24, left))
                    if not b:
                        break
                    o.write(b); left -= len(b)
            log("  заливаю %s (%.1f ГБ)..." % (pn, os.path.getsize(pp)/1e9))
            if not upload(R, pn, pp):
                log("  ОШИБКА заливки %s" % pn); ok = False
                break
            os.remove(pp)
    if not ok:
        return False
    info = os.path.join(D, name + ".done")
    with open(info, "w") as f:
        f.write(json.dumps({"parts": n, "bytes": size}))
    upload(R, name + ".done", info)
    os.remove(info)
    os.remove(binp)
    return True

if __name__ == "__main__":
    log("=== конвейер 700M старт ===")
    R = rid()
    grand = 0
    for i in range(8):
        name = "tok700_%02d" % i
        have = assets(R)
        if name + ".done" in have:
            log("%s: уже на GitHub, пропуск" % name)
            continue
        text = ensure_books(i)
        binp = os.path.join(D, name + ".bin")
        if not os.path.exists(binp):
            toks = tokenize_file(text, binp)
        else:
            toks = os.path.getsize(binp) // 2
        grand += toks
        if ship_bin(R, binp, name):
            os.remove(text)
            log("%s отгружен (%.0fM ток), текст удалён" % (name, toks/1e6))
    # диалоги x2
    dgz = os.path.join(UP, "corpus_big.txt.gz")
    have = assets(R)
    if os.path.exists(dgz) and "tok700_dial.done" not in have:
        T = tokenizer()
        ddst = os.path.join(D, "tok700_dial.bin")
        t0 = time.time(); cnt = [0]
        with gzip.open(dgz, "rt", encoding="utf-8") as f, open(ddst + ".tmp", "wb") as out:
            buf, blen = [], 0
            def flush():
                enc = T.encode_batch(buf)
                out.write(b''.join(ids2bytes(e.ids) for e in enc))
                cnt[0] += sum(len(e.ids) for e in enc)
                buf.clear()
            for rep in (1, 2):
                f.seek(0)
                for line in f:
                    buf.append(line); blen += len(line)
                    if blen >= 8_000_000:
                        flush(); blen = 0
                if buf:
                    flush()
        os.rename(ddst + ".tmp", ddst)
        log("диалоги x2: %.0fM ток, %.0f с" % (cnt[0]/1e6, time.time()-t0))
        ship_bin(R, ddst, "tok700_dial")
        grand += cnt[0]
    log("=== ГОТОВО: всего отгружено ~%.2f млрд ток ===" % (grand/1e9))
