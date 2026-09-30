import argparse, time
from locallens import index
from locallens.redact import redact
from locallens.backend import get_embedder, providers

p = argparse.ArgumentParser(description="LocalLens: private on-device document search")
s = p.add_subparsers(dest="cmd", required=True)
a = s.add_parser("index"); a.add_argument("folder"); a.add_argument("--no-redact", action="store_true"); a.add_argument("--fallback", action="store_true")
b = s.add_parser("search"); b.add_argument("query"); b.add_argument("-k", type=int, default=3); b.add_argument("--fallback", action="store_true")
c = s.add_parser("redact"); c.add_argument("text")
s.add_parser("bench")
x = p.parse_args()

if x.cmd == "index":
    n, be = index.build(x.folder, redact_pii=not x.no_redact, fallback=x.fallback)
    print(f"Indexed {n} chunks using backend: {be}")
elif x.cmd == "search":
    for sc, src, ch in index.search(x.query, k=x.k, fallback=x.fallback):
        print(f"[{sc:.2f}] {src}: {ch[:200]}...\n")
elif x.cmd == "redact":
    t, n = redact(x.text); print(t); print(n)
elif x.cmd == "bench":
    e = get_embedder(); texts = ["Employees must report security incidents within 24 hours."] * 64
    e.encode(texts[:4]); t = time.time(); e.encode(texts); d = time.time() - t
    print(f"Providers: {providers()}\nBackend: {e.name}\n64 chunks in {d:.3f}s ({64/d:.0f} chunks/s)")
