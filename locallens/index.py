import os, json, numpy as np
from .redact import redact
from .backend import get_embedder

def read_file(p):
    if p.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(p).pages)
    return open(p, encoding="utf-8", errors="ignore").read()

def chunk(text, size=600, overlap=100):
    text = " ".join(text.split())
    return [text[i:i + size] for i in range(0, max(len(text), 1), size - overlap) if text[i:i + size].strip()]

def build(folder, out="index.npz", redact_pii=True, fallback=False):
    emb = get_embedder(fallback)
    chunks, src = [], []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".txt", ".md", ".pdf")):
            t = read_file(os.path.join(folder, f))
            if redact_pii: t, _ = redact(t)
            for c in chunk(t): chunks.append(c); src.append(f)
    vecs = emb.encode(chunks)
    np.savez(out, vecs=vecs, chunks=json.dumps(chunks), src=json.dumps(src), backend=emb.name)
    return len(chunks), emb.name

def search(query, index="index.npz", k=3, fallback=False):
    d = np.load(index, allow_pickle=False)
    chunks, src = json.loads(str(d["chunks"])), json.loads(str(d["src"]))
    emb = get_embedder(fallback or str(d["backend"]).startswith("hashing"))
    scores = d["vecs"] @ emb.encode([query])[0]
    return [(float(scores[i]), src[i], chunks[i]) for i in np.argsort(-scores)[:k]]
