"""Embedding backends. Uses the Snapdragon NPU (QNN) when available,
otherwise CPU, otherwise a dependency-free hashing fallback."""
import os, re, zlib, numpy as np

MODEL = os.path.join("models", "model.onnx")
TOK = os.path.join("models", "tokenizer.json")

def providers():
    try:
        import onnxruntime as ort
        av = ort.get_available_providers()
        if "QNNExecutionProvider" in av:
            return [("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}), "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    except ImportError:
        return []

class HashEmbedder:
    name = "hashing-fallback"
    def __init__(self, dim=512): self.dim = dim
    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", t.lower())
            for w in words + [a + "_" + b for a, b in zip(words, words[1:])]:
                out[i, zlib.crc32(w.encode()) % self.dim] += 1
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out

class OnnxEmbedder:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(TOK)
        self.tok.enable_truncation(256)
        self.tok.enable_padding()
        self.sess = ort.InferenceSession(MODEL, providers=providers())
        self.name = "onnx:" + self.sess.get_providers()[0]
        self.inputs = {i.name for i in self.sess.get_inputs()}
    def encode(self, texts, bs=16):
        res = []
        for s in range(0, len(texts), bs):
            enc = self.tok.encode_batch(texts[s:s + bs])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs: feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            m = mask[..., None]
            v = (h * m).sum(1) / np.maximum(m.sum(1), 1)
            res.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9))
        return np.vstack(res).astype(np.float32)

def get_embedder(force_fallback=False):
    if not force_fallback and os.path.exists(MODEL) and os.path.exists(TOK):
        try: return OnnxEmbedder()
        except Exception as e: print("ONNX unavailable, using fallback:", e)
    return HashEmbedder()
