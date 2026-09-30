"""Run once (needs internet). Downloads MiniLM ONNX embedding model (~90MB)."""
import os, urllib.request
B = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/"
os.makedirs("models", exist_ok=True)
for remote, local in [("onnx/model.onnx", "model.onnx"), ("tokenizer.json", "tokenizer.json")]:
    dest = os.path.join("models", local)
    if not os.path.exists(dest):
        print("Downloading", local); urllib.request.urlretrieve(B + remote, dest)
print("Done.")
