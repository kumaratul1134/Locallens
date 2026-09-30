# LocalLens – Private On-Device Knowledge Search for Snapdragon PCs

Enterprises can't paste HR, legal or finance documents into cloud AI tools. **LocalLens** runs semantic search and PII redaction *entirely on the laptop*, using ONNX Runtime with the Qualcomm QNN execution provider so embeddings run on the Snapdragon NPU (falls back to CPU automatically).

## Features
- Semantic search over .txt / .md / .pdf files (MiniLM embeddings, open-source ONNX model)
- Automatic redaction of emails, phone numbers, Aadhaar and PAN before indexing
- NPU → CPU → hashing fallback so it always runs
- `bench` command to compare NPU vs CPU speed

## Setup (Snapdragon Windows laptop, ARM64 Python 3.11+)
```
pip install numpy tokenizers pypdf onnxruntime-qnn
python download_model.py
python cli.py index sample_docs
python cli.py search "how many leave days can I carry forward"
python cli.py bench
```
On x86/other machines use `pip install onnxruntime` instead. Add `--fallback` to skip the model.

## Validation status
Logic validated on CPU with automated tests. NPU speed has not been measured yet; run `python cli.py bench` on Snapdragon hardware to get real numbers.

## Tests
`pip install pytest && python -m pytest -q tests`

## Optimisation for Snapdragon
Batching (16), sequence cap (256), mean-pooling on NPU-friendly ONNX graph, provider auto-selection (`QNNExecutionProvider` with HTP backend). Models can be swapped for Qualcomm AI Hub-compiled variants.
