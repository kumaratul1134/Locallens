"""On-device PII redaction (India-focused). Nothing leaves the laptop."""
import re
PATTERNS = {
    "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "PHONE": r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b",
}
def redact(text):
    counts = {}
    for name, pat in PATTERNS.items():
        text, n = re.subn(pat, f"[{name}]", text)
        counts[name] = n
    return text, counts
