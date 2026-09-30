import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from locallens.redact import redact
from locallens import index

def test_redact():
    t, n = redact("mail a@b.com pan ABCDE1234F aadhaar 1234 5678 9012 ph 9876543210")
    assert "a@b.com" not in t and "ABCDE1234F" not in t and "9876543210" not in t
    assert n["EMAIL"] == 1 and n["PAN"] == 1 and n["AADHAAR"] == 1 and n["PHONE"] == 1

def test_search(tmp_path):
    idx = str(tmp_path / "i.npz")
    index.build("sample_docs", out=idx, fallback=True)
    assert index.search("how many leave days carry forward", idx, fallback=True)[0][1] == "hr_policy.txt"
    assert index.search("report data breach", idx, fallback=True)[0][1] == "security_policy.txt"
    assert index.search("hotel cap per night", idx, fallback=True)[0][1] == "expense_policy.txt"
