#!/usr/bin/env python3
"""Deep-context benchmark: prefill rate + decode rate at ~190K token depth."""
import json, time, urllib.request, sys

API = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "/models/Hy3-NVFP4"

words = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel","india","juliet"]
text = " ".join(words[i % 10] + str(i) for i in range(55000))
base = {"model": MODEL, "prompt": "Summarize the pattern in this data:\n" + text, "temperature": 0}

def run(max_tokens):
    t0 = time.time()
    req = urllib.request.Request(API + "/v1/completions",
        data=json.dumps(dict(base, max_tokens=max_tokens)).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return time.time() - t0, r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]

t1, ptok, _ = run(1)
print(f"prefill: {ptok} tokens in {t1:.1f}s = {ptok/t1:.0f} tok/s")
t2, _, ctok = run(200)  # prefix-cached second pass ≈ pure decode
print(f"decode at {ptok}-token depth: {ctok} tokens in {t2:.1f}s = {ctok/t2:.1f} tok/s")
