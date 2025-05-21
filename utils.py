import json
import re
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
def load_fiqa_data(path="data/fiqa_corpus.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def chunk_text(text, max_tokens=100, overlap=20):
    """
    Splits text into overlapping chunks.
    Basic whitespace tokenizer used to avoid dependencies.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_tokens]
        chunks.append(" ".join(chunk))
        i += max_tokens - overlap
    return chunks