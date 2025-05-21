
import os
import faiss
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
from utils import load_fiqa_data, chunk_text

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_corpus(data_path="data/fiqa_corpus.json"):
    docs = load_fiqa_data(data_path)  # this returns a list of strings
    chunks = []
    sources = []

    for i, doc_text in enumerate(docs):
        # doc_text is already a text string
        doc_chunks = chunk_text(doc_text)  # split if necessary
        chunks.extend(doc_chunks)
        sources.extend([f"doc_{i}"] * len(doc_chunks))

    return chunks, sources

def build_faiss_index(chunks, index_path="faiss_index.index"):
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = normalize(embeddings, axis=1)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)  # Save to disk
    return index, embeddings

def load_faiss_index(index_path="faiss_index.index"):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        return index
    else:
        return None

def search(query, index, chunks, sources, k=5):
    query_embedding = model.encode([query])
    query_embedding = normalize(query_embedding, axis=1)
    D, I = index.search(query_embedding, k)

    results = []
    for idx in I[0]:
        results.append({
            "text": chunks[idx],
            "source": sources[idx]
        })

    return results