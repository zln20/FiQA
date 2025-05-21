
import pickle
from retriever import build_corpus, build_faiss_index, load_faiss_index
from qa_pipeline import answer_query
import faiss
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
INDEX_PATH = "faiss_index.index"
CHUNKS_PATH = "chunks.pkl"
SOURCES_PATH = "sources.pkl"

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def main():
    print("Loading or building index...")

    if os.path.exists(CHUNKS_PATH) and os.path.exists(SOURCES_PATH):
        chunks = load_pickle(CHUNKS_PATH)
        sources = load_pickle(SOURCES_PATH)
    else:
        chunks, sources = build_corpus("data/fiqa_corpus.json")
        save_pickle(chunks, CHUNKS_PATH)
        save_pickle(sources, SOURCES_PATH)

    index = load_faiss_index(INDEX_PATH)
    if index is None:
        print("Building FAISS index...")
        index, _ = build_faiss_index(chunks)
        faiss.write_index(index, INDEX_PATH)
    else:
        print("FAISS index loaded.")

    while True:
        query = input("\nEnter a query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        print("\nFetching answer from LLM...")
        answer = answer_query(query, index, chunks, sources)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()