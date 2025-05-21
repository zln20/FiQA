import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs, top_k=5):
    """
    Reranks the list of `docs` based on relevance to `query`.
    
    Args:
        query (str): User query.
        docs (List[str]): Retrieved text chunks.
        top_k (int): Number of top docs to return.
    
    Returns:
        List[str]: Reranked top-k chunks.
    """
    pairs = [(query, doc) for doc in docs]
    
    scores = cross_encoder.predict(pairs)
    
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in ranked[:top_k]]