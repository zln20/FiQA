from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from reranker import rerank
from retriever import model as embedder_model  
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
model_id="gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
from transformers import StoppingCriteria, StoppingCriteriaList

import re

def trim_to_sentences(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return ' '.join(sentences[:max_sentences])

def get_embedding(text):
    return embedder_model.encode([text])[0]

def retrieve_top_k(query, index, chunks, k=3):
    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding])
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

def answer_query(query, index, chunks, sources, k=3, initial_k=10):
    
    initial_retrieved = retrieve_top_k(query, index, chunks, k=initial_k)
    
    top_chunks = rerank(query, initial_retrieved, top_k=k)

    context = "\n\n".join(top_chunks)
    prompt = (
        f"You are a helpful financial assistant. "
        f"Use the following information to answer clearly and accurately.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding="max_length").to(device)
    inputs["attention_mask"] = inputs["attention_mask"].to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=256,  
        no_repeat_ngram_size=4,
        do_sample=False,
        num_beams=3,
        early_stopping=True
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if "Question:" in answer and "Answer:" in answer:
        parts = answer.rsplit("Question:", 1)[-1]
        if "Answer:" in parts:
            final_answer = parts.split("Answer:")[-1].strip()
            return trim_to_sentences(final_answer)

    return trim_to_sentences(answer)