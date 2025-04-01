import numpy as np
import faiss
import time
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from pdf_utils import pdf_to_text, clean_text, split_into_chunks

@st.cache_resource
def load_model(model_name):
    return SentenceTransformer(model_name)

def build_indices(uploaded_files, model_name="all-MiniLM-L6-v2", chunk_size=500):
    model_emb = load_model(model_name)
    indices = {}
    for pdf_file in uploaded_files:
        st.info(f"**Reading file:** {pdf_file.name}")
        text = pdf_to_text(pdf_file)
        if not text:
            continue
        st.write(f"- Extracted **{len(text)} characters** from {pdf_file.name}.")
        st.write(f"- Cleaning text for {pdf_file.name}...")
        cleaned = clean_text(text)
        st.write(f"- Text length after cleaning: **{len(cleaned)} characters**")
        st.write(f"- Splitting into chunks of {chunk_size} words...")
        chunks = split_into_chunks(cleaned, chunk_size=chunk_size)
        st.write(f"- Created **{len(chunks)} chunks**.")
        st.write(f"- Generating embeddings for {pdf_file.name}...")
        embeddings = load_model(model_name).encode(chunks)
        if len(embeddings.shape) == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        st.write(f"- Embedding shape: **{embeddings.shape}**")
        st.write(f"- Building FAISS index for {pdf_file.name}...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        st.success(f"- FAISS index built for {pdf_file.name}.")
        indices[pdf_file.name] = {"chunks": chunks, "index": index}
    return model_emb, indices

def retrieve_relevant_chunks_multi(query, model_emb, indices, top_k=3):
    all_results = []
    query_embedding = model_emb.encode([query])
    for filename, data in indices.items():
        chunks = data["chunks"]
        index = data["index"]
        distances, idxs = index.search(np.array(query_embedding), top_k)
        for distance, idx in zip(distances[0], idxs[0]):
            result = {
                "pdf": filename,
                "chunk": chunks[idx],
                "distance": distance
            }
            all_results.append(result)
    all_results = sorted(all_results, key=lambda x: x["distance"])
    return all_results[:top_k]

def query_gemini(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    start_time = time.time()
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        elapsed = time.time() - start_time
        st.info(f"API call took **{elapsed:.2f} seconds**.")
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request error: {e}")
        return {}
