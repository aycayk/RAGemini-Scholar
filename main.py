import streamlit as st
from pdf_utils import extract_pdf_files
from model_utils import build_indices, retrieve_relevant_chunks_multi, query_gemini
from chat_prompt import create_prompt

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    

    model_info = {
        "all-MiniLM-L6-v2": "Fast & efficient, general semantic search",
        "all-mpnet-base-v2": "Higher accuracy, detailed analysis",
        "paraphrase-MiniLM-L6-v2": "Paraphrase analysis",
        "distiluse-base-multilingual-cased-v2": "Multilingual support",
        "all-distilroberta-v1": "Alternative, robust model",
        "allenai-specter": "Scientific articles, technical and mathematical content",
        #"mathbert": "Optimized for mathematical expressions"
    }

    st.sidebar.header("PDF Processing")
    uploaded_files = st.sidebar.file_uploader("Upload PDFs or ZIP", type=["pdf", "zip"], accept_multiple_files=True)
    model_name = st.sidebar.selectbox(
        "Model Selection",
        options=list(model_info.keys()),
        format_func=lambda x: f"{x} - {model_info[x]}",
        help="Select the model to analyze the article content"
    )
    chunk_size = st.sidebar.slider("Chunk Size", min_value=200, max_value=800, value=500, step=50)
    if uploaded_files and st.sidebar.button("Process Articles"):
        with st.sidebar:
            #st.info("Processing articles...")
            pdf_files = extract_pdf_files(uploaded_files)
            model_emb, indices = build_indices(pdf_files, model_name=model_name, chunk_size=chunk_size)
            st.session_state.indices = indices
            st.session_state.model_emb = model_emb
            st.success("All articles processed successfully!")
            st.session_state.chat_history = []

    st.title("RAGemini Scholar")
    st.markdown("<p class='header-title'>Using Google's Gemini 2.0 Flash model, this project processes and analyzes uploaded articles to create a dynamic platform for accurate Q&A interactions.</p>", unsafe_allow_html=True)

    if "indices" not in st.session_state:
        st.warning("Please upload and process PDFs first.")
        return
    

    st.subheader("Conversation History")
    for msg in st.session_state.chat_history:
        role = "User" if msg["role"] == "User" else "Bot"
        css_class = "user" if msg["role"] == "User" else "bot"
        st.markdown(f"<div class='chat-msg {css_class}'><strong>{role}:</strong> {msg['content']}</div>", unsafe_allow_html=True)

    with st.form("query_form"):
        query = st.text_input("Enter your query:")
        submitted = st.form_submit_button("Get Answer")
    
    if submitted and query:
        with st.spinner("Searching for answer..."):
            retrieved_results = retrieve_relevant_chunks_multi(
                query, 
                st.session_state.model_emb, 
                st.session_state.indices, 
                top_k=3
            )
            final_prompt = create_prompt(query, retrieved_results, st.session_state.chat_history)
            with st.expander("Generated Prompt"):
                st.code(final_prompt, language="text")
            api_key = st.secrets["API_KEY"]  # Replace with your API key
            result = query_gemini(final_prompt, api_key)
            answer = ""
            for candidate in result.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    answer += part.get("text", "")
            st.session_state.chat_history.append({"role": "User", "content": query})
            st.session_state.chat_history.append({"role": "Bot", "content": answer})
            st.subheader("Answer")
            st.write(answer)

    
    st.markdown("""
    <style>
    .header-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #333333;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 500;
        color: #555555;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .chat-msg {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .user {
        background-color: #333333;
        color: white;
    }
    .bot {
        background-color: #00008B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
