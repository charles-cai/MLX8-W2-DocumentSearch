import streamlit as st
import pickle
import torch
import numpy as np

from train_two_tower import TwoTowerModel
from load_and_train_text8 import load_pretrained_embedding_layer

st.title("Simple Document Search (Two-Tower RNN Model)")

# --- Load resources ---
with open("text8_word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)
embedding_layer, word_to_index = load_pretrained_embedding_layer(
    "text8_embeddings.pt", "text8_word_to_index.pkl", freeze=True
)
embedding_dim = embedding_layer.embedding_dim
hidden_dim = 128
model = TwoTowerModel(embedding_layer, embedding_dim, hidden_dim)
model.load_state_dict(torch.load("two_tower_rnn_model.pt", map_location=torch.device("cpu")))
model.eval()

with open("raw_candidate_docs.pkl", "rb") as f:
    candidate_docs = pickle.load(f)
with open("tokenized_candidate_docs.pkl", "rb") as f:
    tokenized_candidate_docs = pickle.load(f)

def tokenize_with_vocab(text, word_to_index):
    return [word_to_index.get(word, 0) for word in text.split()]

# Replace embed_doc with model-based embedding
def embed_doc_tower(token_indices):
    if not token_indices:
        return np.zeros(embedding_dim)
    input_tensor = torch.tensor(token_indices).unsqueeze(0)  # shape (1, seq_len)
    with torch.no_grad():
        _, doc_vec = model(torch.zeros_like(input_tensor), input_tensor)
    return doc_vec.squeeze(0).numpy()

# --- Search UI ---
query = st.text_input("Enter your search query:")

if query:
    # Clean and tokenize the query using the same logic as in data.py
    import unicodedata, re
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    lemmatizer = WordNetLemmatizer()
    def clean_text(text):
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = word_tokenize(text)
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmatized)

    cleaned_query = clean_text(query)
    tokenized_query = tokenize_with_vocab(cleaned_query, word_to_index)
    # Use the model to embed the query
    def embed_query_tower(token_indices):
        if not token_indices:
            return np.zeros(embedding_dim)
        input_tensor = torch.tensor(token_indices).unsqueeze(0)
        with torch.no_grad():
            query_vec, _ = model(input_tensor, torch.zeros_like(input_tensor))
        return query_vec.squeeze(0).numpy()
    query_vec = embed_query_tower(tokenized_query)

    # Use the model to embed all candidate docs
    doc_vecs = [embed_doc_tower(doc) for doc in tokenized_candidate_docs]
    doc_vecs = np.stack(doc_vecs)
    sims = doc_vecs @ query_vec / (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8)
    top_n = st.slider("Top N results", 1, 10, 5)
    top_idx = np.argsort(sims)[::-1][:top_n]

    st.subheader("Top Results:")
    for rank, idx in enumerate(top_idx, 1):
        st.markdown(f"**Rank {rank}** (Score: {sims[idx]:.3f})")
        st.write(candidate_docs[idx])
        st.markdown("---")
