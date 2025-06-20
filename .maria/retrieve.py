import torch
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from train_two_tower import TwoTowerModel
from load_and_train_text8 import load_pretrained_embedding_layer
import os
import faiss


# --- Load data ---
with open("candidate_doc_texts.pkl", "rb") as f:
    raw_docs = pickle.load(f)
with open("tokenized_candidate_docs.pkl", "rb") as f:
    tokenized_docs = pickle.load(f)
with open("text8_word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)

# --- Load model ---
embedding_layer, _ = load_pretrained_embedding_layer("text8_embeddings.pt", "text8_word_to_index.pkl", freeze=True)
embedding_dim = embedding_layer.embedding_dim
hidden_dim = 128
model = TwoTowerModel(embedding_layer, embedding_dim, hidden_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("two_tower_rnn_model.pt", map_location=device))
model.to(device)
model.eval()

# --- Neural embeddings: cache doc_matrix as numpy for fast search ---
doc_matrix_path = "doc_matrix.npy"
if os.path.exists(doc_matrix_path):
    print("✅ Loaded cached document matrix (numpy).")
    doc_matrix = np.load(doc_matrix_path)
else:
    print("🔄 Computing document embeddings...")
    doc_tensors = [torch.tensor(doc, dtype=torch.long).unsqueeze(0) for doc in tokenized_docs]
    doc_embeddings = []
    with torch.no_grad():
        for doc_tensor in doc_tensors:
            _, doc_vec = model(torch.zeros_like(doc_tensor), doc_tensor)
            doc_embeddings.append(doc_vec.squeeze(0).cpu().numpy())
    doc_matrix = np.stack(doc_embeddings)
    np.save(doc_matrix_path, doc_matrix)
    print("✅ Saved document matrix to disk.")

# --- TF-IDF embeddings ---
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(raw_docs)

# --- Build or load FAISS index ---
faiss_index_path = "faiss.index"
embedding_dim = doc_matrix.shape[1]
if os.path.exists(faiss_index_path):
    print("✅ Loaded FAISS index.")
    index = faiss.read_index(faiss_index_path)
else:
    print("🔄 Building FAISS index...")
    index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine if vectors are normalized)
    # Normalize doc_matrix for cosine similarity
    doc_matrix_norm = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-8)
    index.add(doc_matrix_norm.astype(np.float32))
    faiss.write_index(index, faiss_index_path)
    print("✅ Saved FAISS index to disk.")

# --- Search Function ---
def hybrid_search(query, top_k=5, alpha=0.5):
    # Tokenize and embed query
    token_ids = [word_to_index.get(w, 0) for w in query.lower().split()]
    query_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        query_vec, _ = model(query_tensor, torch.zeros_like(query_tensor))
    query_vec = query_vec.cpu().numpy().squeeze(0)
    # Normalize query vector for cosine similarity
    query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    # Fast neural similarity search with FAISS
    D, I = index.search(query_vec_norm[np.newaxis, :].astype(np.float32), top_k)
    neural_top_idx = I[0]
    neural_sims = D[0]
    # TF-IDF similarities (compute only for top neural candidates)
    tfidf_query = tfidf_vectorizer.transform([query])
    lexical_sims = cosine_similarity(tfidf_query, tfidf_matrix[neural_top_idx]).flatten()
    # Combine
    combined_sims = alpha * neural_sims + (1 - alpha) * lexical_sims
    sorted_idx = np.argsort(combined_sims)[::-1]
    top_indices = neural_top_idx[sorted_idx]
    return [(i, raw_docs[i]) for i in top_indices]

# --- Test ---
if __name__ == "__main__":
    query = input("Enter your query: ")
    results = hybrid_search(query, top_k=5, alpha=0.5)
    print("\nTop Results")
    for rank, (idx, doc) in enumerate(results, 1):
        snippet = doc[:400].strip().replace('\n', ' ')
        print(f"\nRank {rank} (Doc {idx}):\n{snippet}...")