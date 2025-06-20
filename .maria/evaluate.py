import torch
import pickle
from train_two_tower import TwoTowerModel
from load_and_train_text8 import load_pretrained_embedding_layer
import torch.nn.functional as F

# --- Evaluation ---
def evaluate(model, eval_data, word_to_index, doc_matrix, k=5):
    total = len(eval_data)
    hits_at_1 = 0
    hits_at_k = 0
    mrr_total = 0

    for query_text, relevant_indices in eval_data:
        tokens = [word_to_index.get(word, 0) for word in query_text.lower().split()]
        query_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            query_vec, _ = model(query_tensor, torch.zeros_like(query_tensor))
            sims = F.cosine_similarity(query_vec, doc_matrix)
            top_k = torch.topk(sims, k).indices.tolist()

        # Metrics
        if any(i in top_k for i in relevant_indices):
            hits_at_k += 1
        if top_k[0] in relevant_indices:
            hits_at_1 += 1
        for rank, doc_idx in enumerate(top_k, 1):
            if doc_idx in relevant_indices:
                mrr_total += 1.0 / rank
                break

    print(f"\n📊 Evaluated {total} queries")
    print(f"Precision@1: {hits_at_1 / total:.2f}")
    print(f"Recall@{k}: {hits_at_k / total:.2f}")
    print(f"MRR: {mrr_total / total:.2f}")

# --- Main logic ---
if __name__ == "__main__":
    # --- Load pretrained model and embeddings ---
    embedding_layer, word_to_index = load_pretrained_embedding_layer(
        "text8_embeddings.pt", "text8_word_to_index.pkl", freeze=True
    )
    embedding_dim = embedding_layer.embedding_dim
    hidden_dim = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerModel(embedding_layer, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load("two_tower_rnn_model.pt", map_location=device))
    model.to(device)
    model.eval()

    # --- Load tokenized candidate docs ---
    with open("tokenized_candidate_docs.pkl", "rb") as f:
        candidate_docs = pickle.load(f)

    # --- Load raw texts for mapping index -> text (optional, for debugging)
    with open("raw_candidate_doc_texts.pkl", "rb") as f:
        candidate_texts = pickle.load(f)

    # --- Encode candidate docs ---
    doc_vectors = []
    for doc in candidate_docs:
        doc_tensor = torch.tensor(doc, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            _, doc_vec = model(torch.zeros_like(doc_tensor), doc_tensor)
            doc_vectors.append(doc_vec.squeeze(0))
    doc_matrix = torch.stack(doc_vectors)  # (num_docs, hidden_dim)

    # --- Load evaluation queries and their relevant doc indices ---
    with open("evaluation_set.pkl", "rb") as f:
        eval_data = pickle.load(f)

    # Make sure relevant indices are integers
    fixed_eval_data = []
    for query, relevant in eval_data:
        if isinstance(relevant, str):
            relevant = [int(x.strip()) for x in relevant.split(',') if x.strip().isdigit()]
        fixed_eval_data.append((query, relevant))

    # --- Run Evaluation ---
    evaluate(model, fixed_eval_data, word_to_index, doc_matrix, k=5)
# --- Run Evaluation ---
if __name__ == "__main__":
    evaluate(model, fixed_eval_data, word_to_index, doc_matrix, k=5)
