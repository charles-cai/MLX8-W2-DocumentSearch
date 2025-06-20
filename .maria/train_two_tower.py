# -----------------------------------------------------------------------------
# Training Script: Two-Tower Model using ParallelEncoders
#
# Step 1: Loads pretrained text8 embeddings (from text8_embeddings.pt) ✅
# Step 2: Loads tokenized MS MARCO triples (query, relevant doc, irrelevant doc) ✅
# Step 3: Initializes a two-tower model that uses ParallelEncoders to encode 
#         queries and documents separately ✅
# Step 4: Trains the model with triplet loss so relevant documents are closer 
#         to queries than irrelevant ones ✅
# Step 5: Saves the trained model as two_tower_rnn_model.pt ✅
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from two_tower import TwoTowerModel
import pickle
from load_and_train_text8 import load_pretrained_embedding_layer
from torch.nn.utils.rnn import pad_sequence

# --- Dataset for tokenized MS MARCO triples ---
class TripleDataset(Dataset):
    # Loads tokenized triples and provides them as tensors for training
    def __init__(self, triples, word_to_index):
        self.triples = triples
        self.word_to_index = word_to_index
    def __len__(self):
        return len(self.triples)
    def __getitem__(self, idx):
        q, pos, neg = self.triples[idx]
        if isinstance(q[0], str):
            q = [self.word_to_index.get(w, 0) for w in q]
            pos = [self.word_to_index.get(w, 0) for w in pos]
            neg = [self.word_to_index.get(w, 0) for w in neg]
        return (
            torch.tensor(q, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )

def collate_triples(batch):
    # Pads batches of variable-length triples for training
    queries, pos_docs, neg_docs = zip(*batch)
    queries_padded = pad_sequence(queries, batch_first=True, padding_value=0)
    pos_docs_padded = pad_sequence(pos_docs, batch_first=True, padding_value=0)
    neg_docs_padded = pad_sequence(neg_docs, batch_first=True, padding_value=0)
    return queries_padded, pos_docs_padded, neg_docs_padded

# --- Training pipeline ---
if __name__ == "__main__":
    # --- Load pretrained text8 embeddings and vocab ---
    embedding_layer, word_to_index = load_pretrained_embedding_layer(
        "text8_embeddings.pt",
        "text8_word_to_index.pkl",
        freeze=True
    )
    embedding_dim = embedding_layer.embedding_dim
    hidden_dim = 128

    # --- Load tokenized train and val triples ---
    train_triples_path = "tokenized_train_triples.pkl"
    val_triples_path = "tokenized_val_triples_semantic.pkl"  # or your actual val triples file

    with open(train_triples_path, "rb") as f:
        train_triples = pickle.load(f)
    with open(val_triples_path, "rb") as f:
        val_triples = pickle.load(f)

    train_dataset = TripleDataset(train_triples, word_to_index)
    val_dataset = TripleDataset(val_triples, word_to_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, pin_memory=(torch.cuda.is_available()), collate_fn=collate_triples
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, pin_memory=(torch.cuda.is_available()), collate_fn=collate_triples
    )

    # --- Initialize two-tower model and optimizer ---
    model = TwoTowerModel(embedding_layer, embedding_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    epochs = 10

    def evaluate(model, loader, criterion, device):
        # Computes average triplet loss on validation set
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for query, pos_doc, neg_doc in loader:
                query = query.to(device)
                pos_doc = pos_doc.to(device)
                neg_doc = neg_doc.to(device)
                query_vec, pos_vec = model(query, pos_doc)
                _, neg_vec = model(query, neg_doc)
                loss = criterion(query_vec, pos_vec, neg_vec)
                total_loss += loss.item()
        model.train()
        return total_loss / len(loader)

    # --- Training loop ---
    for epoch in range(epochs):
        total_loss = 0
        for query, pos_doc, neg_doc in train_loader:
            # Forward, backward, optimize for each batch
            query = query.to(device)
            pos_doc = pos_doc.to(device)
            neg_doc = neg_doc.to(device)
            optimizer.zero_grad()
            query_vec, pos_vec = model(query, pos_doc)
            _, neg_vec = model(query, neg_doc)
            loss = criterion(query_vec, pos_vec, neg_vec)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # --- Save trained model weights ---
    torch.save(model.state_dict(), "two_tower_rnn_model.pt")
    