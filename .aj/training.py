import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import defaultdict
import pickle
import json
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# -----------------------------
# 1. Load MS MARCO V1.1 training dataset
# -----------------------------

# This will stream the data, you don't have to download the full file
dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")  # or "validation"

# 1. Build passage pool and index mapping for fast sampling and seed for reproducibility
passage_to_idx = dict()
idx_to_passage = []
for row in tqdm(dataset, desc="Building passage pool..."):
    for p in row['passages']['passage_text']:
        if p not in passage_to_idx:
            passage_to_idx[p] = len(idx_to_passage)
            idx_to_passage.append(p)
num_passages = len(idx_to_passage)

# 2. For each query, map relevant passage indices
triples = []
for row in tqdm(dataset, desc="Creating triples..."):
    query = row['query']
    relevant_passages = row['passages']['passage_text'][:10]
    relevant_indices = [passage_to_idx[p] for p in relevant_passages]
    
    # For fast sampling: mask out relevant indices
    mask = np.ones(num_passages, dtype=bool)
    mask[relevant_indices] = False
    irrelevant_indices = np.random.choice(np.where(mask)[0], 10, replace=False)
    irrelevant_passages = [idx_to_passage[i] for i in irrelevant_indices]

    triples.append((query, relevant_passages, irrelevant_passages))

with open("triples_full.pkl", "wb") as f:
    pickle.dump(triples, f)


# -------------------------------
# 2. Embed queries, rel and irrel documents using a pre-trained SentenceTransformer model
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/msmarco-MiniLM-L6-v3')
model     = AutoModel.from_pretrained('sentence-transformers/msmarco-MiniLM-L6-v3').to(device)
embed_dim = model.config.hidden_size  # 384 for MiniLM

def tokenize_and_embed(texts, batch_size=64, show_progress=True):
    token_embeddings = []
    lengths = []
    total = len(texts)
    it = range(0, total, batch_size)
    if show_progress:
        it = tqdm(it, desc="Embedding", total=(total+batch_size-1)//batch_size)
    for i in it:
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**enc).last_hidden_state  # (batch, seq_len, embed_dim)
        # Move outputs to CPU only once
        token_embeddings.extend(output.cpu().split(1, dim=0))
        lengths.extend(enc['attention_mask'].sum(dim=1).cpu().tolist())
    # Remove the batch dimension from each embedding
    token_embeddings = [emb.squeeze(0) for emb in token_embeddings]
    return token_embeddings, lengths

query_texts   = [t[0] for t in triples]
rel_doc_texts = [t[1] for t in triples]
irrel_doc_texts = [t[2] for t in triples]

# Flatten the lists of lists to a single list of strings
rel_doc_texts_flat = [doc for docs in rel_doc_texts for doc in docs]
irrel_doc_texts_flat = [doc for docs in irrel_doc_texts for doc in docs]

query_embeds, query_lens         = tokenize_and_embed(query_texts, batch_size=64)
rel_doc_embeds, rel_doc_lens     = tokenize_and_embed(rel_doc_texts_flat, batch_size=64)
irrel_doc_embeds, irrel_doc_lens = tokenize_and_embed(irrel_doc_texts_flat, batch_size=64)

# Save embeddings to files
with open("query_embeds.pkl", "wb") as f:
    pickle.dump((query_embeds, query_lens), f)
with open("rel_doc_embeds.pkl", "wb") as f:
    pickle.dump((rel_doc_embeds, rel_doc_lens), f)
with open("irrel_doc_embeds.pkl", "wb") as f:
    pickle.dump((irrel_doc_embeds, irrel_doc_lens), f)

# # Load embeddings from files
# with open("query_embeds.pkl", "rb") as f:
#     query_embeds, query_lens = pickle.load(f)
# with open("rel_doc_embeds.pkl", "rb") as f:
#     rel_doc_embeds, rel_doc_lens = pickle.load(f)
# with open("irrel_doc_embeds.pkl", "rb") as f:
#     irrel_doc_embeds, irrel_doc_lens = pickle.load(f)


# -----------------------------
# 3. Define distance functions & Triplet loss function
# -----------------------------

# Cosine similarity for calculation of cosine distance
def cosine_similarity(x, y):
    return F.cosine_similarity(x, y, dim=1)

# Cosine - Smallest value means most similar
def cosine_distance(x, y):
    return 1 - cosine_similarity(x, y)

# Euclidean (L2) - Smallest value means most similar
def euclidean_distance(x, y):
    return torch.norm(x - y, p=2, dim=1)

# Squared Euclidean - Smallest value means most similar
def squared_euclidean_distance(x, y):
    return torch.sum((x - y) ** 2, dim=1)

# Manhattan (L1) - Smallest value means most similar
def manhattan_distance(x, y):
    return torch.norm(x - y, p=1, dim=1)

# Chebyshev (L-infinity) - Smallest value means most similar
def chebyshev_distance(x, y):
    return torch.max(torch.abs(x - y), dim=1).values

# Minkowski - Smallest value means most similar
def minkowski_distance(x, y, p=3):
    return torch.norm(x - y, p=p, dim=1)

# Triplet loss function - will compute the loss for a batch of triplets
def triplet_loss_function(query, relevant_doc, irrelevant_doc, distance_function, margin):
    rel_dist = distance_function(query, relevant_doc)         # (batch,)
    irrel_dist = distance_function(query, irrelevant_doc)     # (batch,)
    triplet_loss = torch.relu(rel_dist - irrel_dist + margin)
    return triplet_loss.mean()                                # Average over batch


# -----------------------------
# 4. Define Two Tower Model (QueryTower and DocTower)
# -----------------------------

class QueryTower(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers=1, rnn_type='gru'):
        super().__init__()
        if rnn_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("Unknown rnn_type: choose 'gru' or 'lstm'")

    def forward(self, x, lengths):
        # x: (batch, seq_len, embed_dim)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        if isinstance(h, tuple):  # LSTM
            h = h[0]
        return h[-1]  # (batch, hidden_dim)

class DocTower(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers=1, rnn_type='gru'):
        super().__init__()
        if rnn_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("Unknown rnn_type: choose 'gru' or 'lstm'")

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        if isinstance(h, tuple):
            h = h[0]
        return h[-1]


# -----------------------------
# 5. Prepare dataset for DataLoader
# -----------------------------

class TripleDataset(Dataset):
    def __init__(self, X_queries, X_rels, X_irrels):
        self.X_queries = X_queries    # list of N tensors (seq_len, embed_dim)
        self.X_rels = X_rels          # list of N lists of 10 tensors (seq_len, embed_dim)
        self.X_irrels = X_irrels      # list of N lists of 10 tensors (seq_len, embed_dim)
        self.n = len(X_queries)

    def __len__(self):
        return self.n * 10

    def __getitem__(self, idx):
        triple_idx = idx // 10
        doc_idx = idx % 10

        qry = self.X_queries[triple_idx]              # (q_seq_len, embed_dim)
        rel = self.X_rels[triple_idx][doc_idx]        # (rel_seq_len, embed_dim)
        irrel = self.X_irrels[triple_idx][doc_idx]    # (irrel_seq_len, embed_dim)

        return qry, rel, irrel
    
def collate_fn(batch):
    q_seqs, r_seqs, i_seqs = zip(*batch)
    q_lens = [x.shape[0] for x in q_seqs]
    r_lens = [x.shape[0] for x in r_seqs]
    i_lens = [x.shape[0] for x in i_seqs]
    q_padded = pad_sequence(q_seqs, batch_first=True)
    r_padded = pad_sequence(r_seqs, batch_first=True)
    i_padded = pad_sequence(i_seqs, batch_first=True)
    return q_padded, r_padded, i_padded, q_lens, r_lens, i_lens


# Create the dataset and dataloader for batching
batch_size = 32
dataset = TripleDataset(query_embeds, rel_doc_embeds, irrel_doc_embeds)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# -----------------------------
# 6. Train the model (change hyperparameters as needed)
# -----------------------------

# embed_dim defined in section 3.

# Define hyperparameters
hidden_dim = 128  # Dimension of the hidden state in RNNs (GRU/LSTM - can be adjusted)
margin = 0.2  # Margin for triplet loss (can be adjusted)
distance_function = cosine_distance  # Choose distance function (cosine_distance, euclidean_distance, manhattan_distance, squared_euclidean_distance, chebyshev_distance, minkowski_distance)

qry_tower = QueryTower(embed_dim, hidden_dim)
doc_tower = DocTower(embed_dim, hidden_dim)

optimizer = torch.optim.Adam(list(qry_tower.parameters()) + list(doc_tower.parameters()), lr=1e-3)
num_epochs = 5  # Set as needed

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        qry_embeds, rel_embeds, irrel_embeds, q_lens, r_lens, i_lens = batch
        # qry_embeds: (batch, q_seq_len, embed_dim)
        # rel_embeds: (batch, r_seq_len, embed_dim)
        # irrel_embeds: (batch, i_seq_len, embed_dim)

        # Mask: keep items where rel doc is *not* all zeros
        # mask shape: (batch,)
        mask = ~torch.all(rel_embeds == 0, dim=(1,2))

        # If mask is all False, skip batch (shouldn't happen)
        if mask.sum() == 0:
            continue

        # Only keep non-padded triples
        qry_embeds    = qry_embeds[mask]
        rel_embeds    = rel_embeds[mask]
        irrel_embeds  = irrel_embeds[mask]
        q_lens        = [q_lens[i] for i in range(len(mask)) if mask[i]]
        r_lens        = [r_lens[i] for i in range(len(mask)) if mask[i]]
        i_lens        = [i_lens[i] for i in range(len(mask)) if mask[i]]

        qry_vecs    = qry_tower(qry_embeds, q_lens)
        rel_vecs    = doc_tower(rel_embeds, r_lens)
        irrel_vecs  = doc_tower(irrel_embeds, i_lens)

        loss = triplet_loss_function(
            qry_vecs, rel_vecs, irrel_vecs,
            distance_function=distance_function,
            margin=margin
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss / len(dataloader):.4f}")


# Save final model after all epochs are done
torch.save({
    'epoch': num_epochs,
    'qry_tower_state_dict': qry_tower.state_dict(),
    'doc_tower_state_dict': doc_tower.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,  # from last epoch
}, "twotower_final.pt")