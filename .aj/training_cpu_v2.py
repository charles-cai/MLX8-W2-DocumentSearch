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
import re
from tqdm import tqdm

# -----------------------------
# 1. Load Vocabulary & Pre-trained Embeddings
# -----------------------------
with open("vocab_new.json", "r", encoding="utf-8") as f:
    word_to_ix = json.load(f)

ix_to_word = {int(i): w for w, i in word_to_ix.items()}
vocab_size = len(word_to_ix)

embed_dim = 200  
state = torch.load("text8_cbow_embeddings.pth", map_location='cpu')  # Shape: [vocab_size, embed_dim]
embeddings = state["embeddings.weight"] 

assert embeddings.shape[0] == vocab_size, "Vocab size mismatch!"

# -----------------------------
# 2. Load MS MARCO V1.1 training dataset & create triples + selected passages
# -----------------------------

# This will stream the data, you don't have to download the full file
dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")  # or "validation"


# 1. Build passage pool
passage_to_idx = {}
idx_to_passage = []

for row in tqdm(dataset, desc="Building passage pool..."):
    for p in row['passages']['passage_text']:
        if p not in passage_to_idx:
            passage_to_idx[p] = len(idx_to_passage)
            idx_to_passage.append(p)
num_passages = len(idx_to_passage)

# 2. Create triples and selected passages
triples = []
selected_passages = []

for row in tqdm(dataset, desc="Creating triples..."):
    query = row['query']
    passage_texts = row['passages']['passage_text']
    is_selected = row['passages']['is_selected']

    num_rels = len(passage_texts)
    if num_rels == 0:
        continue  # No passages for this query

    # Gold passage: any with is_selected==1
    gold_idxs = [i for i, flag in enumerate(is_selected) if flag == 1]
    if gold_idxs:
        selected_passages.append(passage_texts[gold_idxs[0]])
    else:
        selected_passages.append(None)

    # Relevant docs: all passages for this query (usually 1-10, can be 0, already filtered)
    relevant_passages = passage_texts

    # Build mask to exclude relevant passages
    rel_indices = [passage_to_idx[p] for p in passage_texts]
    mask = np.ones(num_passages, dtype=bool)
    mask[rel_indices] = False

    # Sample matching number of irrels
    if mask.sum() >= num_rels:
        irrel_indices = np.random.choice(np.where(mask)[0], num_rels, replace=False)
    else:
        irrel_indices = np.random.choice(np.where(mask)[0], num_rels, replace=True)
    irrelevant_passages = [idx_to_passage[i] for i in irrel_indices]

    triples.append((query, relevant_passages, irrelevant_passages))

# Save
with open("triples_full.pkl", "wb") as f:
    pickle.dump(triples, f)

with open("selected_passages.pkl", "wb") as g:
    pickle.dump(selected_passages, g)

# -------------------------------
# 2b. Load triples & selected docs from file
# -------------------------------

# Load triples
with open("triples_full.pkl", "rb") as f:
    triples = pickle.load(f)

with open("selected_passages.pkl", "rb") as g:
    selected_passages= pickle.load(g)


# -------------------------------
# 3. Tokenize triples
# -------------------------------

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]+', '', text)
    return text.split()

tokenized_triples = []
for query, rel_docs, irrel_docs in tqdm(triples, desc="Tokenizing triples"):
    tokenized_query = preprocess(query)
    tokenized_rels = [preprocess(doc) for doc in rel_docs]
    tokenized_irrels = [preprocess(doc) for doc in irrel_docs]
    tokenized_triples.append((tokenized_query, tokenized_rels, tokenized_irrels))

# -----------------------------
# 4. CBOW Model
# -----------------------------
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).mean(dim=1)
        return self.linear(embeds)

cbow_model = CBOW(vocab_size, embed_dim)
cbow_model.embeddings.weight.data.copy_(embeddings)
cbow_model.embeddings.weight.requires_grad = False 


# -------------------------------
# 5. Embed queries, rel and irrel documents using pre-trained CBOW model
# -------------------------------

device = torch.device("cpu")
torch.set_num_threads(12)

batch_size = 1024 # adjust to taste; big enough to speed up, small enough to never threaten RAM

def process_and_save_embeddings(
    tokenized_triples, word_to_ix, cbow_model, 
    query_embeds_path, rel_doc_embeds_path, irrel_doc_embeds_path,
    batch_size=64
):
    query_embeds_batch = []
    rel_doc_embeds_batch = []
    irrel_doc_embeds_batch = []
    
    for i, (tokenized_query, tokenized_rels, tokenized_irrels) in enumerate(
        tqdm(tokenized_triples, desc="CBOW embedding + streaming", total=len(tokenized_triples))
    ):
        # Query: embeddings per token
        q_ids = [word_to_ix[t] for t in tokenized_query if t in word_to_ix]
        if q_ids:
            with torch.no_grad():
                q_vecs = cbow_model.embeddings(torch.tensor(q_ids))
            query_embeds_batch.append(q_vecs)  # [q_len, embed_dim]
        else:
            query_embeds_batch.append(torch.zeros(1, cbow_model.embeddings.embedding_dim))
        
        # Relevant docs: list of (doc_len, embed_dim)
        rel_embs = []
        for doc_tokens in tokenized_rels:
            doc_ids = [word_to_ix[t] for t in doc_tokens if t in word_to_ix]
            if doc_ids:
                with torch.no_grad():
                    doc_vecs = cbow_model.embeddings(torch.tensor(doc_ids))
                rel_embs.append(doc_vecs)
            else:
                rel_embs.append(torch.zeros(1, cbow_model.embeddings.embedding_dim))
        rel_doc_embeds_batch.append(rel_embs)

        # Irrelevant docs: list of (doc_len, embed_dim)
        irrel_embs = []
        for doc_tokens in tokenized_irrels:
            doc_ids = [word_to_ix[t] for t in doc_tokens if t in word_to_ix]
            if doc_ids:
                with torch.no_grad():
                    doc_vecs = cbow_model.embeddings(torch.tensor(doc_ids))
                irrel_embs.append(doc_vecs)
            else:
                irrel_embs.append(torch.zeros(1, cbow_model.embeddings.embedding_dim))
        irrel_doc_embeds_batch.append(irrel_embs)

        # Save every batch_size triples
        if (i + 1) % batch_size == 0 or (i + 1) == len(tokenized_triples):
            with open(query_embeds_path, 'ab') as fq:
                pickle.dump(query_embeds_batch, fq)
            with open(rel_doc_embeds_path, 'ab') as fr:
                pickle.dump(rel_doc_embeds_batch, fr)
            with open(irrel_doc_embeds_path, 'ab') as fi:
                pickle.dump(irrel_doc_embeds_batch, fi)

            # Free RAM
            query_embeds_batch.clear()
            rel_doc_embeds_batch.clear()
            irrel_doc_embeds_batch.clear()

# Usage:
process_and_save_embeddings(
    tokenized_triples, word_to_ix, cbow_model,
    "query_embeds.pkl", "rel_doc_embeds.pkl", "irrel_doc_embeds.pkl",
    batch_size=1024
)

# -----------------------------
# 6. Load Embeddings
# -----------------------------

def load_all_batches(path):
    all_data = []
    with open(path, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
                all_data.extend(batch)  # for lists of tensors, this flattens batches
            except EOFError:
                break
    return all_data

# Load them all
query_embeds = load_all_batches("query_embeds.pkl")           # list of [query_len, embed_dim] tensors
rel_doc_embeds = load_all_batches("rel_doc_embeds.pkl")       # list of lists: each is [num_docs] of [doc_len, embed_dim] tensors
irrel_doc_embeds = load_all_batches("irrel_doc_embeds.pkl")


# -----------------------------
# 7. Define distance function & Triplet loss function
# -----------------------------

# Cosine similarity for calculation of cosine distance
def cosine_similarity(x, y):
    return F.cosine_similarity(x, y, dim=1)

# Triplet loss function - will compute the loss for a batch of triplets
def triplet_loss_function(query, relevant_doc, irrelevant_doc, distance_function, margin):
    rel_dist = distance_function(query, relevant_doc)         # (batch,)
    irrel_dist = distance_function(query, irrelevant_doc)     # (batch,)
    triplet_loss = torch.relu(rel_dist - irrel_dist + margin)
    return triplet_loss.mean()                                # Average over batch


# -----------------------------
# 8. Define Two Tower Model (QueryTower and DocTower)
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
# 6. Prepare dataset for DataLoader
# -----------------------------

class TripleDataset(Dataset):
    def __init__(self, X_queries, X_rels, X_irrels):
        self.X_queries = X_queries    # list of N_query tensors
        self.X_rels = X_rels          # list of lists (variable-length)
        self.X_irrels = X_irrels      # list of lists (same variable-length)
        self.pairs = []
        for i in range(len(self.X_queries)):
            n = len(self.X_rels[i])
            for j in range(n):
                self.pairs.append((i, j))  # (query idx, doc pair idx)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q_idx, d_idx = self.pairs[idx]
        qry = self.X_queries[q_idx]
        rel = self.X_rels[q_idx][d_idx]
        irrel = self.X_irrels[q_idx][d_idx]
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
batch_size = 64
dataset = TripleDataset(query_embeds, rel_doc_embeds, irrel_doc_embeds)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# -----------------------------
# 9. Load MS MARCO V1.1 validation dataset, process into triples and tokenize & embed
# -----------------------------

# Load MS MARCO validation split
val_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

# 1. Build passage pool (same as training)
passage_to_idx = {}
idx_to_passage = []
for row in tqdm(val_dataset, desc="Building passage pool..."):
    for p in row['passages']['passage_text']:
        if p not in passage_to_idx:
            passage_to_idx[p] = len(idx_to_passage)
            idx_to_passage.append(p)
num_passages = len(idx_to_passage)

# 2. Create triples and selected passages
triples_val = []
selected_passages_val = []

for row in tqdm(val_dataset, desc="Creating validation triples..."):
    query = row['query']
    passage_texts = row['passages']['passage_text']
    is_selected = row['passages']['is_selected']

    num_rels = len(passage_texts)
    if num_rels == 0:
        continue  # No passages for this query

    # Gold passage: any with is_selected==1
    gold_idxs = [i for i, flag in enumerate(is_selected) if flag == 1]
    if gold_idxs:
        selected_passages_val.append(passage_texts[gold_idxs[0]])
    else:
        selected_passages_val.append(None)

    # Relevant docs: all passages for this query
    relevant_passages = passage_texts

    # Build mask to exclude relevant passages
    rel_indices = [passage_to_idx[p] for p in passage_texts]
    mask = np.ones(num_passages, dtype=bool)
    mask[rel_indices] = False

    # Sample matching number of irrels
    if mask.sum() >= num_rels:
        irrel_indices = np.random.choice(np.where(mask)[0], num_rels, replace=False)
    else:
        irrel_indices = np.random.choice(np.where(mask)[0], num_rels, replace=True)
    irrelevant_passages = [idx_to_passage[i] for i in irrel_indices]

    triples_val.append((query, relevant_passages, irrelevant_passages))

# Save
with open("triples_val.pkl", "wb") as f:
    pickle.dump(triples_val, f)
with open("selected_passages_val.pkl", "wb") as g:
    pickle.dump(selected_passages_val, g)


# Load triples & selected docs
with open("triples_val.pkl", "rb") as f:
    triples_val = pickle.load(f)
with open("selected_passages_val.pkl", "rb") as g:
    selected_passages_val = pickle.load(g)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]+', '', text)
    return text.split()

tokenized_triples_val = []
for query, rel_docs, irrel_docs in tqdm(triples, desc="Tokenizing triples"):
    tokenized_query = preprocess(query)
    tokenized_rels = [preprocess(doc) for doc in rel_docs]
    tokenized_irrels = [preprocess(doc) for doc in irrel_docs]
    tokenized_triples_val.append((tokenized_query, tokenized_rels, tokenized_irrels))


def process_and_save_embeddings(
    tokenized_triples_val, word_to_ix, cbow_model, 
    query_embeds_path, rel_doc_embeds_path, irrel_doc_embeds_path,
    batch_size=64
):
    query_embeds_batch = []
    rel_doc_embeds_batch = []
    irrel_doc_embeds_batch = []
    
    for i, (tokenized_query, tokenized_rels, tokenized_irrels) in enumerate(
        tqdm(tokenized_triples_val, desc="CBOW embedding + streaming", total=len(tokenized_triples_val))
    ):
        # Query: embeddings per token
        q_ids = [word_to_ix[t] for t in tokenized_query if t in word_to_ix]
        if q_ids:
            with torch.no_grad():
                q_vecs = cbow_model.embeddings(torch.tensor(q_ids))
            query_embeds_batch.append(q_vecs)  # shape: [query_len, embed_dim]
        else:
            query_embeds_batch.append(torch.zeros(1, cbow_model.embeddings.embedding_dim))
        
        # Relevant docs: list of (doc_len, embed_dim)
        rel_embs = []
        for doc_tokens in tokenized_rels:
            doc_ids = [word_to_ix[t] for t in doc_tokens if t in word_to_ix]
            if doc_ids:
                with torch.no_grad():
                    doc_vecs = cbow_model.embeddings(torch.tensor(doc_ids))
                rel_embs.append(doc_vecs)
            else:
                rel_embs.append(torch.zeros(1, cbow_model.embeddings.embedding_dim))
        rel_doc_embeds_batch.append(rel_embs)

        # Irrelevant docs: list of (doc_len, embed_dim)
        irrel_embs = []
        for doc_tokens in tokenized_irrels:
            doc_ids = [word_to_ix[t] for t in doc_tokens if t in word_to_ix]
            if doc_ids:
                with torch.no_grad():
                    doc_vecs = cbow_model.embeddings(torch.tensor(doc_ids))
                irrel_embs.append(doc_vecs)
            else:
                irrel_embs.append(torch.zeros(1, cbow_model.embeddings.embedding_dim))
        irrel_doc_embeds_batch.append(irrel_embs)

        # Save every batch_size triples
        if (i + 1) % batch_size == 0 or (i + 1) == len(tokenized_triples):
            with open(query_embeds_path, 'ab') as fq:
                pickle.dump(query_embeds_batch, fq)
            with open(rel_doc_embeds_path, 'ab') as fr:
                pickle.dump(rel_doc_embeds_batch, fr)
            with open(irrel_doc_embeds_path, 'ab') as fi:
                pickle.dump(irrel_doc_embeds_batch, fi)

            query_embeds_batch.clear()
            rel_doc_embeds_batch.clear()
            irrel_doc_embeds_batch.clear()


process_and_save_embeddings(
    tokenized_triples_val, word_to_ix, cbow_model,
    "query_embeds_val.pkl", "rel_doc_embeds_val.pkl", "irrel_doc_embeds_val.pkl",
    batch_size=64
)

# Load them all
query_embeds_val = load_all_batches("query_embeds_val.pkl")           # list of [query_len, embed_dim] tensors
rel_doc_embeds_val = load_all_batches("rel_doc_embeds_val.pkl")       # list of lists: each is [num_docs] of [doc_len, embed_dim] tensors
irrel_doc_embeds_val = load_all_batches("irrel_doc_embeds_val.pkl")


val_data = []
# Ensure lengths match (or handle indexing errors)
for i in range(len(query_embeds_val)):
    q_embed = query_embeds_val[i]                       # [q_len, embed_dim]
    rel_embed = rel_doc_embeds_val[i][0]                # [rel_len, embed_dim]; use the first relevant doc
    irrel_embed = irrel_doc_embeds_val[i][0]            # [irrel_len, embed_dim]; use the first irrelevant doc
    val_data.append((q_embed, rel_embed, [irrel_embed])) 


# -----------------------------
# 10. Create evaluation for the model using Recall@K
# -----------------------------

def evaluate_model(
    qry_tower, doc_tower, val_data, selected_passages_val, rel_doc_texts_val, irrel_doc_texts_val,
    distance_fn, K=1, device="cpu"
):
    qry_tower.eval()
    doc_tower.eval()
    num_correct = 0
    total = 0

    with torch.no_grad():
        for i, (q_embed, rel_embeds, irrel_embeds) in enumerate(val_data):
            # Skip queries with no gold passage
            gold_text = selected_passages_val[i]
            if gold_text is None:
                continue

            # Build all candidate docs and their texts
            candidate_embeds = rel_embeds + irrel_embeds
            candidate_texts = rel_doc_texts_val[i] + irrel_doc_texts_val[i]  # rels + irrels

            # Find gold doc index among candidates
            try:
                gold_idx = candidate_texts.index(gold_text)
            except ValueError:
                # Gold doc not among candidates
                continue

            all_doc_tensors = [doc.to(device) for doc in candidate_embeds]
            doc_lens = [doc.shape[0] for doc in all_doc_tensors]
            padded_docs = torch.nn.utils.rnn.pad_sequence(all_doc_tensors, batch_first=True)

            # Encode query
            q_input = q_embed.unsqueeze(0).to(device)
            q_len = [q_embed.shape[0]]
            q_vec = qry_tower(q_input, q_len)

            # Encode all docs in batch
            d_vecs = doc_tower(padded_docs, doc_lens)

            # Compute distances (query vs. all docs)
            sims = distance_fn(q_vec.repeat(len(doc_lens), 1), d_vecs)  # (num_candidates,)
            sorted_indices = torch.argsort(sims, descending=True)  # Sort by highest similarity

            # Recall@K: Is gold doc in top K?
            if gold_idx in sorted_indices[:K]:
                num_correct += 1
            total += 1

    recall_at_k = num_correct / total if total > 0 else 0.0
    return recall_at_k


rel_doc_texts_val = [rels for _, rels, _ in triples_val]
irrel_doc_texts_val = [irrels for _, _, irrels in triples_val]


# -----------------------------
# 12. Train the model
# -----------------------------

# Define hyperparameters
hidden_dim = 128
margin = 0.2
distance_function = cosine_similarity

qry_tower = QueryTower(embed_dim, hidden_dim, rnn_type='gru')  # or 'lstm'
doc_tower = DocTower(embed_dim, hidden_dim, rnn_type='gru')  # or 'lstm'

optimizer = torch.optim.Adam(list(qry_tower.parameters()) + list(doc_tower.parameters()), lr=1e-3)
num_epochs = 10

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
    if (epoch + 1) % 2 == 0:
        recall = evaluate_model(
            qry_tower, doc_tower, val_data,
            selected_passages_val, rel_doc_texts_val, irrel_doc_texts_val,
            cosine_similarity, K=1
        )
        print(f"Recall@1: {recall:.4f}")

    torch.save({
        'epoch': epoch + 1,
        'qry_tower_state_dict': qry_tower.state_dict(),
        'doc_tower_state_dict': doc_tower.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,  # for this epoch
    }, f"two_tower_epoch_{epoch+1}.pt")