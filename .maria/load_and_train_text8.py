"""
Train a CBOW model on text8 to produce word embeddings.
Saves: text8_embeddings.pt, text8_word_to_index.pkl
Do NOT use this file for retrieval, queries, or triplet loss.
No BERT or transformer code is used or required.
"""

# Requires the 'datasets' library. Install with: pip install datasets

from datasets import load_dataset
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import Dataset, DataLoader
import tqdm

# -------------------------
# Preprocessing Functions
# -------------------------
def build_vocab(words, vocab_size):
    counter = Counter(words)
    top_words = counter.most_common(vocab_size - 1)  # Reserve 0 for <UNK>
    word_to_index = {'<UNK>': 0}
    for idx, (word, _) in enumerate(top_words):
        word_to_index[word] = idx + 1
    return word_to_index

def generate_cbow_pairs(data, window_size):
    pairs = []
    for i in range(window_size, len(data) - window_size):
        context = data[i - window_size : i] + data[i + 1 : i + window_size + 1]
        center = data[i]
        pairs.append((context, center))
    return pairs

def tokenize_with_vocab(text, word_to_index):
    """
    Tokenizes a string using the provided word_to_index mapping.
    Unknown words are mapped to 0 (<UNK>).
    """
    return [word_to_index.get(word, 0) for word in text.split()]

# -------------------------
# Dataset Class
# -------------------------
class CBOWDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context, center = self.pairs[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(center, dtype=torch.long)

# -------------------------
# CBOW Model Definition
# -------------------------
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout=0.5):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        embeds = self.embeddings(context_idxs)
        context_embeds = embeds.mean(dim=1)
        context_embeds = self.dropout(context_embeds)
        return self.linear(context_embeds)

# -------------------------
# Evaluation Function
# -------------------------
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for context, center in loader:
            context = context.to(device)
            center = center.to(device)
            output = model(context)
            loss = criterion(output, center)
            total_loss += loss.item()
    model.train()
    return total_loss / len(loader)

# -------------------------
# Main Training Pipeline
# -------------------------
def main():
    # --- Hyperparameters ---
    embedding_dim = 256
    window_size = 4
    vocab_size = 20000
    batch_size = 512
    epochs = 5
    learning_rate = 0.0005
    dropout = 0.5

    # --- Data Loading and Preprocessing ---
    dataset = load_dataset("afmck/text8")
    words = dataset['train'][0]['text'].split()
    total_words = len(words)
    # If you want to use only a portion, slice here. Otherwise, use all.
    used_words = words  # Use all words
    print(f"✅ Loaded text8 from HuggingFace: {total_words:,} total words.")
    print(f"Using {len(used_words):,} words out of {total_words:,} ({len(used_words)/total_words:.2%})")

    word_to_index = build_vocab(used_words, vocab_size)
    data = [word_to_index.get(word, 0) for word in used_words]
    pairs = generate_cbow_pairs(data, window_size)
    print("Example CBOW pair:")
    for context, center in pairs[:5]:
        print(f"context: {context}, center: {center}")

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and DataLoader ---
    dataset = CBOWDataset(pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))

    # --- Model, Loss, Optimizer, Scheduler ---
    model = CBOWModel(vocab_size=len(word_to_index), embedding_dim=embedding_dim, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # --- Wandb Initialization ---
    try:
        import wandb
        wandb.init(project='cbow-text8', config={
            "embedding_dim": embedding_dim,
            "window_size": window_size,
            "vocab_size": vocab_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "dropout": dropout,
        })
        use_wandb = True
    except Exception as e:
        print(f"wandb not available or failed to initialize: {e}")
        use_wandb = False

    # --- Training Loop ---
    for epoch in range(epochs):
        total_loss = 0
        prgs = tqdm.tqdm(loader, desc=f'Epoch {epoch+1}', leave=False)
        for context, center in prgs:
            context, center = context.to(device, non_blocking=True), center.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, center)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if use_wandb:
                wandb.log({'loss': loss.item()})
        scheduler.step()
        val_loss = evaluate(model, loader, device, criterion)
        print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(loader):.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint
        checkpoint_name = f'cbow_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_name)
        if use_wandb:
            artifact = wandb.Artifact('model-weights', type='model')
            artifact.add_file(checkpoint_name)
            wandb.log_artifact(artifact)

    if use_wandb:
        wandb.finish()

    # --- Save Model and Embeddings ---
    # Only keep the final model and embeddings for downstream use
    torch.save(model.state_dict(), "cbow_model.pt")
    torch.save(model.embeddings.weight.data.cpu(), "text8_embeddings.pt")
    with open("text8_word_to_index.pkl", "wb") as f:
        pickle.dump(word_to_index, f)
    print("✅ Saved cbow_model.pt, text8_embeddings.pt, and text8_word_to_index.pkl")

    # Optionally, remove old epoch checkpoint files to save space
    import os
    for epoch in range(1, epochs + 1):
        fname = f"cbow_epoch{epoch}.pth"
        if os.path.exists(fname):
            os.remove(fname)
            print(f"Deleted old checkpoint: {fname}")

    # --- Preview Embeddings ---
    for word, idx in list(word_to_index.items())[:5]:
        idx_tensor = torch.tensor([idx], device=device)
        embedding = model.embeddings(idx_tensor)
        print(f"Word: {word}, Index: {idx}, Embedding: {embedding}")

if __name__ == "__main__":
    main()

    # --- Example usage and preview (only runs when called directly, not on import) ---
    # --- Load word_to_index mapping ---
    with open("text8_word_to_index.pkl", "rb") as f:
        word_to_index = pickle.load(f)

    # --- Example: Tokenize text using Text8 vocab ---
    raw_text = "this is an example query or document"
    tokenized = tokenize_with_vocab(raw_text, word_to_index)
    print(f"Raw text: {raw_text}")
    print(f"Tokenized indices: {tokenized}")

    # Build index_to_word mapping for decoding indices back to words
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    # Example: decode tokenized indices back to words
    decoded = [index_to_word.get(idx, "<UNK>") for idx in tokenized]
    print(f"Decoded words: {decoded}")

# --- Function to load pretrained embeddings as a frozen or fine-tunable layer ---
def load_pretrained_embedding_layer(embedding_path, vocab_path, freeze=True):
    weights = torch.load(embedding_path)
    with open(vocab_path, "rb") as f:
        word_to_index = pickle.load(f)
    embedding_layer = nn.Embedding.from_pretrained(weights, freeze=freeze)
    return embedding_layer, word_to_index

# --- Streamlit interface for interactive query and top-5 results ---
try:
    import streamlit as st
    import numpy as np  # <-- Add this import

    st.title("Text8 Vocabulary & Embedding Explorer")

    # Load vocab and embeddings
    with open("text8_word_to_index.pkl", "rb") as f:
        word_to_index = pickle.load(f)
    embeddings = torch.load("text8_embeddings.pt")

    st.write("Enter a query below to see its tokenization, embedding average, and top-5 closest vocab words:")

    query = st.text_input("Enter a query:")

    if query:
        tokens = [word_to_index.get(word, 0) for word in query.lower().split()]
        st.write(f"Token indices: {tokens}")
        if tokens:
            vectors = embeddings[tokens].cpu().numpy()
            avg_embedding = vectors.mean(axis=0)
            st.write(f"Average embedding vector (first 10 dims): {avg_embedding[:10]}")

            # Compute cosine similarity to all vocab embeddings
            all_embs = embeddings.cpu().numpy()
            avg_norm = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
            all_norm = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
            sims = all_norm @ avg_norm
            top5_idx = sims.argsort()[::-1][:5]
            idx_to_word = {idx: word for word, idx in word_to_index.items()}
            st.write("Top 5 closest vocab words:")
            for rank, idx in enumerate(top5_idx, 1):
                word = idx_to_word.get(idx, "<UNK>")
                st.write(f"{rank}. {word} (similarity: {sims[idx]:.3f})")
        else:
            st.write("No known tokens in vocab.")

    st.markdown("---")
    st.write("This app only shows embedding info. For retrieval accuracy, use the search or evaluate apps.")

except ImportError:
    print("Streamlit not installed. To use the Streamlit interface, run: pip install streamlit")