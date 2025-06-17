import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors

def load_word2vec(w2v_path='GoogleNews-vectors-negative300.bin', freeze=True):
    """
    Load pre-trained Word2Vec model and build embedding matrix.
    
    Args:
        w2v_path: Path to Word2Vec model file
        freeze: Whether to freeze the embedding weights during training
    
    Returns:
        tuple: (embedding_layer, vocab_dict, vocab_size, embedding_dim)
    """
    # Load pre-trained Word2Vec
    w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    
    vocab = w2v.key_to_index  # word -> idx
    vocab_size = len(vocab)
    emb_dim = w2v.vector_size  # 300
    
    # Build embedding weight tensor (words not in w2v get zero vector)
    weight = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    for word, idx in vocab.items():
        weight[idx] = w2v[word]
        
    # PyTorch embedding layer
    embedding = nn.Embedding.from_pretrained(
        torch.from_numpy(weight), 
        freeze=freeze
    )
    
    return embedding, vocab, vocab_size, emb_dim

# 2) Two-Tower RNN model
class TwoTowerRNN(nn.Module):
    def __init__(self, emb_layer, hidden_dim=256, num_layers=1, bidirectional=True, dropout=0.2):
        super().__init__()
        self.emb = emb_layer
        self.rnn_q = nn.LSTM(
            input_size=emb_dim, hidden_size=hidden_dim, 
            num_layers=num_layers, bidirectional=bidirectional, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.rnn_d = nn.LSTM(
            input_size=emb_dim, hidden_size=hidden_dim, 
            num_layers=num_layers, bidirectional=bidirectional, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

    def encode(self, ids, lengths, rnn):
        """
        ids: LongTensor (batch_size, seq_len)
        lengths: LongTensor (batch_size,)
        """
        x = self.emb(ids)  # (bsz, L, emb_dim)
        # pack padded
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = rnn(packed)
        # h_n: (num_layers * num_directions, bsz, hidden_dim)
        if self.bidirectional:
            # take last layer's forward and backward
            # forward: h_n[-2], backward: h_n[-1]
            h_f = h_n[-2]
            h_b = h_n[-1]
            rep = torch.cat([h_f, h_b], dim=-1)  # (bsz, 2*hidden_dim)
        else:
            rep = h_n[-1]  # (bsz, hidden_dim)
        return rep

    def forward(self, q_ids, q_len, pos_ids, pos_len, neg_ids, neg_len):
        # Encode
        q_emb = self.encode(q_ids, q_len, self.rnn_q)            # (bsz, H)
        pos_emb = self.encode(pos_ids, pos_len, self.rnn_d)      # (bsz * n_pos, H)
        neg_emb = self.encode(neg_ids, neg_len, self.rnn_d)      # (bsz * n_neg, H)

        # reshape for similarity
        bsz, n_pos = q_emb.size(0), pos_len.size(1)
        n_neg = neg_len.size(1)
        pos_emb = pos_emb.view(bsz, n_pos, -1)
        neg_emb = neg_emb.view(bsz, n_neg, -1)

        # dot-product similarity
        q = q_emb.unsqueeze(1)           # (bsz, 1, H)
        pos_scores = torch.bmm(q, pos_emb.transpose(1,2)).squeeze(1)
        neg_scores = torch.bmm(q, neg_emb.transpose(1,2)).squeeze(1)
        # (bsz, n_pos), (bsz, n_neg)
        return pos_scores, neg_scores


# 3) Dataset & CollateFn for (query, 10×pos, 10×neg)
class TripletTextDataset(Dataset):
    def __init__(self, queries, pos_docs, neg_docs, word2idx):
        assert len(queries) == len(pos_docs) == len(neg_docs)
        self.queries, self.pos_docs, self.neg_docs = queries, pos_docs, neg_docs
        self.w2i = word2idx

    def __len__(self):
        return len(self.queries)

    def text_to_ids(self, text):
        # simple whitespace tokenize; map OOV→0
        return torch.LongTensor([self.w2i.get(t, 0) for t in text.split()])

    def __getitem__(self, idx):
        q = self.text_to_ids(self.queries[idx])
        pos = [self.text_to_ids(d) for d in self.pos_docs[idx]]
        neg = [self.text_to_ids(d) for d in self.neg_docs[idx]]
        return q, pos, neg

def collate_fn(batch):
    """
    batch: list of (q, [p1..p10], [n1..n10])
    Returns padded tensors and length metadata.
    """
    bsz = len(batch)
    n_pos, n_neg = len(batch[0][1]), len(batch[0][2])

    # flatten all sequences
    all_q, all_pos, all_neg = [], [], []
    q_lens, pos_lens, neg_lens = [], [], []

    for q, pos_list, neg_list in batch:
        all_q.append(q); q_lens.append(len(q))
        for p in pos_list:
            all_pos.append(p); pos_lens.append(len(p))
        for n in neg_list:
            all_neg.append(n); neg_lens.append(len(n))

    # pad
    q_ids = nn.utils.rnn.pad_sequence(all_q, batch_first=True, padding_value=0)
    pos_ids = nn.utils.rnn.pad_sequence(all_pos, batch_first=True, padding_value=0)
    neg_ids = nn.utils.rnn.pad_sequence(all_neg, batch_first=True, padding_value=0)

    # reshape lengths to (bsz, n_pos) and (bsz, n_neg)
    pos_lens = torch.LongTensor(pos_lens).view(bsz, n_pos)
    neg_lens = torch.LongTensor(neg_lens).view(bsz, n_neg)

    return (
        q_ids, torch.LongTensor(q_lens),
        pos_ids, pos_lens,
        neg_ids, neg_lens
    )


# 4) Training loop
def train(model, loader, optimizer, criterion, device, epochs=3):
    """
    Training loop for the Two-Tower RNN model.
    
    Args:
        model: TwoTowerRNN model
        loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss criterion
        device: Device to train on
        epochs: Number of training epochs
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for q_ids, q_len, p_ids, p_len, n_ids, n_len in loader:
            # to device
            q_ids, q_len = q_ids.to(device), q_len.to(device)
            p_ids, p_len = p_ids.to(device), p_len.to(device)
            n_ids, n_len = n_ids.to(device), n_len.to(device)

            pos_scores, neg_scores = model(q_ids, q_len, p_ids, p_len, n_ids, n_len)
            # pos_scores: (bsz, n_pos), neg_scores: (bsz, n_neg)

            # expand and flatten for every pos×neg
            p = pos_scores.unsqueeze(2)  # (bsz, n_pos, 1)
            n = neg_scores.unsqueeze(1)  # (bsz, 1, n_neg)
            p_flat = p.expand(-1, -1, n.size(2)).reshape(-1)
            n_flat = n.expand(-1, p.size(1), -1).reshape(-1)
            target = torch.ones_like(p_flat, device=device)

            loss = criterion(p_flat, n_flat, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

def main():
    """
    Main function to initialize model and run training.
    """
    # Load Word2Vec and initialize embedding
    embedding, vocab, vocab_size, emb_dim = load_word2vec()
    
    # Initialize model and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TwoTowerRNN(embedding).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    margin = 1.0
    criterion = nn.MarginRankingLoss(margin=margin)

    # assume lists: queries, pos_docs, neg_docs already defined
    ds = TripletTextDataset(queries, pos_docs, neg_docs, vocab)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # Run training
    train(model, loader, optimizer, criterion, device, epochs=3)

if __name__ == "__main__":
    main()
