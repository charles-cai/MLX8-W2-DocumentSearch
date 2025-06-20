import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, rnn_type='gru', bidirectional=False):
        super().__init__()
        if rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

    def forward(self, x, lengths=None):
        # x: (batch, seq_len, embedding_dim)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            output, hidden = self.rnn(x)
            if isinstance(hidden, tuple):  # LSTM
                hidden = hidden[0]
            return hidden[-1]
        else:
            output, hidden = self.rnn(x)
            if isinstance(hidden, tuple):  # LSTM
                hidden = hidden[0]
            return hidden[-1]

class ParallelEncoders(nn.Module):
    """
    Query embeddings go through the query encoder.
    Relevant and irrelevant document embeddings go through the document encoder.
    """
    def __init__(self, embedding_dim, hidden_dim, rnn_type='gru', bidirectional=False):
        super().__init__()
        self.query_encoder = RNNEncoder(embedding_dim, hidden_dim, rnn_type, bidirectional)
        self.doc_encoder = RNNEncoder(embedding_dim, hidden_dim, rnn_type, bidirectional)

    def forward(self, query_emb, doc_emb, query_lengths=None, doc_lengths=None):
        # query_emb: (batch, seq_len, embedding_dim)
        # doc_emb: (batch, seq_len, embedding_dim)
        query_vec = self.query_encoder(query_emb, query_lengths)
        doc_vec = self.doc_encoder(doc_emb, doc_lengths)
        return query_vec, doc_vec

    def encode_query(self, query_emb, query_lengths=None):
        # For encoding queries only
        return self.query_encoder(query_emb, query_lengths)

    def encode_doc(self, doc_emb, doc_lengths=None):
        # For encoding relevant or irrelevant docs
        return self.doc_encoder(doc_emb, doc_lengths)

if __name__ == "__main__":
    # Test the parallel encoders with dummy data
    batch_size = 2
    seq_len = 5
    embedding_dim = 8
    hidden_dim = 16

    # Dummy embedded queries and docs
    query_emb = torch.randn(batch_size, seq_len, embedding_dim)
    doc_emb = torch.randn(batch_size, seq_len, embedding_dim)

    model = ParallelEncoders(embedding_dim, hidden_dim, rnn_type='gru', bidirectional=False)
    query_vec, doc_vec = model(query_emb, doc_emb)

    print("Query vector shape:", query_vec.shape)  # Should be (batch_size, hidden_dim)
    print("Doc vector shape:", doc_vec.shape)      # Should be (batch_size, hidden_dim)
    print("Query vector:", query_vec)
    print("Doc vector:", doc_vec)
