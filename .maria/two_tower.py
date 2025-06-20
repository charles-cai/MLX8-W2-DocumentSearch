import torch
import torch.nn as nn
from parallel_encoders import ParallelEncoders

class TwoTowerModel(nn.Module):
    def __init__(self, embedding_layer, embedding_dim, hidden_dim):
        super().__init__()
        self.query_embedding = embedding_layer
        self.doc_embedding = embedding_layer
        self.query_encoder = ParallelEncoders(embedding_dim, hidden_dim)
        self.doc_encoder = ParallelEncoders(embedding_dim, hidden_dim)

    def forward(self, query_idxs, doc_idxs):
        query_emb = self.query_embedding(query_idxs)
        doc_emb = self.doc_embedding(doc_idxs)
        query_vec, _ = self.query_encoder(query_emb, query_emb)
        _, doc_vec = self.doc_encoder(doc_emb, doc_emb)
        return query_vec, doc_vec