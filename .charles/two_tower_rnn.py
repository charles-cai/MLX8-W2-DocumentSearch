#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from dotenv import load_dotenv
from logging_utils import setup_logging
import pandas as pd
import numpy as np
import faiss
import gc
from word2vec_utils import Word2vecUtils

load_dotenv()

class QryTower(torch.nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, output_dim=128):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        # Output will be 2 * hidden_dim due to bidirectional
        self.projection = torch.nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        gru_out, hidden = self.gru(x)
        # Use last output for sequence representation
        last_output = gru_out[:, -1, :]  # (batch_size, 2*hidden_dim)
        emb = self.projection(last_output)
        return F.normalize(emb, p=2, dim=-1)  # L2 normalize

class DocTower(torch.nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, output_dim=128):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        # Output will be 2 * hidden_dim due to bidirectional
        self.projection = torch.nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        gru_out, hidden = self.gru(x)
        # Use last output for sequence representation
        last_output = gru_out[:, -1, :]  # (batch_size, 2*hidden_dim)
        emb = self.projection(last_output)
        return F.normalize(emb, p=2, dim=-1)  # L2 normalize

class TwoTowerRNN:
    def __init__(self):
        self.logger = setup_logging(__name__)

        # Environment variables
        self.MLX_DATASET_OUTPUT_DIR = os.getenv("MLX_DATASET_OUTPUT_DIR", "./.data/processed")
        self.MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "./.data/models")
        
        self.NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 5))
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
        self.LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-3))
        self.MARGIN = float(os.getenv("MARGIN", 0.15))
        self.TOP_K = int(os.getenv("TOP_K", 10))
        self.MAX_TOKEN = int(os.getenv("MAX_TOKEN", 256))

        self.TRIPLES_DATA_PATH_TRAIN = os.path.join(self.MLX_DATASET_OUTPUT_DIR, "train_triples.parquet")
        self.TRIPLES_DATA_PATH_VALIDATION = os.path.join(self.MLX_DATASET_OUTPUT_DIR,  "validation_triples.parquet")
        self.TRIPLES_DATA_PATH_TEST = os.path.join(self.MLX_DATASET_OUTPUT_DIR,  "test_triples.parquet")
    
        self.WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mlx8-week2-document-search")
        self.WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", "two-tower")
        
        self.TWO_TOWER_RNN_MODEL_PATH = os.path.join(self.MODEL_OUTPUT_DIR, "two_tower_rnn.pt")

        # Log environment values
        self.logger.info(f"MODEL_OUTPUT_DIR: {self.MODEL_OUTPUT_DIR}")
        self.logger.info(f"NUM_EPOCHS: {self.NUM_EPOCHS}")
        self.logger.info(f"BATCH_SIZE: {self.BATCH_SIZE}")
        self.logger.info(f"LEARNING_RATE: {self.LEARNING_RATE}")
        self.logger.info(f"MARGIN: {self.MARGIN}")
        self.logger.info(f"MAX_TOKEN: {self.MAX_TOKEN}")
        self.logger.info(f"TRIPLES_DATA_PATH_TRAIN: {self.TRIPLES_DATA_PATH_TRAIN}")
        self.logger.info(f"TRIPLES_DATA_PATH_VALIDATION: {self.TRIPLES_DATA_PATH_VALIDATION}")
        self.logger.info(f"TRIPLES_DATA_PATH_TEST: {self.TRIPLES_DATA_PATH_TEST}")
        self.logger.info(f"WANDB_PROJECT: {self.WANDB_PROJECT}")
        self.logger.info(f"WANDB_RUN_NAME: {self.WANDB_RUN_NAME}")
        self.logger.info(f"TWO_TOWER_RNN_MODEL_PATH: {self.TWO_TOWER_RNN_MODEL_PATH}")

        # Initialize Word2Vec utils
        self.logger.info("Loading Word2Vec model...")
        self.w2v_utils = Word2vecUtils()
        if self.w2v_utils.w2v_model is None:
            self.w2v_utils.load_word2vec()
        self.w2v_model = self.w2v_utils.w2v_model
        self.vector_size = self.w2v_model.vector_size

        self.qry_tower = QryTower(input_dim=self.vector_size)
        self.doc_tower = DocTower(input_dim=self.vector_size)
        
        # Initialize faiss configuration
        self._setup_faiss()

    def _get_seq_embedding(self, row_id, query_id, text):
        """
        Generate sequence embeddings with memory-efficient processing.
        """
        tokens = text.lower().split()
        if len(tokens) > self.MAX_TOKEN:
            self.logger.warning(f"#{row_id} query_id {query_id}: text length {len(tokens)} exceeds MAX_TOKEN {self.MAX_TOKEN}")
            tokens = tokens[:self.MAX_TOKEN]

        # Use generator to avoid creating intermediate list
        vectors = []
        for word in tokens:
            if word in self.w2v_model:
                vectors.append(self.w2v_model[word].astype(np.float32))
        
        if not vectors:
            result = [np.zeros(self.vector_size, dtype=np.float32)]
        else:
            result = vectors
        
        # Clear intermediate data
        del vectors
        return result

    def _generate_batch_embeddings(self, texts, batch_start_idx):
        """
        Generate sequence embeddings for a batch of texts
        """
        batch_embeddings = []
        for i, text in enumerate(texts):
            row_id = batch_start_idx + i
            seq_emb = self._get_seq_embedding(row_id, "batch", text)
            batch_embeddings.append(seq_emb)
        return batch_embeddings

    def _setup_faiss(self):
        """Setup faiss for GPU or CPU usage"""
        self.use_faiss_gpu = torch.cuda.is_available()
        
        if self.use_faiss_gpu:
            try:
                # Check if faiss-gpu is properly installed and can use GPU
                self.faiss_res = faiss.StandardGpuResources()
                self.gpu_id = 0
                self.logger.info("Faiss GPU resources initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize faiss GPU: {e}")
                self.logger.warning("Falling back to CPU faiss")
                self.use_faiss_gpu = False
        else:
            self.logger.info("CUDA not available, using CPU faiss")

    def _create_faiss_index(self, embedding_dim):
        """Create faiss index with GPU/CPU fallback"""
        if self.use_faiss_gpu:
            try:
                index = faiss.IndexFlatIP(embedding_dim)
                index = faiss.index_cpu_to_gpu(self.faiss_res, self.gpu_id, index)
                return index
            except Exception as e:
                self.logger.warning(f"Failed to create GPU index: {e}, falling back to CPU")
                return faiss.IndexFlatIP(embedding_dim)
        else:
            return faiss.IndexFlatIP(embedding_dim)

    def _save_faiss_index(self, index, filepath):
        """Save faiss index with GPU to CPU conversion if needed"""
        if self.use_faiss_gpu:
            try:
                index_cpu = faiss.index_gpu_to_cpu(index)
                faiss.write_index(index_cpu, filepath)
            except Exception as e:
                self.logger.warning(f"Failed to convert GPU index to CPU for saving: {e}")
                # Fallback: create CPU version with same data
                embedding_dim = index.d
                index_cpu = faiss.IndexFlatIP(embedding_dim)
                # Note: This fallback loses the index data, but prevents crashes
                faiss.write_index(index_cpu, filepath)
        else:
            faiss.write_index(index, filepath)

    def train(self):
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.warning(f"Using device: {self.device}")
        self.logger.warning(f"Using faiss on: {'GPU' if self.use_faiss_gpu else 'CPU'}")

        # Move models to device
        self.qry_tower.to(self.device)
        self.doc_tower.to(self.device)

        wandb.init(
            project=self.WANDB_PROJECT,
            name=self.WANDB_RUN_NAME + "_rnn"
        )

        # Log environment values to wandb
        wandb.config.update({
            "MODEL_OUTPUT_DIR": self.MODEL_OUTPUT_DIR,
            "NUM_EPOCHS": self.NUM_EPOCHS,
            "BATCH_SIZE": self.BATCH_SIZE,
            "LEARNING_RATE": self.LEARNING_RATE,
            "MARGIN": self.MARGIN,
            "MAX_TOKEN": self.MAX_TOKEN,
        })

        optimizer = torch.optim.Adam(
            list(self.qry_tower.parameters()) + list(self.doc_tower.parameters()),
            lr=self.LEARNING_RATE
        )

        df_validation = pd.read_parquet(self.TRIPLES_DATA_PATH_VALIDATION)
        df_test = pd.read_parquet(self.TRIPLES_DATA_PATH_TEST)

        # Load training data from parquet
        self.logger.info(f"Loading training data from {self.TRIPLES_DATA_PATH_TRAIN}")
        df = pd.read_parquet(self.TRIPLES_DATA_PATH_TRAIN)

        num_samples = len(df)
        self.logger.warning(f"Loaded {num_samples} samples for training")

        # Calculate and log batch info
        total_batches = (num_samples + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        self.logger.warning(f"Total rows: {num_samples}, Batch size: {self.BATCH_SIZE}, Total batches: {total_batches}")

        def pad_batch(batch_seq):
            batch_tensors = [torch.tensor(s, dtype=torch.float32) for s in batch_seq]
            padded = torch.nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True, padding_value=0.0)
            return padded.to(self.device)

        for epoch in tqdm(range(self.NUM_EPOCHS), desc="Epochs"):
            epoch_loss = 0.0
            for i in tqdm(range(0, num_samples, self.BATCH_SIZE), desc=f"Epoch {epoch+1}/{self.NUM_EPOCHS}", total=total_batches):
                batch_df = df.iloc[i:i+self.BATCH_SIZE]
                
                # Generate embeddings on-the-fly
                batch_qry_seq = self._generate_batch_embeddings(batch_df["query"].values, i)
                batch_pos_seq = self._generate_batch_embeddings(batch_df["positive_doc"].values, i)
                batch_neg_seq = self._generate_batch_embeddings(batch_df["negative_doc"].values, i)

                batch_qry = pad_batch(batch_qry_seq)
                batch_pos = pad_batch(batch_pos_seq)
                batch_neg = pad_batch(batch_neg_seq)

                qry_emb = self.qry_tower(batch_qry)
                pos_emb = self.doc_tower(batch_pos)
                neg_emb = self.doc_tower(batch_neg)

                # Cosine similarities
                s_pos = (qry_emb * pos_emb).sum(dim=-1)
                s_neg = (qry_emb * neg_emb).sum(dim=-1)
                loss = torch.clamp(self.MARGIN - (s_pos - s_neg), min=0.0).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_qry.size(0)

                # wandb logging per batch
                wandb.log({"batch_loss": loss.item(), "epoch": epoch+1})

                # Clear batch data to free memory
                del batch_qry_seq, batch_pos_seq, batch_neg_seq
                del batch_qry, batch_pos, batch_neg
                gc.collect()

            avg_loss = epoch_loss / num_samples
            self.logger.info(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch+1})
            
            # Fixed evaluation call - use correct parameters
            metrics = self.compute_precision_recall_at_k(epoch, self.qry_tower, self.doc_tower, self.TOP_K, df_validation, split="validation")
            wandb.log(metrics)

        # Save model
        os.makedirs(self.MODEL_OUTPUT_DIR, exist_ok=True)
        torch.save({
            "qry_tower_state_dict": self.qry_tower.state_dict(),
            "doc_tower_state_dict": self.doc_tower.state_dict(),
        }, self.TWO_TOWER_RNN_MODEL_PATH)
        self.logger.info(f"Model saved to {self.TWO_TOWER_RNN_MODEL_PATH}")
        wandb.save(self.TWO_TOWER_RNN_MODEL_PATH)
        wandb.finish()

    def compute_precision_recall_at_k(self, epoch, qry_tower, doc_tower, top_k, df, split):
        self.logger.info(f"Computing precision@{top_k} and recall@{top_k} for {split} split")

        def pad_batch(batch_seq):
            batch_tensors = [torch.tensor(s, dtype=torch.float32) for s in batch_seq]
            padded = torch.nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True, padding_value=0.0)
            return padded.to(self.device)

        # Transform all query and doc embeddings using the towers
        with torch.no_grad():
            qry_tower_emb_list = []
            for i in tqdm(range(0, len(df), self.BATCH_SIZE), desc=f"Embedding {split} queries"):
                batch_df = df.iloc[i:i+self.BATCH_SIZE]
                batch_qry_seq = self._generate_batch_embeddings(batch_df["query"].values, i)
                padded_batch = pad_batch(batch_qry_seq)
                batch_emb = qry_tower(padded_batch).cpu().numpy()
                qry_tower_emb_list.append(batch_emb)
                
                # Clear batch data
                del batch_qry_seq, padded_batch
                gc.collect()
            qry_tower_emb = np.vstack(qry_tower_emb_list)

            doc_tower_emb_list = []
            for i in tqdm(range(0, len(df), self.BATCH_SIZE), desc=f"Embedding {split} documents"):
                batch_df = df.iloc[i:i+self.BATCH_SIZE]
                batch_doc_seq = self._generate_batch_embeddings(batch_df["positive_doc"].values, i)
                padded_batch = pad_batch(batch_doc_seq)
                batch_emb = doc_tower(padded_batch).cpu().numpy()
                doc_tower_emb_list.append(batch_emb)
                
                # Clear batch data
                del batch_doc_seq, padded_batch
                gc.collect()
            doc_tower_emb = np.vstack(doc_tower_emb_list)
        
        # Build faiss index using transformed doc embeddings (128-dim)
        embedding_dim = doc_tower_emb.shape[1]
        
        index_doc = self._create_faiss_index(embedding_dim)
        index_doc.add(doc_tower_emb.astype("float32"))
        
        self.logger.info(f"Built document index with {index_doc.ntotal} documents of {embedding_dim} dimensions")
        self.logger.info(f"Using faiss on: {'GPU' if self.use_faiss_gpu else 'CPU'}")

        # Store the index for later use (especially useful for the last epoch)
        vector_store_dir = os.path.join(self.MLX_DATASET_OUTPUT_DIR, "faiss")
        os.makedirs(vector_store_dir, exist_ok=True)
        
        # Save both query and doc indexes with epoch info
        query_index_path = os.path.join(vector_store_dir, f"epoch_{epoch+1}_{split}_query_tower_rnn.index")
        doc_index_path = os.path.join(vector_store_dir, f"epoch_{epoch+1}_{split}_doc_tower_rnn.index")
        
        # Create query index for completeness
        qry_index = self._create_faiss_index(qry_tower_emb.shape[1])
        qry_index.add(qry_tower_emb.astype("float32"))
        
        # Save indexes using helper method
        self._save_faiss_index(qry_index, query_index_path)
        self._save_faiss_index(index_doc, doc_index_path)
        
        # Also save the transformed embeddings as parquet for later use
        df_with_towers = df.copy()
        df_with_towers["query_tower_emb"] = list(qry_tower_emb)
        df_with_towers["positive_doc_tower_emb"] = list(doc_tower_emb)
        
        tower_embeddings_path = os.path.join(self.MLX_DATASET_OUTPUT_DIR, f"epoch_{epoch+1}_{split}_triples_embeddings_tower_rnn.parquet")
        df_with_towers.to_parquet(tower_embeddings_path, index=False)
        
        self.logger.info(f"Saved indexes and embeddings for epoch {epoch+1}:")
        self.logger.info(f"  Query index: {query_index_path}")
        self.logger.info(f"  Doc index: {doc_index_path}")
        self.logger.info(f"  Tower embeddings: {tower_embeddings_path}")

        # Get unique queries and their embeddings
        unique_query_indices = df['query_id'].drop_duplicates().index
        unique_query_embs = qry_tower_emb[unique_query_indices]
        unique_query_ids = df.loc[unique_query_indices, 'query_id'].values

        # Search all unique queries at once
        D, I = index_doc.search(unique_query_embs.astype("float32"), top_k)

        hit_count = 0
        for i, qid in enumerate(unique_query_ids):
            topk_indices = I[i]
            # Check if any retrieved document belongs to the same query_id (positive match)
            hit = any(df.iloc[idx]["query_id"] == qid for idx in topk_indices)
            if hit:
                hit_count += 1

        total_queries = len(unique_query_ids)
        precision_at_k = hit_count / total_queries if total_queries > 0 else 0
        recall_at_k = precision_at_k  # assuming one relevant document per query
        
        self.logger.info(f"Epoch {epoch+1} - {split.upper()} Evaluation:")
        self.logger.info(f"  Precision@{top_k}: {precision_at_k:.4f}")
        self.logger.info(f"  Recall@{top_k}: {recall_at_k:.4f}")
        self.logger.info(f"  Total queries: {total_queries}")
        self.logger.info(f"  Hits: {hit_count}")
        
        return {
            f"{split}_precision_at_{top_k}": precision_at_k, 
            f"{split}_recall_at_{top_k}": recall_at_k,
            f"{split}_total_queries": total_queries,
            f"{split}_hits": hit_count
        }

def main():
    model = TwoTowerRNN()
    model.train()

if __name__ == "__main__":
    main()
