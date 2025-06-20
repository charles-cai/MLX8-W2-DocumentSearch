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

load_dotenv()

class QryTower(torch.nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, output_dim=128):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        emb = self.mlp(x)
        return F.normalize(emb, p=2, dim=-1)  # L2 normalize

class DocTower(torch.nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, output_dim=128):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        emb = self.mlp(x)
        return F.normalize(emb, p=2, dim=-1)  # L2 normalize

class TwoTowerTiny:
    def __init__(self):
        self.logger = setup_logging(__name__)

        # Environment variables
        self.MLX_DATASET_OUTPUT_DIR = os.getenv("MLX_DATASET_OUTPUT_DIR", "./.data/processed")
        self.MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "./.data/models")
        
        self.NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 5))
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
        self.LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-3))
        self.MARGIN = float(os.getenv("MARGIN", 0.2))
        self.TOP_K = int(os.getenv("TOP_K", 10))

        self.TRIPLES_EMBEDDINGS_DATA_PATH_TRAIN = os.path.join(self.MLX_DATASET_OUTPUT_DIR, "train_triples_embeddings.parquet")
        self.TRIPLES_EMBEDDINGS_DATA_PATH_VALIDATION = os.path.join(self.MLX_DATASET_OUTPUT_DIR, "validation_triples_embeddings.parquet")
        self.TRIPLES_EMBEDDINGS_DATA_PATH_TEST = os.path.join(self.MLX_DATASET_OUTPUT_DIR, "test_triples_embeddings.parquet")
        
        self.WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mlx8-week2-document-search")
        self.WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", "two-tower")
        
        self.TWO_TOWER_TINY_MODEL_PATH = os.path.join(self.MODEL_OUTPUT_DIR, "two_tower_tiny.pt")

        # Log environment values
        self.logger.info(f"MODEL_OUTPUT_DIR: {self.MODEL_OUTPUT_DIR}")
        self.logger.info(f"NUM_EPOCHS: {self.NUM_EPOCHS}")
        self.logger.info(f"BATCH_SIZE: {self.BATCH_SIZE}")
        self.logger.info(f"LEARNING_RATE: {self.LEARNING_RATE}")
        self.logger.info(f"MARGIN: {self.MARGIN}")
        self.logger.info(f"TRIPLES_EMBEDDINGS_DATA_PATH_TRAIN: {self.TRIPLES_EMBEDDINGS_DATA_PATH_TRAIN}")
        self.logger.info(f"TRIPLES_EMBEDDINGS_DATA_PATH_VALIDATION: {self.TRIPLES_EMBEDDINGS_DATA_PATH_VALIDATION}")
        self.logger.info(f"TRIPLES_EMBEDDINGS_DATA_PATH_TEST: {self.TRIPLES_EMBEDDINGS_DATA_PATH_TEST}")
        self.logger.info(f"WANDB_PROJECT: {self.WANDB_PROJECT}")
        self.logger.info(f"WANDB_RUN_NAME: {self.WANDB_RUN_NAME}")
        self.logger.info(f"TWO_TOWER_TINY_MODEL_PATH: {self.TWO_TOWER_TINY_MODEL_PATH}")

        self.qry_tower = QryTower()
        self.doc_tower = DocTower()
        
        # Initialize faiss configuration
        self._setup_faiss()

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
            name=self.WANDB_RUN_NAME + "_tiny"
        )

        # Log environment values to wandb
        wandb.config.update({
            "MODEL_OUTPUT_DIR": self.MODEL_OUTPUT_DIR,
            "NUM_EPOCHS": self.NUM_EPOCHS,
            "BATCH_SIZE": self.BATCH_SIZE,
            "LEARNING_RATE": self.LEARNING_RATE,
            "MARGIN": self.MARGIN,
        })

        optimizer = torch.optim.Adam(
            list(self.qry_tower.parameters()) + list(self.doc_tower.parameters()),
            lr=self.LEARNING_RATE
        )

        # Load validation data
        df_validation = pd.read_parquet(self.TRIPLES_EMBEDDINGS_DATA_PATH_VALIDATION)
        df_test = pd.read_parquet(self.TRIPLES_EMBEDDINGS_DATA_PATH_TEST)

        # Load training data from parquet
        self.logger.info(f"Loading training data from {self.TRIPLES_EMBEDDINGS_DATA_PATH_TRAIN}")
        df = pd.read_parquet(self.TRIPLES_EMBEDDINGS_DATA_PATH_TRAIN)

        qry = torch.tensor(np.stack(df["query_emb"].values)).float().to(self.device)
        pos = torch.tensor(np.stack(df["positive_doc_emb"].values)).float().to(self.device)
        neg = torch.tensor(np.stack(df["negative_doc_emb"].values)).float().to(self.device)
        num_samples = len(df)
        self.logger.warning(f"Loaded {num_samples} samples for training")

        # Calculate and log batch info
        total_batches = (num_samples + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        self.logger.warning(f"Total rows: {num_samples}, Batch size: {self.BATCH_SIZE}, Total batches: {total_batches}")

        for epoch in tqdm(range(self.NUM_EPOCHS), desc="Epochs"):
            epoch_loss = 0.0
            for i in tqdm(range(0, num_samples, self.BATCH_SIZE), desc=f"Epoch {epoch+1}/{self.NUM_EPOCHS}", total=total_batches):
                batch_qry = qry[i:i+self.BATCH_SIZE]
                batch_pos = pos[i:i+self.BATCH_SIZE]
                batch_neg = neg[i:i+self.BATCH_SIZE]

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

            avg_loss = epoch_loss / num_samples
            self.logger.info(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch+1})
            
            # Evaluation with validation data
            metrics = self.compute_precision_recall_at_k(epoch, self.qry_tower, self.doc_tower, self.TOP_K, df_validation, split="validation")
            wandb.log(metrics)

        # Save model
        os.makedirs(self.MODEL_OUTPUT_DIR, exist_ok=True)
        torch.save({
            "qry_tower_state_dict": self.qry_tower.state_dict(),
            "doc_tower_state_dict": self.doc_tower.state_dict(),
        }, self.TWO_TOWER_TINY_MODEL_PATH)
        self.logger.info(f"Model saved to {self.TWO_TOWER_TINY_MODEL_PATH}")
        wandb.save(self.TWO_TOWER_TINY_MODEL_PATH)
        wandb.finish()

    def compute_precision_recall_at_k(self, epoch, qry_tower, doc_tower, top_k, df, split):
        self.logger.info(f"Computing precision@{top_k} and recall@{top_k} for {split} split")

        query_embeddings = np.stack(df["query_emb"].values)
        doc_embeddings = np.stack(df["positive_doc_emb"].values)
        
        # Transform all query and doc embeddings using the towers
        with torch.no_grad():
            qry_tower_emb_list = []
            for i in tqdm(range(0, len(query_embeddings), self.BATCH_SIZE), desc=f"Embedding {split} queries"):
                batch = torch.tensor(query_embeddings[i:i+self.BATCH_SIZE]).float().to(self.device)
                batch_emb = qry_tower(batch).cpu().numpy()
                qry_tower_emb_list.append(batch_emb)
            qry_tower_emb = np.vstack(qry_tower_emb_list)

            doc_tower_emb_list = []
            for i in tqdm(range(0, len(doc_embeddings), self.BATCH_SIZE), desc=f"Embedding {split} documents"):
                batch = torch.tensor(doc_embeddings[i:i+self.BATCH_SIZE]).float().to(self.device)
                batch_emb = doc_tower(batch).cpu().numpy()
                doc_tower_emb_list.append(batch_emb)
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
        query_index_path = os.path.join(vector_store_dir, f"epoch_{epoch+1}_{split}_query_tower_tiny.index")
        doc_index_path = os.path.join(vector_store_dir, f"epoch_{epoch+1}_{split}_doc_tower_tiny.index")
        
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
        
        tower_embeddings_path = os.path.join(self.MLX_DATASET_OUTPUT_DIR, f"epoch_{epoch+1}_{split}_triples_embeddings_tower_tiny.parquet")
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
    model = TwoTowerTiny()
    model.train()

if __name__ == "__main__":
    main()
