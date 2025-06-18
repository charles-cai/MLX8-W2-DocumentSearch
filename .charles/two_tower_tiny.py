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

load_dotenv()

class TwoTowerTiny:
    def __init__(self):
        self.logger = setup_logging(__name__)

        # Environment variables
        self.MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "./.data/models")
        self.NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 5))
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
        self.LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-3))
        self.MARGIN = float(os.getenv("MARGIN", 0.2))
        self.TRIPLES_EMBEDDINGS_DATA_PATH = os.path.join(
            os.getenv("MLX_DATASET_OUTPUT_DIR", "./.data/processed"),
            f"{os.getenv('MLX_DATASET_TRIPLE_SPLIT', 'train')}_triples_embeddings.parquet"
        )
        self.WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mlx8-week2-document-search")
        self.WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", "two-tower-training")
        self.TWO_TOWER_TINY_MODEL_PATH = os.path.join(self.MODEL_OUTPUT_DIR, "two_tower_tiny.pt")

        # Log environment values
        self.logger.info(f"MODEL_OUTPUT_DIR: {self.MODEL_OUTPUT_DIR}")
        self.logger.info(f"NUM_EPOCHS: {self.NUM_EPOCHS}")
        self.logger.info(f"BATCH_SIZE: {self.BATCH_SIZE}")
        self.logger.info(f"LEARNING_RATE: {self.LEARNING_RATE}")
        self.logger.info(f"MARGIN: {self.MARGIN}")
        self.logger.info(f"TRIPLES_EMBEDDINGS_DATA_PATH: {self.TRIPLES_EMBEDDINGS_DATA_PATH}")
        self.logger.info(f"WANDB_PROJECT: {self.WANDB_PROJECT}")
        self.logger.info(f"WANDB_RUN_NAME: {self.WANDB_RUN_NAME}")
        self.logger.info(f"TWO_TOWER_TINY_MODEL_PATH: {self.TWO_TOWER_TINY_MODEL_PATH}")

        self.qry_tower = self.QryTower()
        self.doc_tower = self.DocTower()

    class QryTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(300, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128)
            )

        def forward(self, x):
            emb = self.mlp(x)
            return F.normalize(emb, p=2, dim=-1)  # L2 normalize

    class DocTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(300, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128)
            )

        def forward(self, x):
            emb = self.mlp(x)
            return F.normalize(emb, p=2, dim=-1)  # L2 normalize

    def train(self):
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.warning(f"Using device: {self.device}")

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

        # Load training data from parquet
        self.logger.info(f"Loading training data from {self.TRIPLES_EMBEDDINGS_DATA_PATH}")
        df = pd.read_parquet(self.TRIPLES_EMBEDDINGS_DATA_PATH)

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

        # Save model
        os.makedirs(self.MODEL_OUTPUT_DIR, exist_ok=True)
        torch.save({
            "qry_tower_state_dict": self.qry_tower.state_dict(),
            "doc_tower_state_dict": self.doc_tower.state_dict(),
        }, self.TWO_TOWER_TINY_MODEL_PATH)
        self.logger.success(f"Model saved to {self.TWO_TOWER_TINY_MODEL_PATH}")
        wandb.save(self.TWO_TOWER_TINY_MODEL_PATH)
        wandb.finish()

def main():
    model = TwoTowerTiny()
    model.train()

if __name__ == "__main__":
    main()
