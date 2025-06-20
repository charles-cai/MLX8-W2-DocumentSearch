import os
import sys
import argparse
import pandas as pd
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from word2vec_utils import Word2vecUtils
from logging_utils import setup_logging, with_exception_logging

import faiss
import torch
import torch.nn.functional as F

from dotenv import load_dotenv
load_dotenv()

class DataProcessing:
    def __init__(self):
        self.logger = setup_logging(__name__)
        self.logger.info("Initializing DataProcessing...")

        # Only load env vars actually used in this file, with default values
        self.HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE", "./.data/hf")
        self.HF_TOKEN = os.getenv("HF_TOKEN", "")
        self.HF_ORGANIZATION = os.getenv("HF_ORGANIZATION", "microsoft")
        self.HF_DATASET = os.getenv("HF_DATASET", "ms_marco")
        self.HF_DATASET_VERSION = os.getenv("HF_DATASET_VERSION", "v1.1")
        self.HF_DATASET_OUTPUT_DIR = os.getenv("HF_DATASET_OUTPUT_DIR", "./.data")
        self.MLX_DATASET_OUTPUT_DIR = os.getenv("MLX_DATASET_OUTPUT_DIR", "./.data/processed")
    
        splits_str = os.getenv("HF_DATASET_SPLITS", "train,validation,test")
        self.HF_DATASET_SPLITS = [s.strip() for s in splits_str.split(",")]
 
        self.BASE_OUTPUT_DIR = os.path.join(self.HF_DATASET_OUTPUT_DIR, f"{self.HF_DATASET}_{self.HF_DATASET_VERSION}")

        self.logger.warning(f"Cache directory: {self.HF_DATASETS_CACHE}")
        self.logger.warning(f"HF Output directory: {self.HF_DATASET_OUTPUT_DIR}")
        self.logger.warning(f"MLX Output directory: {self.MLX_DATASET_OUTPUT_DIR}")
        self.logger.warning(f"BASE Output directory: {self.BASE_OUTPUT_DIR}")

    def download_and_save(self):
        self.logger.info(f"Starting download_and_save...")

        os.makedirs(self.BASE_OUTPUT_DIR, exist_ok=True)        
        
        # Check if file already exists using existing method
        for split in self.HF_DATASET_SPLITS:
            exists, parquet_file_path = self._check_raw_parquet_file(split)

            if exists and not self._human_confirm(f"Overwrite existing {split}.parquet file? [Y/n]: "): 
                continue
            
            dataset = load_dataset(
                f"{self.HF_ORGANIZATION}/{self.HF_DATASET}",
                self.HF_DATASET_VERSION,
                split=split,
                token=self.HF_TOKEN if self.HF_TOKEN else None
            )
            
            self.logger.info(f"Saving {split} dataset to: {parquet_file_path}")
            dataset.to_parquet(parquet_file_path)

            self.logger.success(f"Successfully saved {split} dataset ({len(dataset)} records)")
            self.logger.info(f"Completed download_and_save for split: {split}")

    def _human_confirm(self, message):
        """
        Helper method to confirm user input for overwriting files.
        Returns True if user confirms, False otherwise.
        """
        response = input(f"{message} [Y/n]: ").strip().lower()
        if response in ['y', 'yes', 'Y']:
            return True
        elif response in ['n', 'no', 'N']:
            return False
        else:
            self.logger.error(f"Invalid response '{response}', please enter 'Y' or 'N'.")
            return self._human_confirm(message)

    def _check_raw_parquet_file(self, split):
        """
        Check if parquet file exists for the given split.
        Returns tuple (exists, file_path).
        """

        base_output_dir = os.path.join(self.HF_DATASET_OUTPUT_DIR, f"{self.HF_DATASET}_{self.HF_DATASET_VERSION}")
        parquet_file_path = os.path.join(base_output_dir, f"{split}.parquet")

        exists = os.path.exists(parquet_file_path)
        if not exists:
            self.logger.warning(f"Parquet file not found: {parquet_file_path}")

        return exists, parquet_file_path
    
    def gen_triples_all(self):
        """
        Generate training triples for all splits defined in HF_DATASET_SPLITS.
        Calls gen_triples for each split and saves the results.
        """
        self.logger.info("Starting gen_triples_all for all splits")
        
        for split in self.HF_DATASET_SPLITS:
            self._gen_triples(split)
  
    def _gen_triples(self, split):
        """
        Generate training triples from the parquet file by parsing JSON passages.
        Each record is expanded into multiple rows based on passage_text array.
        Returns a DataFrame with columns: id, query, query_id, is_selected, negative_query_id, positive_doc
        """
        self.logger.info(f"Starting gen_triples for {split}")

        exists, parquet_file_path = self._check_raw_parquet_file(split)
        self.logger.info(f"Reading parquet file: {parquet_file_path}")
        df = pd.read_parquet(parquet_file_path)
        total_records = len(df)
        
        self.logger.info(f"Processing {total_records} records to generate triples...")
        
        results = []
        auto_id = 1

        for idx, row in tqdm(df.iterrows(), total=total_records, desc="Processing records"):
            passage_texts = row['passages']['passage_text']
            is_selected = row['passages']['is_selected']

            # Validate array lengths
            if len(is_selected) != len(passage_texts):
                self.logger.error(f"Mismatch in array lengths for query_id {row['query_id']}: is_selected={len(is_selected)}, passage_text={len(passage_texts)}, skipping")
                continue

            # For each passage_text, treat as positive_doc, generate a triple
            for i, passage_text in enumerate(passage_texts):
                results.append({
                    'id': auto_id,
                    'query': row['query'],
                    'answer': row['answers'],
                    'query_id': row['query_id'],
                    'is_selected': is_selected[i],
                    'positive_doc': passage_text,
                })
                auto_id += 1

        result_df = pd.DataFrame(results)
        if len(result_df) == 0:
            self.logger.error("No valid triples generated")
            return
        
        self.logger.info(f"Generated {len(result_df)} triples from {total_records} original records; generating randomized negative docs")

        # Randomize negative id and negative_query_id columns
        result_df = self._def_randomize_negative_query_id(result_df)
        
        os.makedirs(self.MLX_DATASET_OUTPUT_DIR, exist_ok=True)
        triples_file_path = os.path.join(self.MLX_DATASET_OUTPUT_DIR, f"{split}_triples.parquet")
        result_df.to_parquet(triples_file_path, index=False)
        self.logger.info(f"Saved triples to: {triples_file_path}")
        
        self.logger.info(f"Completed gen_triples for split: {split}")
        return result_df

    def _def_randomize_negative_query_id(self, df):
        self.logger.info("Starting _def_randomize_negative_query_id")
        
        # Use numpy arrays for faster operations
        ids = df['id'].values
        query_ids = df['query_id'].values
        positive_docs = df['positive_doc'].values
        
        negative_ids, negative_query_ids = self._fast_derangement(ids, query_ids)
        
        # Vectorized lookup for negative_doc using negative_ids
        id_to_doc_map = dict(zip(ids, positive_docs))
        negative_docs = np.array([id_to_doc_map[neg_id] for neg_id in negative_ids])
        
        df = df.copy()
        df['negative_id'] = negative_ids
        df['negative_query_id'] = negative_query_ids
        df['negative_doc'] = negative_docs

        self.logger.info("Completed _def_randomize_negative_query_id")
        return df

    # Optimized derangement using numpy
    def _fast_derangement(self, arr1, arr2):
        
        n = len(arr1)
        indices = np.arange(n)
        max_attempts = 1000
        
        for _ in range(max_attempts):
            np.random.shuffle(indices)
            if np.all(arr2 != arr2[indices]):
                return arr1[indices], arr2[indices]
        
        # Fallback: manual swap if needed
        while np.any(arr2 == arr2[indices]):
            conflicts = np.where(arr2 == arr2[indices])[0]
            if len(conflicts) >= 2:
                indices[conflicts[0]], indices[conflicts[1]] = indices[conflicts[1]], indices[conflicts[0]]
            else:
                # Find a non-conflicting position
                for i in range(n):
                    if i not in conflicts and arr2[conflicts[0]] != arr2[i]:
                        indices[conflicts[0]], indices[i] = indices[i], indices[conflicts[0]]
                        break
        
        return arr1[indices], arr2[indices]
    
    def store_embeddings_all(self):
        self.logger.info("Starting store_embeddings, loading word2vec model, ..")
        w2v_utils = Word2vecUtils()
        if w2v_utils.w2v_model is None:
            self.logger.warning("Word2Vec model not loaded, loading now...")
            w2v_utils.load_word2vec()

        self.logger.info("Completed store_embeddings_all for all splits")
        for split in self.HF_DATASET_SPLITS:
            self._store_embeddings(split, w2v_utils)
        
    def _store_embeddings(self, split, w2v_utils):
        """
        Compute and store embeddings for query, positive_doc, and negative_doc columns.
        Uses instance variables for input/output paths.
        """
            
        triples_output_path = os.path.join(self.MLX_DATASET_OUTPUT_DIR, f"{split}_triples.parquet")
        
        self.logger.info(f"Loading triples from {triples_output_path}")
        df = pd.read_parquet(triples_output_path)
        self.logger.info(f"Loaded {len(df)} triples.")

        # Compute embeddings for each row with tqdm progress bar
        self.logger.info("Computing embeddings for query, positive_doc, and negative_doc...")
        query_embs = []
        pos_embs = []
        neg_embs = []

        # Add lists for sequence embeddings
        query_embs_seq = []
        pos_doc_embs_seq = []
        neg_doc_embs_seq = []

        w2v_model = w2v_utils.w2v_model
        vector_size = w2v_model.vector_size

        def get_seq_embedding(text):
            tokens = text.lower().split()
            vectors = [w2v_model[word] for word in tokens if word in w2v_model]
            if not vectors:
                return [np.zeros(vector_size, dtype=np.float32)]
            return [vec.astype(np.float32) for vec in vectors]

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding rows"):
            query_embs.append(w2v_utils.embedding(row["query"]))
            pos_embs.append(w2v_utils.embedding(row["positive_doc"]))
            neg_embs.append(w2v_utils.embedding(row["negative_doc"]))

            # Sequence embeddings
            query_embs_seq.append(get_seq_embedding(row["query"]))
            pos_doc_embs_seq.append(get_seq_embedding(row["positive_doc"]))
            neg_doc_embs_seq.append(get_seq_embedding(row["negative_doc"]))
            

        # Convert to numpy arrays
        query_embs = np.stack(query_embs)
        pos_embs = np.stack(pos_embs)
        neg_embs = np.stack(neg_embs)

        # Add new columns to DataFrame
        df["query_emb"] = list(query_embs)
        df["positive_doc_emb"] = list(pos_embs)
        df["negative_doc_emb"] = list(neg_embs)

        # Add sequence embeddings to DataFrame
        df["query_emb_seq"] = query_embs_seq
        df["positive_doc_emb_seq"] = pos_doc_embs_seq
        df["negative_doc_emb_seq"] = neg_doc_embs_seq

        triple_embedding_path = os.path.join(self.MLX_DATASET_OUTPUT_DIR, f"{split}_triples_embeddings.parquet")
        self.logger.info(f"Saving embeddings to {triple_embedding_path}")
    
        df.to_parquet(triple_embedding_path, index=False)
        self.logger.info(f"Stored embeddings for {len(df)} triples at {triple_embedding_path}")

@with_exception_logging
def main():
    
    parser = argparse.ArgumentParser(description="MS MARCO dataset processing")
    parser.add_argument(
        "--download", 
        action="store_true", 
        help="Step 1: Download and save datasets to local storage"
    )
    parser.add_argument(
        "--gen-triples",
        action="store_true",
         help="Step 2: Generate training triples from parquet files."
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Step 3: Generate and store embeddings for triples"
    )

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    logger = setup_logging(__name__)
    logger.info("Starting MS MARCO dataset processing...")
    
    dp = DataProcessing()
    if args.download:
        dp.download_and_save()
    elif args.gen_triples:
        dp.gen_triples_all()
    elif args.embedding:
        dp.store_embeddings_all()
    else:
        logger.error("No valid action flags provided")
        parser.print_help()
        sys.exit(-1)
            
if __name__ == "__main__":
    main()