import os
import warnings
import sys
import argparse
import json
import pandas as pd
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

from logging_utils import setup_logging, with_exception_logging

from dotenv import load_dotenv
load_dotenv()

class DataProcessing:
    def __init__(self):
        self.logger = setup_logging(__name__)
        self.logger.info("Initializing DataProcessing...")

        self.HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE")
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.HF_ORGANIZATION = os.getenv("HF_ORGANIZATION")
        self.HF_DATASET = os.getenv("HF_DATASET")
        self.HF_DATASET_VERSION = os.getenv("HF_DATASET_VERSION")
        self.HF_DATASET_SPLITS = os.getenv("HF_DATASET_SPLITS", "train,validation,test").split(",")
        self.HF_DATASET_OUTPUT_DIR = os.getenv("HF_DATASET_OUTPUT_DIR")
        self.MLX_DATASET_OUTPUT_DIR = os.getenv("MLX_DATASET_OUTPUT_DIR")
        self.MLX_DATASET_TRIPLE_SPLIT = os.getenv("MLX_DATASET_TRIPLE_SPLIT", "train")
        
        if not self.HF_DATASETS_CACHE:
            self.logger.warning("HF_DATASETS_CACHE not set, using default cache location")
        
        os.environ["HF_DATASETS_CACHE"] = self.HF_DATASETS_CACHE
        self.logger.info(f"Cache directory: {self.HF_DATASETS_CACHE}")
        self.logger.info(f"HF Output directory: {self.HF_DATASET_OUTPUT_DIR}")
        self.logger.info(f"MLX Output directory: {self.MLX_DATASET_OUTPUT_DIR}")
        self.logger.info(f"MLX Triple split: {self.MLX_DATASET_TRIPLE_SPLIT}")

    def download_and_save(self, split):
        self.logger.info(f"Starting download_and_save for split: {split}")
        base_output_dir = os.path.join(
            self.HF_DATASET_OUTPUT_DIR, f"{self.HF_DATASET}_{self.HF_DATASET_VERSION}"
        )
        os.makedirs(base_output_dir, exist_ok=True)
        parquet_file_path = os.path.join(base_output_dir, f"{split}.parquet")
        
        # Check if file already exists using existing method
        try:
            self._check_parquet_file(split)
            self.logger.warning(f"File already exists: {parquet_file_path}")
            response = input(f"Overwrite existing {split}.parquet file? [Y/n]: ").strip().lower()
            if response in ['n', 'no']:
                self.logger.info(f"Skipping download for {split} split")
                return
            elif response == '' or response in ['y', 'yes', 'Y']:
                self.logger.info(f"Proceeding with overwrite for {split} split")
            else:
                self.logger.warning(f"Invalid response '{response}', skipping download for {split} split")
                return
        except FileNotFoundError:
            # File doesn't exist, proceed with download
            pass
        
        self.logger.info(f"Starting download for split: {split}")
        
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

    def _check_parquet_file(self, split):
        self.logger.debug(f"Checking parquet file for split: {split}")
        """
        Check if parquet file exists for the given split.
        Returns the file path if exists, raises FileNotFoundError if not.
        """
        base_output_dir = os.path.join(
            self.HF_DATASET_OUTPUT_DIR, f"{self.HF_DATASET}_{self.HF_DATASET_VERSION}"
        )
        parquet_file_path = os.path.join(base_output_dir, f"{split}.parquet")
        
        if not os.path.exists(parquet_file_path):
            self.logger.error(f"Parquet file not found: {parquet_file_path}")
            raise FileNotFoundError(f"Parquet file not found: {parquet_file_path}")
        
        self.logger.debug(f"Parquet file found: {parquet_file_path}")
        return parquet_file_path
    
    def gen_triples(self, split):
        self.logger.info(f"Starting gen_triples for split: {split}")
        """
        Generate training triples from the parquet file by parsing JSON passages.
        Each record is expanded into multiple rows based on passage_text array.
        Returns a DataFrame with columns: id, query, query_id, is_selected, negative_query_id, positive_doc
        """
        parquet_file_path = self._check_parquet_file(split)
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
                    'answer': row['answers'],  # Fix: use 'answers' and add comma
                    'query_id': row['query_id'],
                    'is_selected': is_selected[i],  # Use actual value
                    'positive_doc': passage_text,  # Fix: use passage_text variable
                })
                auto_id += 1

        result_df = pd.DataFrame(results)
        
        if len(result_df) == 0:
            self.logger.error("No valid triples generated")
            return pd.DataFrame()
        
        self.logger.info(f"Generated {len(result_df)} triples from {total_records} original records")
        
        # Randomize negative id and negative_query_id columns
        result_df = self.def_randomize_negative_query_id(result_df)
        
        os.makedirs(self.MLX_DATASET_OUTPUT_DIR, exist_ok=True)
        triples_file_path = os.path.join(self.MLX_DATASET_OUTPUT_DIR, f"{split}_triples.parquet")
        result_df.to_parquet(triples_file_path, index=False)
        self.logger.info(f"Saved triples to: {triples_file_path}")
        
        self.logger.info(f"Completed gen_triples for split: {split}")
        return result_df

    def def_randomize_negative_query_id(self, df):
        self.logger.info("Starting def_randomize_negative_query_id")
        
        # Use numpy arrays for faster operations
        ids = df['id'].values
        query_ids = df['query_id'].values
        positive_docs = df['positive_doc'].values
        n = len(ids)

        # Optimized derangement using numpy
        def fast_derangement(arr1, arr2):
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

        negative_ids, negative_query_ids = fast_derangement(ids, query_ids)
        
        # Vectorized lookup for negative_doc using negative_ids
        id_to_doc_map = dict(zip(ids, positive_docs))
        negative_docs = np.array([id_to_doc_map[neg_id] for neg_id in negative_ids])
        
        df = df.copy()
        df['negative_id'] = negative_ids
        df['negative_query_id'] = negative_query_ids
        df['negative_doc'] = negative_docs

        self.logger.info("Completed def_randomize_negative_query_id")
        return df

@with_exception_logging
def main():
    
    parser = argparse.ArgumentParser(description="MS MARCO dataset processing")
    parser.add_argument(
        "--download", 
        action="store_true", 
        help="Download and save datasets to local storage"
    )
    parser.add_argument(
        "--gen-triples", 
        action="store_true", 
        help="Generate training triples from parquet files"
    )
    args = parser.parse_args()
    
    # Show help and exit if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    logger = setup_logging(__name__)
    logger.info("Starting MS MARCO dataset processing...")
    
    try:
        dp = DataProcessing()
        
        if args.download:
            logger.info("Download flag detected, proceeding with dataset download...")
            for split in dp.HF_DATASET_SPLITS:
                dp.download_and_save(split)
            logger.success("All datasets processed successfully!")
        
        if args.gen_triples:
            logger.info("Generate triples flag detected, checking parquet files...")
            # Only check the specific split for triple generation
            triple_split = dp.MLX_DATASET_TRIPLE_SPLIT
            try:
                dp._check_parquet_file(triple_split)
            except FileNotFoundError:
                logger.error(f"Missing parquet file for split: {triple_split}. Run with --download first.")
                sys.exit(1)
            
            # Generate triples only for the specified split
            logger.info(f"Generating triples for {triple_split} split only...")
            triples_df = dp.gen_triples(triple_split)
            logger.info(f"Generated {len(triples_df)} triples for {triple_split} split")
            logger.success("Triple generation completed successfully!")
        
        if not args.download and not args.gen_triples:
            logger.info("No action flags provided. Use --download to download datasets or --gen-triples to generate triples.")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()