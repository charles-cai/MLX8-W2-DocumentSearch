import os
import warnings
import sys
import argparse
import json
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

from logging_utils import setup_logging, with_exception_logging

class DataProcessing:
    def __init__(self):
        self.logger = setup_logging(__name__)
        self.logger.info("Initializing DataProcessing...")
        
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
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
        try:
            base_output_dir = os.path.join(
                self.HF_DATASET_OUTPUT_DIR, f"{self.HF_DATASET}_{self.HF_DATASET_VERSION}"
            )
            os.makedirs(base_output_dir, exist_ok=True)
            parquet_file_path = os.path.join(base_output_dir, f"{split}.parquet")
            
            # Check if file already exists
            if os.path.exists(parquet_file_path):
                self.logger.warning(f"File already exists: {parquet_file_path}")
                response = input(f"Overwrite existing {split}.parquet file? [Y/n]: ").strip().lower()
                if response in ['n', 'no']:
                    self.logger.info(f"Skipping download for {split} split")
                    return
                elif response == '' or response in ['y', 'yes']:
                    self.logger.info(f"Proceeding with overwrite for {split} split")
                else:
                    self.logger.warning(f"Invalid response '{response}', skipping download for {split} split")
                    return
            
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
            
        except Exception as e:
            self.logger.error(f"Failed to download and save {split} dataset: {str(e)}")
            raise

    def _check_parquet_file(self, split):
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
        
        return parquet_file_path

    def add_negative_query_ids(self, split):
        """
        Add a negative_query_id column to the parquet file for the given split,
        such that each negative_query_id is a random query_id not equal to its own.
        """
        parquet_file_path = self._check_parquet_file(split)
        self.logger.info(f"Reading parquet file for negative_query_id generation: {parquet_file_path}")
        df = pd.read_parquet(parquet_file_path)
        query_ids = df['query_id'].tolist()
        # Generate derangement
        def derangement(lst):
            while True:
                shuffled = lst[:]
                np.random.shuffle(shuffled)
                if all(a != b for a, b in zip(lst, shuffled)):
                    return shuffled
        negative_query_ids = derangement(query_ids)
        df['negative_query_id'] = negative_query_ids
        # Save back to parquet (overwrite)
        df.to_parquet(parquet_file_path, index=False)
        self.logger.info(f"Added negative_query_id column and saved to: {parquet_file_path}")

    def gen_triples(self, split):
        """
        Generate training triples from the parquet file by parsing JSON passages.
        Each record is expanded into multiple rows based on passage_text array.
        Returns a DataFrame with columns: id, query, query_id, is_selected, negative_query_id, positive_doc
        """
        parquet_file_path = self._check_parquet_file(split)
        self.logger.info(f"Reading parquet file: {parquet_file_path}")
        df = pd.read_parquet(parquet_file_path)
        total_records = len(df)
        
        # Ensure negative_query_id exists
        if 'negative_query_id' not in df.columns:
            self.logger.info("negative_query_id column not found, generating...")
            self.add_negative_query_ids(split)
            df = pd.read_parquet(parquet_file_path)

        self.logger.info(f"Processing {total_records} records to generate triples...")
        
        results = []
        auto_id = 1

        for idx, row in tqdm(df.iterrows(), total=total_records, desc="Processing records"):
            passages_data = row['passages']
            passage_texts = passages_data['passage_text']
            is_selected = passages_data['is_selected']

            # Validate array lengths
            if len(is_selected) != len(passage_texts):
                self.logger.error(f"Mismatch in array lengths for query_id {row['query_id']}: is_selected={len(is_selected)}, passage_text={len(passage_texts)}, skipping")
                continue

            # For each passage_text, treat as positive_doc, generate a triple
            for i, passage_text in enumerate(passage_texts):
                results.append({
                    'id': auto_id,
                    'query': row['query'],
                    'query_id': row['query_id'],
                    'is_selected': is_selected[i],  # Use actual value
                    'negative_query_id': row['negative_query_id'],
                    'positive_doc': passage_text
                })
                auto_id += 1

        result_df = pd.DataFrame(results)
        
        if len(result_df) == 0:
            self.logger.error("No valid triples generated")
            return pd.DataFrame()
        
        self.logger.info(f"Generated {len(result_df)} triples from {total_records} original records")
        
        os.makedirs(self.MLX_DATASET_OUTPUT_DIR, exist_ok=True)
        triples_file_path = os.path.join(self.MLX_DATASET_OUTPUT_DIR, f"{split}_triples.parquet")
        result_df.to_parquet(triples_file_path, index=False)
        self.logger.info(f"Saved triples to: {triples_file_path}")
        
        return result_df

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