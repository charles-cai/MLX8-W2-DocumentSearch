import os
import warnings
import sys
import argparse
import json
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np

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

    def gen_triples(self, split):
        """
        Generate training triples from the parquet file using columnar operations.
        Returns a DataFrame with columns: query_id, query, positive_docs, negative_query_id, negative_docs
        """
        try:
            parquet_file_path = self._check_parquet_file(split)
            self.logger.info(f"Reading parquet file: {parquet_file_path}")
            df = pd.read_parquet(parquet_file_path)
            total_records = len(df)

            # Generate negative indices (not equal to their own index)
            indices = df.index.to_numpy()
            neg_indices = np.random.randint(0, total_records, size=total_records)
            # Ensure negative index is not the same as positive index
            mask = neg_indices == indices
            while mask.any():
                neg_indices[mask] = np.random.randint(0, total_records, size=mask.sum())
                mask = neg_indices == indices

            # Gather negative rows
            negative_rows = df.iloc[neg_indices].reset_index(drop=True)

            # Only keep passage_text from passages for positive and negative docs
            def extract_passage_text(passages):
                # passages is expected to be a dict with 'passage_text' key
                if isinstance(passages, dict) and 'passage_text' in passages:
                    return passages['passage_text']
                # If it's a list of dicts, fallback for other formats
                if isinstance(passages, list) and len(passages) > 0 and isinstance(passages[0], dict):
                    return [p.get('passage_text', '') for p in passages]
                return []

            result_df = pd.DataFrame({
                'query_id': df['query_id'],
                'query': df['query'],
                'positive_docs': df['passages'].apply(extract_passage_text),
                'negative_query_id': negative_rows['query_id'],
                'negative_docs': negative_rows['passages'].apply(extract_passage_text)
            })

            self.logger.info(f"Generated {len(result_df)} triples from {split} split")
            os.makedirs(self.MLX_DATASET_OUTPUT_DIR, exist_ok=True)
            triples_file_path = os.path.join(self.MLX_DATASET_OUTPUT_DIR, f"{split}_triples.parquet")
            result_df.to_parquet(triples_file_path, index=False)
            self.logger.info(f"Saved triples to: {triples_file_path}")
            return result_df

        except Exception as e:
            self.logger.error(f"Failed to generate triples for {split}: {str(e)}")
            raise

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
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()