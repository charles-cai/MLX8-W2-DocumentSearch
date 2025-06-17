import os

from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from logging_utils import setup_logging, with_exception_logging

class BertInference:
    def __init__(self, model_name='bert-base-uncased'):
        self.logger = setup_logging(__name__)

        hf_cache_dir = os.getenv("HF_CACHE_DIR", "./.data/hf/.cache")
        self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.model = BertModel.from_pretrained(model_name, cache_dir=hf_cache_dir).to(self.device)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token embedding as sentence representation
        return outputs.last_hidden_state[:, 0, :].squeeze(0)

    def calculate_similarity(self, text1, text2):
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        similarity = F.cosine_similarity(emb1, emb2, dim=0)
        return similarity.item()

    def calculate_similarity_for_test_dataset(self):
        """
        Calculate cosine similarity between each query and the first passage_text in the test dataset.
        Stores the result as 'cosine_similarity' in MLX_DATASET_OUTPUT_DIR/test_similarity.parquet.
        """
        # Use environment variables for all paths
        hf_dataset_output_dir = os.getenv("HF_DATASET_OUTPUT_DIR", "./.data")
        hf_dataset = os.getenv("HF_DATASET", "ms_marco")
        hf_dataset_version = os.getenv("HF_DATASET_VERSION", "v1.1")
        mlx_output_dir = os.getenv("MLX_DATASET_OUTPUT_DIR", "./.data/processed")

        parquet_path = os.path.join(
            hf_dataset_output_dir,
            f"{hf_dataset}_{hf_dataset_version}",
            "test.parquet"
        )
        os.makedirs(mlx_output_dir, exist_ok=True)
        output_path = os.path.join(mlx_output_dir, "test_similarity.parquet")

        df = pd.read_parquet(parquet_path)
     
        passages = df["passages"].apply(lambda x: x["passage_text"][0])
        queries = df["query"]

        similarities = [
            self.calculate_similarity(q, p)
            for q, p in tqdm(zip(queries, passages), total=len(df), desc="Calculating similarities")
        ]
        df["cosine_similarity"] = similarities

        df.to_parquet(output_path, index=False)
        return df

@with_exception_logging
def main():
    query = "What is the capital of France?"
    document = "Paris is the capital of France, a country in Europe."

    # Use BertInference class
    bert = BertInference()
    similarity = bert.calculate_similarity(query, document)

    print(f"Cosine Similarity: {similarity}")

# Example usage from Python command line: python
# >>> from bert_inference import BertInference
# >>> bert = BertInference()
# >>> bert.calculate_similarity_for_test_dataset()

if __name__ == "__main__":
    main()

