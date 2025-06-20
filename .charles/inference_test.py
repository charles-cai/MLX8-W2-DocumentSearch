import os
import torch
import numpy as np
import pandas as pd
import faiss
import random
from dotenv import load_dotenv
from logging_utils import setup_logging
from word2vec_utils import Word2vecUtils
from two_tower_tiny import QryTower, DocTower
from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)

load_dotenv()

class InferenceTest:
    def __init__(self):
        self.logger = setup_logging(__name__)
        
        # Environment variables
        self.MLX_DATASET_OUTPUT_DIR = os.getenv("MLX_DATASET_OUTPUT_DIR", "./.data/processed")
        self.MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "./.data/models")
        self.TOP_K = int(os.getenv("TOP_K", "10"))
        self.NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "5"))
        self.INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "auto")  # auto, cpu, cuda
        
        # File paths - focus on test data only
        self.TWO_TOWER_TINY_MODEL_PATH = os.path.join(self.MODEL_OUTPUT_DIR, "two_tower_tiny.pt")
        self.TEST_TRIPLES_EMBEDDINGS_PATH = os.path.join(self.MLX_DATASET_OUTPUT_DIR, "test_triples_embeddings.parquet")
        
        # Use validation faiss index from final epoch (since we trained on validation set)
        self.DOC_INDEX_PATH = os.path.join(self.MLX_DATASET_OUTPUT_DIR, "faiss", f"epoch_{self.NUM_EPOCHS}_validation_doc_tower_tiny.index")
        self.VALIDATION_TOWER_EMBEDDINGS_PATH = os.path.join(self.MLX_DATASET_OUTPUT_DIR, f"epoch_{self.NUM_EPOCHS}_validation_triples_embeddings_tower_tiny.parquet")
        
        self.logger.info(f"Model path: {self.TWO_TOWER_TINY_MODEL_PATH}")
        self.logger.info(f"Test data path: {self.TEST_TRIPLES_EMBEDDINGS_PATH}")
        self.logger.info(f"Doc index path: {self.DOC_INDEX_PATH}")
        self.logger.info(f"TOP_K: {self.TOP_K}")
        
        # Initialize device selection
        self._setup_device()
        
        # Initialize models and utilities
        self.qry_tower = None
        self.doc_tower = None
        self.w2v_utils = None  # Still needed for preprocessing user queries
        self.doc_index = None
        self.validation_data = None  # For document lookup only
        self.test_data = None  # Always load test data for random selection
        
        self._load_components()
    
    def _setup_device(self):
        """Setup device for inference with detailed logging"""
        if self.INFERENCE_DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.INFERENCE_DEVICE == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        
        # Log GPU information if available
        if self.device.type == "cuda":
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
            
            self.logger.success(f"Using GPU inference:")
            self.logger.info(f"  Device: {self.device} ({gpu_name})")
            self.logger.info(f"  GPU Memory: {gpu_memory:.1f} GB")
            self.logger.info(f"  Total GPUs: {gpu_count}")
            self.logger.info(f"  CUDA Version: {torch.version.cuda}")
        else:
            self.logger.warning(f"Using CPU inference: {self.device}")
    
    def _load_components(self):
        """Load all required components: models, word2vec, faiss index, and data"""
        self.logger.info("Loading inference components...")
        
        # Load word2vec (needed because trained model expects 300-dim word2vec input)
        self.logger.info("Loading Word2Vec model for query preprocessing...")
        self.logger.warning("Word2Vec is required because the trained two-tower model expects 300-dim word2vec embeddings as input")
        self.w2v_utils = Word2vecUtils()
        self.w2v_utils.load_word2vec()
        
        # Load trained two-tower models
        self.logger.info("Loading trained two-tower models...")
        self.qry_tower = QryTower()
        self.doc_tower = DocTower()
        
        if not os.path.exists(self.TWO_TOWER_TINY_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {self.TWO_TOWER_TINY_MODEL_PATH}")
        
        checkpoint = torch.load(self.TWO_TOWER_TINY_MODEL_PATH, map_location=self.device)
        self.qry_tower.load_state_dict(checkpoint["qry_tower_state_dict"])
        self.doc_tower.load_state_dict(checkpoint["doc_tower_state_dict"])
        
        # Move models to device and set to eval mode
        self.qry_tower.to(self.device)
        self.doc_tower.to(self.device)
        self.qry_tower.eval()
        self.doc_tower.eval()
        
        self.logger.info(f"Models loaded and moved to {self.device}")
        
        # Load faiss index (contains transformed document embeddings)
        self.logger.info("Loading faiss document index...")
        if not os.path.exists(self.DOC_INDEX_PATH):
            raise FileNotFoundError(f"Faiss index not found: {self.DOC_INDEX_PATH}")
        
        self.doc_index = faiss.read_index(self.DOC_INDEX_PATH)
        
        # Setup faiss GPU if available and desired
        self._setup_faiss_gpu()
        
        # Load validation data for document text lookup (index -> document mapping)
        self.logger.info("Loading validation data for document lookup...")
        if not os.path.exists(self.VALIDATION_TOWER_EMBEDDINGS_PATH):
            raise FileNotFoundError(f"Validation tower embeddings not found: {self.VALIDATION_TOWER_EMBEDDINGS_PATH}")
        
        self.validation_data = pd.read_parquet(self.VALIDATION_TOWER_EMBEDDINGS_PATH)
        
        # Load test data from parquet file
        self.logger.info("Loading test data for random selection...")
        if not os.path.exists(self.TEST_TRIPLES_EMBEDDINGS_PATH):
            raise FileNotFoundError(f"Test data not found: {self.TEST_TRIPLES_EMBEDDINGS_PATH}")
        
        self.test_data = pd.read_parquet(self.TEST_TRIPLES_EMBEDDINGS_PATH)
        
        self.logger.info(f"Loaded {len(self.validation_data)} validation documents for lookup")
        self.logger.info(f"Loaded {len(self.test_data)} test samples for random selection")
        self.logger.info(f"Faiss index contains {self.doc_index.ntotal} documents")
        self.logger.success("All components loaded successfully!")
    
    def _setup_faiss_gpu(self):
        """Setup faiss for GPU usage if available"""
        if self.device.type == "cuda":
            try:
                # Try to move faiss index to GPU
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, self.doc_index)
                self.doc_index = gpu_index
                self.logger.success("Faiss index moved to GPU for faster search")
            except Exception as e:
                self.logger.warning(f"Failed to move faiss index to GPU: {e}")
                self.logger.info("Using CPU faiss index")
        else:
            self.logger.info("Using CPU faiss index")
    
    def _embed_query(self, query_text):
        """Convert query text to tower embedding via word2vec preprocessing"""
        # Step 1: Convert text to word2vec embedding (300-dim)
        w2v_embedding = self.w2v_utils.embedding(query_text)
        
        # Step 2: Convert to torch tensor and add batch dimension
        w2v_tensor = torch.tensor(w2v_embedding).float().unsqueeze(0).to(self.device)
        
        # Step 3: Pass through query tower (300-dim -> 128-dim normalized)
        with torch.no_grad():
            tower_embedding = self.qry_tower(w2v_tensor)
        
        return tower_embedding.cpu().numpy().squeeze()
    
    def _search_documents(self, query_embedding, top_k=None):
        """Search for top-k similar documents using faiss"""
        if top_k is None:
            top_k = self.TOP_K
        
        # Ensure query embedding is 2D for faiss
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search using faiss (cosine similarity via inner product on normalized vectors)
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        
        if start_time:
            start_time.record()
        
        scores, indices = self.doc_index.search(query_embedding.astype('float32'), top_k)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            search_time = start_time.elapsed_time(end_time)
            self.logger.info(f"Faiss search took {search_time:.2f} ms on GPU")
        
        return scores[0], indices[0]
    
    def _get_random_test_query(self):
        """Get a random query from test data"""
        # Select a random row from already loaded test data
        random_idx = random.randint(0, len(self.test_data) - 1)
        random_row = self.test_data.iloc[random_idx]
        
        return random_row['query'], random_row['query_id'], random_row['positive_doc']
    
    def _get_score_color(self, score):
        """Get color based on RAG scoring system"""
        if score >= 0.8:
            return Fore.GREEN + Style.BRIGHT  # Very green
        elif score >= 0.5:
            return Fore.GREEN  # Less green
        elif score >= 0.3:
            return Fore.YELLOW  # Amber
        else:
            return Fore.RED  # Red
    
    def _get_sky_400_color(self):
        """Get Tailwind Sky 400 equivalent color"""
        return "\033[38;2;56;189;248m"  # RGB(56, 189, 248) - Tailwind Sky 400
    
    def _get_orange_400_color(self):
        """Get Tailwind Orange 400 equivalent color"""
        return "\033[38;2;251;146;60m"  # RGB(251, 146, 60) - Tailwind Orange 400
    
    def _get_fuchsia_400_color(self):
        """Get Tailwind Fuchsia 400 equivalent color"""
        return "\033[38;2;232;121;249m"  # RGB(232, 121, 249) - Tailwind Fuchsia 400
    
    def _format_results(self, query_text, query_id, expected_doc, scores, indices):
        """Format search results for display with colorama colors"""
        print("=" * 80)
        print("INFERENCE TEST RESULTS")
        print("=" * 80)
        print(f"Device: {self.device} ({'GPU' if self.device.type == 'cuda' else 'CPU'})")
        
        # Query in green
        print(f"{Fore.GREEN}Query: {query_text}")
        
        if query_id is not None:
            print(f"{Fore.CYAN}Query ID: {query_id}")
        if expected_doc is not None:
            print(f"{Fore.CYAN}Expected Document: {expected_doc[:150]}...")
        
        print("\n" + "-" * 80)
        print(f"TOP {len(indices)} RETRIEVED DOCUMENTS:")
        print("-" * 80)
        
        sky_400 = self._get_sky_400_color()
        orange_400 = self._get_orange_400_color()
        fuchsia_400 = self._get_fuchsia_400_color()
        
        for rank, (score, idx) in enumerate(zip(scores, indices), 1):
            score_color = self._get_score_color(score)
            
            if idx < len(self.validation_data):
                doc = self.validation_data.iloc[idx]
                doc_text = doc['positive_doc']
                doc_query_id = doc['query_id']
                
                # New format: Rank x [Score: 0.xxxx, Q_ID: xxxxx]: document text
                print(f"{sky_400}Rank {rank}{Style.RESET_ALL} [{fuchsia_400}Score: {score:.4f}{Style.RESET_ALL}, {orange_400}Q_ID: {doc_query_id}{Style.RESET_ALL}]: {Fore.GREEN}{doc_text[:150]}...{Style.RESET_ALL}")
                
                # Check if this is a relevant document (same query_id)
                if query_id is not None and doc_query_id == query_id:
                    print(f"{Fore.MAGENTA + Style.BRIGHT}*** RELEVANT MATCH ***{Style.RESET_ALL}")
                
                print()  # Add blank line between results
            else:
                print(f"{sky_400}Rank {rank}{Style.RESET_ALL} [{fuchsia_400}Score: {score:.4f}{Style.RESET_ALL}, {Fore.RED}Q_ID: OUT_OF_RANGE{Style.RESET_ALL}]: {Fore.RED}Document index out of range{Style.RESET_ALL}")
                print()
        
        print("=" * 80)
    
    def _get_query_by_id(self, query_id):
        """Get query and documents by query_id from test data"""
        try:
            query_id = int(query_id)
            matching_rows = self.test_data[self.test_data['query_id'] == query_id]
            
            if matching_rows.empty:
                return None, None, []
            
            # Get the first matching row for query text
            first_row = matching_rows.iloc[0]
            query_text = first_row['query']
            
            # Get all positive documents for this query_id
            positive_docs = matching_rows['positive_doc'].tolist()
            
            return query_text, query_id, positive_docs
            
        except ValueError:
            return None, None, []
    
    def _format_query_id_results(self, query_text, query_id, positive_docs):
        """Format query ID lookup results for display"""
        print("=" * 80)
        print("QUERY ID LOOKUP RESULTS")
        print("=" * 80)
        
        # Query in green
        print(f"{Fore.GREEN}Query: {query_text}")
        print(f"{Fore.CYAN}Query ID: {query_id}")
        print(f"{Fore.CYAN}Total Documents: {len(positive_docs)}")
        
        print("\n" + "-" * 80)
        print(f"ALL DOCUMENTS FOR QUERY ID {query_id}:")
        print("-" * 80)
        
        for i, doc_text in enumerate(positive_docs, 1):
            print(f"{Fore.YELLOW}Document {i}:{Style.RESET_ALL} {Fore.GREEN}{doc_text[:150]}...{Style.RESET_ALL}")
            print()
        
        print("=" * 80)

    def run_inference(self, query_text=None):
        """Run inference with provided query, query ID lookup, or random selection from test data"""
        if query_text is None or query_text.strip() == "":
            # Random selection from test data
            self.logger.info("No query provided, selecting random query from test data...")
            query_text, query_id, expected_doc = self._get_random_test_query()
            self.logger.info(f"Selected random query: {query_text[:50]}...")
        elif query_text.strip().isdigit():
            # Numeric input - treat as query_id lookup
            self.logger.info(f"Numeric input detected, looking up query ID: {query_text}")
            lookup_query, lookup_query_id, positive_docs = self._get_query_by_id(query_text)
            
            if lookup_query is None:
                print(f"{Fore.RED}No documents found for Query ID {query_text}.{Style.RESET_ALL}")
                return {
                    'query': None,
                    'query_id': int(query_text),
                    'error': 'No documents found for this Query ID'
                }
            
            # Display query ID results and return early
            self._format_query_id_results(lookup_query, lookup_query_id, positive_docs)
            return {
                'query': lookup_query,
                'query_id': lookup_query_id,
                'documents': positive_docs,
                'total_documents': len(positive_docs)
            }
        else:
            # User provided query text
            self.logger.info(f"Processing user query: {query_text[:50]}...")
            query_id = None
            expected_doc = None

        # Embed the query (text -> word2vec -> query tower -> 128-dim)
        embedding_start = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        embedding_end = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        
        if embedding_start:
            embedding_start.record()
        
        query_embedding = self._embed_query(query_text)
        
        if embedding_end:
            embedding_end.record()
            torch.cuda.synchronize()
            embedding_time = embedding_start.elapsed_time(embedding_end)
            self.logger.info(f"Query embedding took {embedding_time:.2f} ms on GPU")
        
        self.logger.info(f"Query embedded to {query_embedding.shape} dimensions")
        
        # Search for similar documents in faiss index
        scores, indices = self._search_documents(query_embedding)
        
        # Display results
        self._format_results(query_text, query_id, expected_doc, scores, indices)
        
        return {
            'query': query_text,
            'query_id': query_id,
            'scores': scores.tolist(),
            'indices': indices.tolist()
        }

def main():
    """Main interactive loop"""
    logger = setup_logging(__name__)
    logger.info("Starting Inference Test...")
    
    try:
        # Initialize inference engine
        inference = InferenceTest()
        
        print("\n" + "=" * 80)
        print("TWO-TOWER DOCUMENT SEARCH INFERENCE TEST")
        print("=" * 80)
        print("Architecture: Text -> Word2Vec(300d) -> QueryTower -> L2Norm(128d) -> Faiss Search")
        print(f"Device: {inference.device} ({'GPU' if inference.device.type == 'cuda' else 'CPU'})")
        print("=" * 80)
        print("Instructions:")
        print("- Press Enter (empty query) to test with a random query from test data")
        print("- Type a query to search for similar documents")
        print("- Type a number to lookup specific Query ID and see all its documents")
        print("- Press Ctrl-C to exit")
        print("=" * 80)
        
        while True:
            try:
                # Blue colored prompt
                query = input(f"\n{Fore.BLUE}Enter your query (or press Enter for random, Ctrl-C to exit): {Style.RESET_ALL}").strip()
                
                # Run inference
                results = inference.run_inference(query if query else None)
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}Exiting...{Style.RESET_ALL}")
                break
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                continue
    
    except Exception as e:
        logger.error(f"Failed to initialize inference: {e}")
        print(f"{Fore.RED}Initialization error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
