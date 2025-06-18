import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
import os

# Add import for downloader
import gensim.downloader as api
from gensim.utils import simple_preprocess

from logging_utils import setup_logging

from dotenv import load_dotenv
load_dotenv()

class Word2vecUtils:
    """
    Utility class for Word2Vec operations including loading pre-trained models
    and creating PyTorch embedding layers.
    """
    
    def __init__(self):
        self.logger = setup_logging(__name__)
        
        self.gensim_cache_dir = os.getenv("GENSIM_CACHE_DIR", "./.data/gensim")
        self.word2vec_model_name = os.getenv("GENSIM_WORD2VEC_MODEL", "GoogleNews-vectors-negative300.bin")
        self.normalize_case = os.getenv("GENSIM_NORMALIZE_CASE", "true").lower() == "true"
        
        self.w2v_model = None
        self.vocab = None
        self.vocab_size = None
        self.emb_dim = None
    
    def load_word2vec(self, w2v_path=None, freeze=True):
        """
        Load pre-trained Word2Vec model and build embedding matrix.
        
        Args:
            w2v_path: Path to Word2Vec model file (if None, uses GENSIM_CACHE_DIR + GENSIM_WORD2VEC_MODEL)
            freeze: Whether to freeze the embedding weights during training
        
        Returns:
            tuple: (embedding_layer, vocab_dict, vocab_size, embedding_dim)
        """
        if w2v_path is None:
            w2v_path = os.path.join(self.gensim_cache_dir, self.word2vec_model_name)
             
        # Download model if not exists
        if not os.path.exists(w2v_path):
            os.makedirs(self.gensim_cache_dir, exist_ok=True)

            self.logger.warning(f"Word2Vec model not found at {w2v_path}. Downloading 'word2vec-google-news-300' via gensim.downloader...")
            # Download using gensim downloader
            model = api.load("word2vec-google-news-300")
            # Save in binary format to the expected path
            model.save_word2vec_format(w2v_path, binary=True)
            self.logger.info(f"Downloaded and saved Word2Vec model to {w2v_path}")

        self.logger.info(f"Loading Word2Vec model from: {w2v_path}")
        self.logger.info(f"Using cache directory: {self.gensim_cache_dir}")
        
        # Load pre-trained Word2Vec
        self.w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        
        self.vocab = self.w2v_model.key_to_index  # word -> idx
        self.vocab_size = len(self.vocab)
        self.emb_dim = self.w2v_model.vector_size  # 300
        
        self.logger.info(f"Word2Vec model loaded successfully:")
        self.logger.info(f"  - Vocabulary size: {self.vocab_size:,}")
        self.logger.info(f"  - Embedding dimension: {self.emb_dim}")
        
        # Build embedding weight tensor (words not in w2v get zero vector)
        self.logger.info("Building embedding weight matrix...")
        weight = np.zeros((self.vocab_size, self.emb_dim), dtype=np.float32)
        for word, idx in self.vocab.items():
            weight[idx] = self.w2v_model[word]
            
        # PyTorch embedding layer
        embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(weight), 
            freeze=freeze
        )
        
        self.logger.success(f"Embedding layer created with freeze={freeze}")
        return embedding, self.vocab, self.vocab_size, self.emb_dim
      
    
    def embedding(self, text):
        """
        Calculate average embedding for a given text string.
        
        Args:
            text: Input text string
            
        Returns:
            numpy.ndarray: Average embedding vector of shape (embedding_dim,)
        """
        if self.w2v_model is None:
            self.logger.error("Word2Vec model not loaded. Call load_word2vec() first.")
            raise ValueError("Word2Vec model not loaded")
        
        if not text or not text.strip():
            self.logger.warning("Empty text provided, returning zero vector")
            return np.zeros(self.emb_dim, dtype=np.float32)
        
        # Apply case normalization based on configuration
        processed_text = text.strip()
        if self.normalize_case:
            processed_text = processed_text.lower()
        
        # Use gensim's simple_preprocess for tokenization and normalization
        words = simple_preprocess(processed_text, deacc=True)  # deacc=True removes accents
        self.logger.debug(f"Processing text with {len(words)} words (normalize_case={self.normalize_case}): {words[:5]}...")
        
        # Collect embeddings for words that exist in vocabulary
        embeddings = []
        oov_count = 0
        
        for word in words:
            if word in self.w2v_model:
                embeddings.append(self.w2v_model[word])
            else:
                oov_count += 1
        
        if not embeddings:
            self.logger.warning(f"No words found in vocabulary (OOV: {oov_count}), returning zero vector")
            return np.zeros(self.emb_dim, dtype=np.float32)
        
        # Calculate average embedding
        avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        
        self.logger.debug(f"Calculated embedding: {len(embeddings)} valid words, {oov_count} OOV words")
        return avg_embedding
    
    @staticmethod
    def load_word2vec_static(w2v_path='GoogleNews-vectors-negative300.bin', freeze=True):
        """
        Static method for backward compatibility.
        """
        utils = Word2vecUtils()
        return utils.load_word2vec(w2v_path, freeze)
    
def main():
    """
    Test function for Word2vecUtils class.
    """
    logger = setup_logging(__name__)
    logger.info("Testing Word2vecUtils functionality...")
    
    try:
        # Initialize Word2vecUtils
        w2v_utils = Word2vecUtils()
        logger.info(f"Initialized Word2vecUtils with:")
        logger.info(f"  - Cache directory: {w2v_utils.gensim_cache_dir}")
        logger.info(f"  - Model name: {w2v_utils.word2vec_model_name}")
        logger.info(f"  - Normalize case: {w2v_utils.normalize_case}")
        
        # Test loading Word2Vec model
        logger.info("Testing Word2Vec model loading...")
        embedding, vocab, vocab_size, emb_dim = w2v_utils.load_word2vec()
        logger.success(f"Successfully loaded Word2Vec model with {vocab_size:,} words and {emb_dim} dimensions")
        
        # Test embedding calculation
        test_texts = [
            "What is machine learning?",
            "How does neural network work?", 
            "Deep learning algorithms",
            "Natural language processing",
            "UPPERCASE AND lowercase TEXT"
        ]
        
        logger.info("Testing embedding calculation...")
        for i, text in enumerate(test_texts):
            try:
                embedding_vec = w2v_utils.embedding(text)
                logger.info(f"Text {i+1}: '{text[:30]}...' -> embedding shape: {embedding_vec.shape}")
                logger.debug(f"  First 5 values: {embedding_vec[:5]}")
            except Exception as e:
                logger.error(f"Failed to calculate embedding for text {i+1}: {e}")
        
        # Test with empty text
        logger.info("Testing with empty text...")
        empty_embedding = w2v_utils.embedding("")
        logger.info(f"Empty text embedding shape: {empty_embedding.shape}")
        
        # Test vocabulary lookup
        logger.info("Testing vocabulary lookup...")
        test_words = ["machine", "learning", "MACHINE", "unknownword123", "the", "of"]
        for word in test_words:
            if word.lower() in vocab:
                logger.info(f"Word '{word}' found in vocabulary (index: {vocab[word.lower()]})")
            else:
                logger.warning(f"Word '{word}' not found in vocabulary")
        
        logger.success("All Word2vecUtils tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Word2vecUtils test failed: {e}")
        raise

if __name__ == "__main__":
    main()
