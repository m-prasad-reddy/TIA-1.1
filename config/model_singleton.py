# config/model_singleton.py: Singleton for SentenceTransformer model
# Ensures single instance of model and disables tqdm progress bars

from sentence_transformers import SentenceTransformer
import tqdm
import logging

class ModelSingleton:
    """Singleton class for managing a single SentenceTransformer instance."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            # Disable tqdm progress bars globally
            tqdm.tqdm.disable = True
            # Initialize the model
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.getLogger("model_singleton").info("Initialized SentenceTransformer model")
        return cls._instance
    
    @property
    def model(self):
        """Get the SentenceTransformer model."""
        return self._model