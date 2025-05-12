# config/model_singleton.py: Singleton for SentenceTransformer model in TableIdentifier-v2.1
# Added ensure_model method for compatibility

import logging
import logging.config
import os
from sentence_transformers import SentenceTransformer

class ModelSingleton:
    """
    Singleton class to manage a single instance of the SentenceTransformer model.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Ensure only one instance of ModelSingleton is created.
        """
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initialize the SentenceTransformer model with logging.
        """
        os.makedirs("logs", exist_ok=True)
        
        logging_config_path = "app-config/logging_config.ini"
        try:
            if os.path.exists(logging_config_path):
                logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            else:
                logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join("logs", "bikestores_app.log")),
                        logging.StreamHandler()
                    ]
                )
                logging.warning(f"Logging config file not found: {logging_config_path}")
        except Exception as e:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(os.path.join("logs", "bikestores_app.log")),
                    logging.StreamHandler()
                ]
            )
            logging.error(f"Error loading logging config: {e}")
        
        self.logger = logging.getLogger("model_singleton")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Initialized SentenceTransformer model")
        except Exception as e:
            self.logger.error(f"Failed to initialize SentenceTransformer model: {e}")
            self.model = None

    def ensure_model(self):
        """
        Ensure the SentenceTransformer model is initialized.
        """
        if self.model is None:
            self.logger.warning("Model not initialized, attempting to reload")
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Reinitialized SentenceTransformer model")
            except Exception as e:
                self.logger.error(f"Failed to reinitialize SentenceTransformer model: {e}")
                raise RuntimeError("Cannot initialize SentenceTransformer model")
        else:
            self.logger.debug("Model already initialized")