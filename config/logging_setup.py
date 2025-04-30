# config/logging_setup.py: Centralized logging configuration

import os
import logging
import logging.config

def setup_logger(logger_name: str) -> logging.Logger:
    """Configure and return a logger with the given name."""
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    logging_config_path = "app-config/logging_config.ini"
    try:
        if os.path.exists(logging_config_path):
            logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
        else:
            # Fallback to basic console logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logging.warning(f"Logging config file not found: {logging_config_path}. Using console logging.")
    except Exception as e:
        # Fallback to console logging on error
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.error(f"Error loading logging config: {e}. Using console logging.")
    
    return logging.getLogger(logger_name)