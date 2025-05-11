# config/cache_synchronizer_in_memory.py: In-memory SQLite for debugging TableIdentifier-v2.1

import os
import sqlite3
import logging
import logging.config
import json
import threading
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import util
from datetime import datetime
from config.model_singleton import ModelSingleton
import re

class CacheSynchronizer:
    """
    Synchronizes cache data using an in-memory SQLite database for debugging.
    """
    
    def __init__(self, db_name: str):
        """
        Initialize CacheSynchronizer with in-memory SQLite database.
        
        Args:
            db_name: Name of the database (e.g., BikeStores).
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
        
        self.logger = logging.getLogger("cache_synchronizer")
        self.db_name = db_name
        self.db_path = ":memory:"
        self.model = ModelSingleton().model
        self.lock = threading.Lock()
        try:
            self.logger.debug("Starting in-memory SQLite database initialization")
            self._initialize_database()
            self.logger.debug("Completed in-memory SQLite database initialization")
            self.logger.info(f"Initialized CacheSynchronizer for {db_name} (in-memory)")
        except Exception as e:
            self.logger.error(f"Failed to initialize in-memory SQLite database: {e}")
            raise RuntimeError("Cannot initialize in-memory database")

    def _get_connection(self):
        """
        Create a new in-memory SQLite connection.
        """
        try:
            self.logger.debug("Creating new in-memory SQLite connection")
            conn = sqlite3.connect(":memory:", timeout=60)
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA busy_timeout=60000")
            self.logger.debug("Configured in-memory SQLite connection")
            return conn
        except Exception as e:
            self.logger.error(f"Error creating in-memory SQLite connection: {e}")
            raise

    def _initialize_database(self):
        """
        Create necessary tables sequentially in in-memory SQLite database.
        """
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                self.logger.debug("Creating weights table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS weights (
                        table_name TEXT,
                        column TEXT,
                        weight REAL,
                        PRIMARY KEY (table_name, column)
                    )
                """)
                conn.commit()
                
                self.logger.debug("Creating name_matches table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS name_matches (
                        column_name TEXT,
                        synonym TEXT,
                        PRIMARY KEY (column_name, synonym)
                    )
                """)
                conn.commit()
                
                self.logger.debug("Creating feedback table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        timestamp TEXT,
                        query TEXT,
                        tables TEXT,
                        embedding BLOB,
                        count INTEGER
                    )
                """)
                conn.commit()
                
                self.logger.debug("Creating ignored_queries table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ignored_queries (
                        query TEXT PRIMARY KEY,
                        embedding BLOB,
                        reason TEXT
                    )
                """)
                conn.commit()
                
                conn.close()
                self.logger.debug("Initialized in-memory database tables: weights, name_matches, feedback, ignored_queries")
                
                self._create_feedback_index()
        except Exception as e:
            self.logger.error(f"Error initializing in-memory database tables: {e}")
            raise

    def _create_feedback_index(self):
        """
        Create index on feedback.query with fallback.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            self.logger.debug("Attempting to create index idx_feedback_query")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query)")
            conn.commit()
            conn.close()
            self.logger.debug("Successfully created index idx_feedback_query")
        except Exception as e:
            self.logger.warning(f"Failed to create index idx_feedback_query: {e}")
            self.logger.info("Proceeding without index; performance may be affected")

    def _checkpoint_wal(self, conn):
        """
        No-op for in-memory database.
        """
        self.logger.debug("WAL checkpoint skipped (in-memory database)")

    # [Rest of the code is identical to version 8f9e4b2c-7a3d-4e1f-b8c6-9d0a5f1e6c7b, omitted for brevity]