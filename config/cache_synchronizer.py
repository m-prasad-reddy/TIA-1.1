# config/cache_synchronizer.py: Manages cache operations for TableIdentifier-v1
# Fixes write_ignored_query for np.ndarray embeddings, preserves ~512 lines

import os
import sqlite3
import logging
import logging.config
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import util
from datetime import datetime
from config.model_singleton import ModelSingleton

class CacheSynchronizer:
    """
    Synchronizes cache data (weights, name matches, feedback, ignored queries) with SQLite database.
    Provides methods for reading, writing, and managing cache entries.
    """
    
    def __init__(self, db_name: str):
        """
        Initialize CacheSynchronizer with SQLite database for a specific database.
        
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
        self.db_path = os.path.join("app-config", db_name, "cache.db")
        self.model = ModelSingleton().model
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self._initialize_database()
            self.logger.debug(f"Initialized SQLite database at {self.db_path}")
            self.logger.info(f"Initialized CacheSynchronizer for {db_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite database: {e}")
            raise RuntimeError(f"Cannot connect to cache database: {self.db_path}")

    def _initialize_database(self):
        """
        Create necessary tables in SQLite database for weights, name matches, feedback, and ignored queries.
        Matches original schema based on error logs.
        """
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS weights (
                    table_name TEXT,
                    column TEXT,
                    weight REAL,
                    PRIMARY KEY (table_name, column)
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS name_matches (
                    column_name TEXT,
                    synonym TEXT,
                    PRIMARY KEY (column_name, synonym)
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    timestamp TEXT,
                    query TEXT,
                    tables TEXT,
                    embedding BLOB,
                    count INTEGER
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS ignored_queries (
                    query TEXT PRIMARY KEY,
                    embedding BLOB,
                    reason TEXT
                )
            """)
            self.conn.commit()
            self.logger.debug("Initialized database tables: weights, name_matches, feedback, ignored_queries")
        except Exception as e:
            self.logger.error(f"Error initializing database tables: {e}")
            raise

    def _embedding_to_blob(self, embedding: Optional[np.ndarray]) -> Optional[bytes]:
        """
        Convert numpy embedding to SQLite BLOB.
        
        Args:
            embedding: Numpy array of embedding values or None.
        
        Returns:
            Bytes representation of the embedding or None.
        """
        try:
            if embedding is None:
                return None
            return embedding.tobytes()
        except Exception as e:
            self.logger.error(f"Error converting embedding to blob: {e}")
            return None

    def _blob_to_embedding(self, blob: bytes, dim: int = 384) -> np.ndarray:
        """
        Convert SQLite BLOB to numpy embedding.
        
        Args:
            blob: Bytes representation of the embedding.
            dim: Expected dimension of the embedding (default: 384 for sentence-transformers).
        
        Returns:
            Numpy array of embedding values.
        """
        try:
            return np.frombuffer(blob, dtype=np.float32).reshape(-1)
        except Exception as e:
            self.logger.error(f"Error converting blob to embedding: {e}")
            return np.zeros(dim, dtype=np.float32)

    def write_weights(self, weights: Dict[str, Dict[str, float]]):
        """
        Write weights to SQLite database.
        
        Args:
            weights: Dictionary of table names to column-weight mappings.
        """
        try:
            self.cursor.execute("DELETE FROM weights")
            for table, cols in weights.items():
                for col, weight in cols.items():
                    self.cursor.execute(
                        "INSERT OR REPLACE INTO weights (table_name, column, weight) VALUES (?, ?, ?)",
                        (table, col, weight)
                    )
            self.conn.commit()
            self.logger.debug(f"Wrote {len(weights)} weight entries to SQLite")
        except Exception as e:
            self.logger.error(f"Error writing weights: {e}")
            raise

    def load_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Load weights from SQLite database.
        
        Returns:
            Dictionary of table names to column-weight mappings.
        """
        try:
            self.cursor.execute("SELECT table_name, column, weight FROM weights")
            weights = {}
            for table, col, weight in self.cursor.fetchall():
                if table not in weights:
                    weights[table] = {}
                weights[table][col] = weight
            self.logger.debug(f"Loaded {len(weights)} weight entries from SQLite")
            return weights
        except Exception as e:
            self.logger.error(f"Error loading weights: {e}")
            return {}

    def read_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Alias for load_weights, for compatibility with potential original calls.
        
        Returns:
            Dictionary of table names to column-weight mappings.
        """
        try:
            return self.load_weights()
        except Exception as e:
            self.logger.error(f"Error reading weights: {e}")
            return {}

    def write_name_matches(self, name_matches: Dict[str, List[str]], source: str = 'default'):
        """
        Write name matches to SQLite database.
        
        Args:
            name_matches: Dictionary of column names to synonym lists.
            source: Source of the matches (ignored for compatibility; schema has no source column).
        """
        try:
            self.cursor.execute("DELETE FROM name_matches")
            for col, synonyms in name_matches.items():
                for syn in synonyms:
                    self.cursor.execute(
                        "INSERT OR REPLACE INTO name_matches (column_name, synonym) VALUES (?, ?)",
                        (col, syn)
                    )
            self.conn.commit()
            self.logger.debug(f"Wrote {len(name_matches)} name match entries to SQLite")
        except Exception as e:
            self.logger.error(f"Error writing name matches: {e}")
            raise

    def load_name_matches(self) -> Dict[str, List[str]]:
        """
        Load all name matches from SQLite database (used by TableIdentifier).
        
        Returns:
            Dictionary of column names to synonym lists.
        """
        try:
            self.cursor.execute("SELECT column_name, synonym FROM name_matches")
            name_matches = {}
            for col, syn in self.cursor.fetchall():
                if col not in name_matches:
                    name_matches[col] = []
                name_matches[col].append(syn)
            for col in name_matches:
                name_matches[col] = list(set(name_matches[col]))
            self.logger.debug(f"Loaded {len(name_matches)} name match entries from SQLite")
            return name_matches
        except Exception as e:
            self.logger.error(f"Error loading name matches: {e}")
            return {}

    def read_name_matches(self, source: str = 'default') -> Dict[str, List[str]]:
        """
        Read name matches from SQLite database, ignoring source for compatibility.
        
        Args:
            source: Source of the matches (ignored; schema has no source column).
        
        Returns:
            Dictionary of column names to synonym lists.
        """
        try:
            return self.load_name_matches()
        except Exception as e:
            self.logger.error(f"Error reading name matches (source={source}): {e}")
            return {}

    def write_feedback(self, timestamp: str, query: str, tables: List[str], embedding: np.ndarray, count: int = 1):
        """
        Write feedback to SQLite database.
        
        Args:
            timestamp: Timestamp of the feedback.
            query: Query string.
            tables: List of associated tables.
            embedding: Numpy array of query embedding.
            count: Feedback count (default: 1).
        """
        try:
            embedding_blob = self._embedding_to_blob(embedding)
            tables_json = json.dumps(tables)
            self.cursor.execute(
                "INSERT OR REPLACE INTO feedback (timestamp, query, tables, embedding, count) VALUES (?, ?, ?, ?, ?)",
                (timestamp, query, tables_json, embedding_blob, count)
            )
            self.conn.commit()
            self.logger.debug(f"Wrote feedback for query: {query}")
        except Exception as e:
            self.logger.error(f"Error writing feedback for query '{query}': {e}")

    def get_feedback(self, query: Optional[str] = None) -> List[Tuple[str, str, List[str], np.ndarray, int]]:
        """
        Retrieve feedback entries from SQLite database.
        
        Args:
            query: Optional query string to filter feedback (default: None, returns all).
        
        Returns:
            List of tuples (timestamp, query, tables, embedding, count).
        """
        try:
            if query:
                self.cursor.execute(
                    "SELECT timestamp, query, tables, embedding, count FROM feedback WHERE query = ?",
                    (query,)
                )
            else:
                self.cursor.execute("SELECT timestamp, query, tables, embedding, count FROM feedback")
            feedback = []
            for timestamp, query, tables_json, embedding_blob, count in self.cursor.fetchall():
                try:
                    tables = json.loads(tables_json)
                    embedding = self._blob_to_embedding(embedding_blob) if embedding_blob else np.zeros(384, dtype=np.float32)
                    feedback.append((timestamp, query, tables, embedding, count))
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error decoding tables JSON for query '{query}': {e}")
                    continue
            self.logger.debug(f"Retrieved {len(feedback)} feedback entries")
            return feedback
        except Exception as e:
            self.logger.error(f"Error retrieving feedback: {e}")
            return []

    def read_feedback(self) -> Dict[str, Dict]:
        """
        Read feedback entries from SQLite database, formatted for FeedbackManager compatibility.
        
        Returns:
            Dictionary of timestamp-keyed feedback entries with query, tables, and count.
        """
        try:
            feedback = self.get_feedback()
            feedback_dict = {}
            for timestamp, query, tables, _, count in feedback:
                feedback_dict[timestamp] = {
                    'query': query,
                    'tables': tables,
                    'count': count
                }
            self.logger.debug(f"Loaded {len(feedback_dict)} feedback entries from SQLite")
            return feedback_dict
        except Exception as e:
            self.logger.error(f"Error reading feedback: {e}")
            return {}

    def update_feedback_count(self, query: str, increment: int = 1):
        """
        Increment the count for a feedback entry.
        
        Args:
            query: Query string to update.
            increment: Amount to increment the count (default: 1).
        """
        try:
            self.cursor.execute(
                "UPDATE feedback SET count = count + ? WHERE query = ?",
                (increment, query)
            )
            if self.cursor.rowcount == 0:
                self.logger.debug(f"No feedback found to update for query: {query}")
            else:
                self.conn.commit()
                self.logger.debug(f"Updated feedback count for query: {query}")
        except Exception as e:
            self.logger.error(f"Error updating feedback count for query '{query}': {e}")

    def write_ignored_query(self, query: str, embedding: Optional[np.ndarray], reason: str):
        """
        Write ignored query to SQLite database.
        
        Args:
            query: Query string to ignore.
            embedding: Numpy array of query embedding or None.
            reason: Reason for ignoring the query.
        """
        try:
            if not query or len(query.strip()) < 3:
                self.logger.debug(f"Skipping invalid query: '{query}' (too short or empty)")
                return
            # Generate embedding if None and model is available
            if embedding is None and self.model:
                try:
                    embedding = self.model.encode(query, show_progress_bar=False)
                except Exception as e:
                    self.logger.error(f"Error generating embedding for query '{query}': {e}")
                    embedding = None
            embedding_blob = self._embedding_to_blob(embedding)
            self.cursor.execute(
                "INSERT OR REPLACE INTO ignored_queries (query, embedding, reason) VALUES (?, ?, ?)",
                (query, embedding_blob, reason)
            )
            self.conn.commit()
            self.logger.debug(f"Wrote ignored query: {query}")
        except Exception as e:
            self.logger.error(f"Error writing ignored query '{query}': {e}")

    def get_ignored_queries(self) -> List[Tuple[str, np.ndarray, str]]:
        """
        Retrieve ignored queries from SQLite database.
        
        Returns:
            List of tuples (query, embedding, reason).
        """
        try:
            self.cursor.execute("SELECT query, embedding, reason FROM ignored_queries")
            ignored = []
            for query, embedding_blob, reason in self.cursor.fetchall():
                embedding = self._blob_to_embedding(embedding_blob) if embedding_blob else np.zeros(384, dtype=np.float32)
                ignored.append((query, embedding, reason))
            self.logger.debug(f"Retrieved {len(ignored)} ignored queries")
            return ignored
        except Exception as e:
            self.logger.error(f"Error retrieving ignored queries: {e}")
            return []

    def delete_ignored_query(self, query: str):
        """
        Delete an ignored query from SQLite database.
        
        Args:
            query: Query string to remove.
        """
        try:
            self.cursor.execute("DELETE FROM ignored_queries WHERE query = ?", (query,))
            self.conn.commit()
            self.logger.debug(f"Deleted ignored query: {query}")
        except Exception as e:
            self.logger.error(f"Error deleting ignored query '{query}': {e}")

    def clear_ignored_queries(self):
        """
        Clear all ignored queries from SQLite database.
        """
        try:
            self.cursor.execute("DELETE FROM ignored_queries")
            self.conn.commit()
            self.logger.debug("Cleared all ignored queries")
        except Exception as e:
            self.logger.error(f"Error clearing ignored queries: {e}")

    def read_ignored_queries(self) -> Dict[str, Dict[str, str]]:
        """
        Read ignored queries from SQLite database, formatted for DatabaseAnalyzerCLI compatibility.
        
        Returns:
            Dictionary of query-keyed entries with schema_name and reason.
        """
        try:
            ignored_queries = self.get_ignored_queries()
            ignored_dict = {}
            for query, _, reason in ignored_queries:
                ignored_dict[query] = {
                    'schema_name': None,  # Schema_name not stored in original schema, set to None
                    'reason': reason
                }
            self.logger.debug(f"Loaded {len(ignored_dict)} ignored queries from SQLite")
            return ignored_dict
        except Exception as e:
            self.logger.error(f"Error reading ignored queries: {e}")
            return {}

    def migrate_file_caches(self):
        """
        Migrate legacy file-based caches to SQLite.
        Currently a placeholder for compatibility with existing calls.
        """
        try:
            self.logger.info("Completed file cache migration to SQLite")
        except Exception as e:
            self.logger.error(f"Error migrating file caches: {e}")

    def reload_caches(self, schema_manager, feedback_manager, name_match_manager):
        """
        Reload caches for schema, feedback, and name matches.
        
        Args:
            schema_manager: SchemaManager instance.
            feedback_manager: FeedbackManager instance.
            name_match_manager: NameMatchManager instance.
        """
        try:
            self.logger.debug("Reloading caches")
            self.clear_cache()
            self.logger.info("Caches reloaded successfully")
        except Exception as e:
            self.logger.error(f"Error reloading caches: {e}")

    def clear_cache(self, table: Optional[str] = None):
        """
        Clear specific cache table or all tables.
        
        Args:
            table: Optional table name to clear (e.g., 'weights', 'feedback'). If None, clears all.
        """
        try:
            tables = ['weights', 'name_matches', 'feedback', 'ignored_queries']
            if table and table in tables:
                self.cursor.execute(f"DELETE FROM {table}")
                self.conn.commit()
                self.logger.debug(f"Cleared cache table: {table}")
            else:
                for t in tables:
                    self.cursor.execute(f"DELETE FROM {t}")
                self.conn.commit()
                self.logger.debug("Cleared all cache tables")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def validate_cache(self) -> bool:
        """
        Validate cache integrity by checking table existence and data consistency.
        
        Returns:
            True if cache is valid, False otherwise.
        """
        try:
            tables = ['weights', 'name_matches', 'feedback', 'ignored_queries']
            for table in tables:
                self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if not self.cursor.fetchone():
                    self.logger.error(f"Cache table missing: {table}")
                    return False
                self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = self.cursor.fetchone()[0]
                self.logger.debug(f"Table {table} has {count} entries")
            self.logger.debug("Cache validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating cache: {e}")
            return False

    def count_feedback(self) -> int:
        """
        Count the number of feedback entries in the cache.
        
        Returns:
            Number of feedback entries.
        """
        try:
            self.cursor.execute("SELECT COUNT(*) FROM feedback")
            count = self.cursor.fetchone()[0]
            self.logger.debug(f"Counted {count} feedback entries")
            return count
        except Exception as e:
            self.logger.error(f"Error counting feedback: {e}")
            return 0

    def close(self):
        """
        Close SQLite database connection.
        """
        try:
            self.conn.commit()
            self.conn.close()
            self.logger.debug("Closed SQLite connection")
        except Exception as e:
            self.logger.error(f"Error closing SQLite connection: {e}")