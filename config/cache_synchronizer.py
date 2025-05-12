# config/cache_synchronizer.py: Manages cache operations for TableIdentifier-v2.1
# Optimized deduplicate_feedback to prevent hanging with single-transaction approach

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
    Synchronizes cache data (weights, name matches, feedback, ignored queries) with SQLite database.
    Provides thread-safe methods with optimized deduplication and deadlock prevention.
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
        self.lock = threading.Lock()
        self.deduplication_lock = threading.Lock()  # Separate lock for deduplication
        self._conn = None  # Persistent connection
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            self.logger.debug(f"Starting SQLite database initialization at {self.db_path}")
            self._validate_db_file()
            self._initialize_database()
            self.logger.debug(f"Completed SQLite database initialization at {self.db_path}")
            self.logger.info(f"Initialized CacheSynchronizer for {db_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite database: {e}")
            self._fallback_to_memory_db()
            self.logger.warning("Using in-memory SQLite database as fallback")

    def _validate_db_file(self):
        """
        Validate that the database file is accessible and not corrupted.
        """
        try:
            if os.path.exists(self.db_path):
                if not os.access(self.db_path, os.R_OK | os.W_OK):
                    self.logger.error(f"Database file {self.db_path} is not readable/writable")
                    raise PermissionError(f"Insufficient permissions for {self.db_path}")
                conn = sqlite3.connect(self.db_path, timeout=5)
                conn.execute("PRAGMA integrity_check")
                conn.close()
                self.logger.debug(f"Validated database file {self.db_path}")
            else:
                self.logger.debug(f"Database file {self.db_path} does not exist, will be created")
        except Exception as e:
            self.logger.error(f"Database file validation failed: {e}")
            if os.path.exists(self.db_path):
                backup_path = self.db_path + ".bak"
                os.rename(self.db_path, backup_path)
                self.logger.info(f"Backed up corrupted database to {backup_path}")
            raise

    def _fallback_to_memory_db(self):
        """
        Switch to an in-memory SQLite database if file-based database fails.
        """
        try:
            self.db_path = ":memory:"
            self._conn = sqlite3.connect(self.db_path, timeout=30)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=15000")
            self._initialize_database()
            self.logger.info("Initialized in-memory SQLite database")
        except Exception as e:
            self.logger.error(f"Failed to initialize in-memory database: {e}")
            raise RuntimeError("Cannot initialize SQLite database")

    def _get_connection(self):
        """
        Get or create a persistent SQLite connection with optimized settings.
        
        Returns:
            SQLite connection object.
        """
        with self.lock:
            if self._conn is None or getattr(self._conn, 'closed', True):
                try:
                    if sqlite3.connect is None:
                        self.logger.error("sqlite3.connect is None, cannot create connection")
                        raise ValueError("sqlite3.connect is None")
                    start_time = time.time()
                    self._conn = sqlite3.connect(self.db_path, timeout=10)
                    if time.time() - start_time > 10:
                        self.logger.warning("SQLite connection took longer than expected")
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA synchronous=NORMAL")
                    self._conn.execute("PRAGMA busy_timeout=15000")
                    self.logger.debug(f"Created and configured SQLite connection to {self.db_path}")
                except Exception as e:
                    self.logger.error(f"Error creating SQLite connection: {e}")
                    raise
            return self._conn

    def _initialize_database(self):
        """
        Create necessary tables sequentially with immediate commits and PRIMARY KEY.
        """
        try:
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
                    count INTEGER,
                    PRIMARY KEY (timestamp, query)
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
            
            self._create_feedback_indexes()
            self._checkpoint_wal(conn)
        except Exception as e:
            self.logger.error(f"Error initializing database tables: {e}")
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()

    def _create_feedback_indexes(self):
        """
        Create indexes on feedback.query and feedback.timestamp with fallback.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            self.logger.debug("Attempting to create index idx_feedback_query")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query)")
            conn.commit()
            self.logger.debug("Successfully created index idx_feedback_query")
            
            self.logger.debug("Attempting to create index idx_feedback_timestamp")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)")
            conn.commit()
            self.logger.debug("Successfully created index idx_feedback_timestamp")
        except Exception as e:
            self.logger.warning(f"Failed to create feedback indexes: {e}")
            self.logger.info("Proceeding without indexes; performance may be affected")
        finally:
            if 'cursor' in locals():
                cursor.close()

    def _checkpoint_wal(self, conn, mode: str = 'TRUNCATE'):
        """
        Perform a WAL checkpoint to manage WAL file size.
        
        Args:
            conn: SQLite connection object.
            mode: Checkpoint mode ('PASSIVE', 'TRUNCATE', 'FULL').
        """
        try:
            self.logger.debug(f"Performing {mode} WAL checkpoint")
            conn.execute(f"PRAGMA wal_checkpoint({mode})")
            self.logger.debug(f"{mode} WAL checkpoint completed")
        except Exception as e:
            self.logger.warning(f"Error performing WAL checkpoint: {e}")

    def normalize_query(self, query: str) -> str:
        """
        Normalize query for consistent storage and comparison.
        """
        try:
            query = re.sub(r'\s+', ' ', query.strip().lower())
            query = re.sub(r'[^\w\s]', '', query)
            return query
        except Exception as e:
            self.logger.error(f"Error normalizing query '{query}': {e}")
            return query.lower()

    def _embedding_to_blob(self, embedding: Optional[np.ndarray]) -> Optional[bytes]:
        """
        Convert numpy embedding to SQLite BLOB.
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
        Convert SQLite BLOB to numpy embedding with dimension validation.
        """
        try:
            embedding = np.frombuffer(blob, dtype=np.float32)
            if len(embedding) != dim:
                self.logger.error(f"Invalid embedding dimension: {len(embedding)}, expected {dim}")
                return np.zeros(dim, dtype=np.float32)
            return embedding.reshape(-1)
        except Exception as e:
            self.logger.error(f"Error converting blob to embedding: {e}")
            return np.zeros(dim, dtype=np.float32)

    def execute_with_retry(self, query: str, params: tuple = (), max_attempts: int = 5) -> None:
        """
        Execute a SQL query with retry logic to handle database lock.
        """
        attempt = 1
        while attempt <= max_attempts:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                self.logger.debug(f"Executing query: {query} with params: {params}")
                cursor.execute("BEGIN TRANSACTION")
                cursor.execute(query, params)
                conn.commit()
                self.logger.debug(f"Successfully executed query: {query}")
                return
            except sqlite3.OperationalError as e:
                conn.rollback()
                if "database is locked" in str(e) and attempt < max_attempts:
                    self.logger.warning(f"Database locked on attempt {attempt}, retrying in {0.5 * attempt}s...")
                    time.sleep(0.5 * attempt)
                    attempt += 1
                else:
                    self.logger.error(f"Error executing query '{query}': {e}")
                    raise
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Unexpected error executing query '{query}': {e}")
                raise
            finally:
                if 'cursor' in locals():
                    cursor.close()

    def deduplicate_feedback(self, timeout: float = 60.0) -> int:
        """
        Remove duplicate feedback entries for the same normalized query in a single transaction.
        
        Args:
            timeout: Maximum time (seconds) to wait for deduplication.
        
        Returns:
            Number of duplicate entries removed.
        """
        try:
            start_time = time.time()
            if not self.deduplication_lock.acquire(timeout=timeout):
                self.logger.error("Failed to acquire deduplication lock within timeout")
                raise TimeoutError("Deduplication lock timeout")
            
            conn = self._get_connection()
            cursor = conn.cursor()
            self.logger.debug("Starting feedback deduplication")
            
            # Perform TRUNCATE WAL checkpoint to clear WAL file
            self._checkpoint_wal(conn, mode='TRUNCATE')
            
            # Count total feedback entries
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_entries = cursor.fetchone()[0]
            self.logger.debug(f"Found {total_entries} feedback entries")
            
            if total_entries > 200:
                self.logger.warning(f"Large feedback table ({total_entries} entries), consider clearing")
            
            # Create temporary table for deduplicated entries
            cursor.execute("""
                CREATE TEMPORARY TABLE temp_feedback (
                    timestamp TEXT,
                    query TEXT,
                    tables TEXT,
                    embedding BLOB,
                    count INTEGER,
                    PRIMARY KEY (timestamp, query)
                )
            """)
            self.logger.debug("Created temporary table for deduplication")
            
            # Insert deduplicated entries into temporary table
            cursor.execute("""
                INSERT INTO temp_feedback (timestamp, query, tables, embedding, count)
                SELECT MAX(timestamp) as timestamp, 
                       query, 
                       tables, 
                       embedding, 
                       SUM(count) as count
                FROM feedback
                GROUP BY query
            """)
            cursor.execute("SELECT COUNT(*) FROM temp_feedback")
            deduplicated_count = cursor.fetchone()[0]
            self.logger.debug(f"Inserted {deduplicated_count} deduplicated entries into temp_feedback")
            
            # Calculate removed count
            removed_count = total_entries - deduplicated_count
            self.logger.debug(f"Will remove {removed_count} duplicate entries")
            
            # Replace feedback table with deduplicated data
            cursor.execute("DELETE FROM feedback")
            self.logger.debug("Cleared original feedback table")
            
            cursor.execute("""
                INSERT INTO feedback (timestamp, query, tables, embedding, count)
                SELECT timestamp, query, tables, embedding, count
                FROM temp_feedback
            """)
            self.logger.debug("Copied deduplicated entries back to feedback table")
            
            # Drop temporary table
            cursor.execute("DROP TABLE temp_feedback")
            self.logger.debug("Dropped temporary table")
            
            conn.commit()
            self._checkpoint_wal(conn, mode='TRUNCATE')
            self.logger.info(f"Removed {removed_count} duplicate feedback entries in {time.time() - start_time:.2f}s")
            return removed_count
        except Exception as e:
            self.logger.error(f"Error deduplicating feedback: {e}")
            conn.rollback()
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()
            if self.deduplication_lock.locked():
                self.deduplication_lock.release()

    def write_weights(self, weights: Dict[str, Dict[str, float]], batch_size: int = 100):
        """
        Write weights to SQLite database in batches.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            all_entries = [(table, col, weight) for table, cols in weights.items() for col, weight in cols.items()]
            total_entries = len(all_entries)
            self.logger.debug(f"Writing {total_entries} weight entries to SQLite")
            
            for i in range(0, total_entries, batch_size):
                batch = all_entries[i:i + batch_size]
                self.logger.debug(f"Writing weights batch {i//batch_size + 1} ({len(batch)} entries)")
                cursor.execute("BEGIN TRANSACTION")
                cursor.executemany(
                    "INSERT OR REPLACE INTO weights (table_name, column, weight) VALUES (?, ?, ?)",
                    batch
                )
                conn.commit()
                self.logger.debug(f"Completed weights batch {i//batch_size + 1}")
            
            self._checkpoint_wal(conn)
        except Exception as e:
            self.logger.error(f"Error writing weights: {e}")
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()

    def load_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Load weights from SQLite database.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT table_name, column, weight FROM weights")
            weights = {}
            for table, col, weight in cursor.fetchall():
                if table not in weights:
                    weights[table] = {}
                weights[table][col] = weight
            self.logger.debug(f"Loaded {sum(len(cols) for cols in weights.values())} weight entries from SQLite")
            return weights
        except Exception as e:
            self.logger.error(f"Error loading weights: {e}")
            return {}
        finally:
            if 'cursor' in locals():
                cursor.close()

    def read_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Alias for load_weights.
        """
        return self.load_weights()

    def write_name_matches(self, name_matches: Dict[str, List[str]], source: str = 'default', batch_size: int = 100):
        """
        Write name matches to SQLite database in batches.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            self.logger.debug("Clearing existing name_matches")
            cursor.execute("DELETE FROM name_matches")
            conn.commit()
            
            all_entries = [(col, syn) for col, synonyms in name_matches.items() for syn in synonyms]
            total_entries = len(all_entries)
            self.logger.debug(f"Writing {total_entries} name match entries to SQLite")
            
            for i in range(0, total_entries, batch_size):
                batch = all_entries[i:i + batch_size]
                self.logger.debug(f"Writing name matches batch {i//batch_size + 1} ({len(batch)} entries)")
                cursor.execute("BEGIN TRANSACTION")
                cursor.executemany(
                    "INSERT OR REPLACE INTO name_matches (column_name, synonym) VALUES (?, ?)",
                    batch
                )
                conn.commit()
                self.logger.debug(f"Completed name matches batch {i//batch_size + 1}")
            
            self._checkpoint_wal(conn)
            self.logger.debug(f"Wrote {total_entries} name match entries to SQLite")
        except Exception as e:
            self.logger.error(f"Error writing name matches: {e}")
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()

    def load_name_matches(self) -> Dict[str, List[str]]:
        """
        Load all name matches from SQLite database.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT column_name, synonym FROM name_matches")
            name_matches = {}
            for col, syn in cursor.fetchall():
                if col not in name_matches:
                    name_matches[col] = []
                name_matches[col].append(syn)
            for col in name_matches:
                name_matches[col] = list(set(name_matches[col]))
            self.logger.debug(f"Loaded {sum(len(syns) for syns in name_matches.values())} name match entries from SQLite")
            return name_matches
        except Exception as e:
            self.logger.error(f"Error loading name matches: {e}")
            return {}
        finally:
            if 'cursor' in locals():
                cursor.close()

    def read_name_matches(self, source: str = 'default') -> Dict[str, List[str]]:
        """
        Read name matches from SQLite database.
        """
        return self.load_name_matches()

    def write_feedback(self, timestamp: str, query: str, tables: List[str], embedding: np.ndarray, count: int = 1):
        """
        Write feedback to SQLite database, checking for near-duplicates.
        """
        try:
            normalized_query = self.normalize_query(query)
            similar_feedback = self.find_similar_feedback(query, threshold=0.90)
            for fb_query, fb_tables, sim in similar_feedback:
                if sim > 0.90 and set(fb_tables) == set(tables):
                    self.update_feedback_count(fb_query, increment=count)
                    self.logger.debug(f"Updated existing feedback for query '{fb_query}' (sim={sim:.2f})")
                    return
            
            embedding_blob = self._embedding_to_blob(embedding)
            tables_json = json.dumps(tables)
            self.execute_with_retry(
                "INSERT OR REPLACE INTO feedback (timestamp, query, tables, embedding, count) VALUES (?, ?, ?, ?, ?)",
                (timestamp, normalized_query, tables_json, embedding_blob, count)
            )
            self.logger.debug(f"Wrote feedback for query: {normalized_query}")
        except Exception as e:
            self.logger.error(f"Error writing feedback for query '{query}': {e}")
            raise

    def delete_feedback(self, query: str):
        """
        Delete feedback entry from SQLite database.
        """
        try:
            normalized_query = self.normalize_query(query)
            self.execute_with_retry("DELETE FROM feedback WHERE query = ?", (normalized_query,))
            self.logger.debug(f"Deleted feedback for query: {normalized_query}")
        except Exception as e:
            self.logger.error(f"Error deleting feedback for query '{query}': {e}")

    def get_feedback(self, query: Optional[str] = None) -> List[Tuple[str, str, List[str], np.ndarray, int]]:
        """
        Retrieve feedback entries from SQLite database.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            if query:
                normalized_query = self.normalize_query(query)
                cursor.execute(
                    "SELECT timestamp, query, tables, embedding, count FROM feedback WHERE query = ?",
                    (normalized_query,)
                )
            else:
                cursor.execute("SELECT timestamp, query, tables, embedding, count FROM feedback")
            feedback = []
            for timestamp, query, tables_json, embedding_blob, count in cursor.fetchall():
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
        finally:
            if 'cursor' in locals():
                cursor.close()

    def read_feedback(self) -> Dict[str, Dict]:
        """
        Read feedback entries from SQLite database.
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
        """
        try:
            normalized_query = self.normalize_query(query)
            self.execute_with_retry(
                "UPDATE feedback SET count = count + ? WHERE query = ?",
                (increment, normalized_query)
            )
            self.logger.debug(f"Updated feedback count for query: {normalized_query}")
        except Exception as e:
            self.logger.error(f"Error updating feedback count for query '{query}': {e}")

    def write_ignored_query(self, query: str, embedding: Optional[np.ndarray], reason: str):
        """
        Write ignored query to SQLite database, skipping short queries.
        """
        try:
            if not query or len(query.strip()) < 3:
                self.logger.debug(f"Skipping invalid query: '{query}' (too short or empty)")
                return
            normalized_query = self.normalize_query(query)
            if embedding is None and self.model:
                try:
                    embedding = self.model.encode(normalized_query, show_progress_bar=False)
                except Exception as e:
                    self.logger.error(f"Error generating embedding for query '{normalized_query}': {e}")
                    embedding = None
            embedding_blob = self._embedding_to_blob(embedding)
            self.execute_with_retry(
                "INSERT OR REPLACE INTO ignored_queries (query, embedding, reason) VALUES (?, ?, ?)",
                (normalized_query, embedding_blob, reason)
            )
            self.logger.debug(f"Wrote ignored query: {normalized_query}")
        except Exception as e:
            self.logger.error(f"Error writing ignored query '{query}': {e}")
            raise

    def get_ignored_queries(self) -> List[Tuple[str, np.ndarray, str]]:
        """
        Retrieve ignored queries from SQLite database.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT query, embedding, reason FROM ignored_queries")
            ignored = []
            for query, embedding_blob, reason in cursor.fetchall():
                embedding = self._blob_to_embedding(embedding_blob) if embedding_blob else np.zeros(384, dtype=np.float32)
                ignored.append((query, embedding, reason))
            self.logger.debug(f"Retrieved {len(ignored)} ignored queries")
            return ignored
        except Exception as e:
            self.logger.error(f"Error retrieving ignored queries: {e}")
            return []
        finally:
            if 'cursor' in locals():
                cursor.close()

    def is_query_ignored(self, query: str, threshold: float = 0.85) -> Optional[str]:
        """
        Check if a query matches an ignored query based on embedding similarity.
        
        Args:
            query: The query to check.
            threshold: Similarity threshold for matching.
        
        Returns:
            Reason for ignoring the query if matched, else None.
        """
        try:
            if not self.model:
                self.logger.warning("No model available for ignored query check")
                return None
            normalized_query = self.normalize_query(query)
            query_embedding = self.model.encode(normalized_query, show_progress_bar=False)
            ignored_queries = self.get_ignored_queries()
            for ignored_query, ignored_embedding, reason in ignored_queries:
                similarity = float(util.cos_sim(query_embedding, ignored_embedding)[0][0])
                if similarity > threshold:
                    self.logger.debug(f"Query '{normalized_query}' matches ignored query '{ignored_query}' (sim={similarity:.2f}, reason={reason})")
                    return reason
            self.logger.debug(f"No ignored query match for '{normalized_query}'")
            return None
        except Exception as e:
            self.logger.error(f"Error checking ignored query '{query}': {e}")
            return None

    def delete_ignored_query(self, query: str):
        """
        Delete an ignored query from SQLite database.
        """
        try:
            normalized_query = self.normalize_query(query)
            self.execute_with_retry("DELETE FROM ignored_queries WHERE query = ?", (normalized_query,))
            self.logger.debug(f"Deleted ignored query: {normalized_query}")
        except Exception as e:
            self.logger.error(f"Error deleting ignored query '{query}': {e}")

    def clear_ignored_queries(self):
        """
        Clear all ignored queries from SQLite database.
        """
        try:
            self.execute_with_retry("DELETE FROM ignored_queries")
            self.logger.debug("Cleared all ignored queries")
        except Exception as e:
            self.logger.error(f"Error clearing ignored queries: {e}")

    def read_ignored_queries(self) -> Dict[str, Dict[str, str]]:
        """
        Read ignored queries from SQLite database.
        """
        try:
            ignored_queries = self.get_ignored_queries()
            ignored_dict = {}
            for query, _, reason in ignored_queries:
                ignored_dict[query] = {
                    'schema_name': None,
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
        """
        try:
            config_dir = os.path.join("app-config", self.db_name)
            weights_file = os.path.join(config_dir, "weights.json")
            name_matches_file = os.path.join(config_dir, "name_matches.json")
            feedback_file = os.path.join(config_dir, "feedback.json")
            ignored_file = os.path.join(config_dir, "ignored_queries.json")
            
            if os.path.exists(weights_file):
                try:
                    with open(weights_file) as f:
                        weights = json.load(f)
                    self.write_weights(weights)
                    os.rename(weights_file, weights_file + ".bak")
                    self.logger.debug(f"Migrated weights from {weights_file}")
                except Exception as e:
                    self.logger.warning(f"Error migrating weights: {e}")
            
            if os.path.exists(name_matches_file):
                try:
                    with open(name_matches_file) as f:
                        name_matches = json.load(f)
                    self.write_name_matches(name_matches)
                    os.rename(name_matches_file, name_matches_file + ".bak")
                    self.logger.debug(f"Migrated name matches from {name_matches_file}")
                except Exception as e:
                    self.logger.warning(f"Error migrating name matches: {e}")
            
            if os.path.exists(feedback_file):
                try:
                    with open(feedback_file) as f:
                        feedback = json.load(f)
                    for timestamp, entry in feedback.items():
                        normalized_query = self.normalize_query(entry['query'])
                        embedding = np.zeros(384, dtype=np.float32)
                        self.write_feedback(timestamp, normalized_query, entry['tables'], embedding, entry.get('count', 1))
                    os.rename(feedback_file, feedback_file + ".bak")
                    self.logger.debug(f"Migrated feedback from {feedback_file}")
                except Exception as e:
                    self.logger.warning(f"Error migrating feedback: {e}")
            
            if os.path.exists(ignored_file):
                try:
                    with open(ignored_file) as f:
                        ignored = json.load(f)
                    for query, info in ignored.items():
                        normalized_query = self.normalize_query(query)
                        embedding = np.zeros(384, dtype=np.float32)
                        self.write_ignored_query(normalized_query, embedding, info.get('reason', 'unknown'))
                    os.rename(ignored_file, ignored_file + ".bak")
                    self.logger.debug(f"Migrated ignored queries from {ignored_file}")
                except Exception as e:
                    self.logger.warning(f"Error migrating ignored queries: {e}")
            
            self.logger.info("Completed file cache migration to SQLite")
        except Exception as e:
            self.logger.error(f"Error migrating file caches: {e}")

    def reload_caches(self, schema_manager, feedback_manager, name_match_manager):
        """
        Reload caches for schema, feedback, and name matches.
        """
        try:
            self.logger.debug("Reloading caches")
            self.clear_cache(table='weights')
            self.clear_cache(table='name_matches')
            self.clear_cache(table='ignored_queries')
            self.logger.debug("Preserving feedback during cache reload")
            self.logger.info("Caches reloaded successfully")
        except Exception as e:
            self.logger.error(f"Error reloading caches: {e}")

    def clear_cache(self, table: Optional[str] = None):
        """
        Clear specific cache table or all tables except feedback.
        """
        try:
            tables = ['weights', 'name_matches', 'ignored_queries']
            if table and table in tables:
                self.execute_with_retry(f"DELETE FROM {table}")
                self.logger.debug(f"Cleared cache table: {table}")
            else:
                for t in tables:
                    self.execute_with_retry(f"DELETE FROM {t}")
                self.logger.debug("Cleared weights, name_matches, and ignored_queries tables")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def validate_cache(self) -> bool:
        """
        Validate cache integrity by checking table existence and data consistency.
        """
        try:
            tables = ['weights', 'name_matches', 'feedback', 'ignored_queries']
            conn = self._get_connection()
            cursor = conn.cursor()
            for table in tables:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if not cursor.fetchone():
                    self.logger.error(f"Cache table missing: {table}")
                    return False
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                self.logger.debug(f"Table {table} has {count} entries")
            self.logger.debug("Cache validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating cache: {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()

    def count_feedback(self) -> int:
        """
        Count the number of feedback entries in the cache.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM feedback")
            count = cursor.fetchone()[0]
            self.logger.debug(f"Counted {count} feedback entries")
            return count
        except Exception as e:
            self.logger.error(f"Error counting feedback: {e}")
            return 0
        finally:
            if 'cursor' in locals():
                cursor.close()

    def find_similar_feedback(self, query: str, threshold: float = 0.90) -> List[Tuple[str, List[str], float]]:
        """
        Find feedback entries similar to the given query based on embedding similarity.
        """
        try:
            if not self.model:
                self.logger.warning("No model available for similarity comparison")
                return []
            normalized_query = self.normalize_query(query)
            query_embedding = self.model.encode(normalized_query, show_progress_bar=False)
            feedback = self.get_feedback()
            similar = []
            for _, fb_query, tables, fb_embedding, _ in feedback:
                similarity = float(util.cos_sim(query_embedding, fb_embedding)[0][0])
                if similarity > threshold:
                    similar.append((fb_query, tables, similarity))
            similar.sort(key=lambda x: x[2], reverse=True)
            self.logger.debug(f"Found {len(similar)} similar feedback entries for query: {normalized_query}")
            return similar
        except Exception as e:
            self.logger.error(f"Error finding similar feedback for query '{query}': {e}")
            return []

    def close(self):
        """
        Close SQLite database connection and deduplicate feedback, idempotent.
        """
        try:
            with self.lock:
                if self._conn is not None and not getattr(self._conn, 'closed', True):
                    try:
                        self.deduplicate_feedback(timeout=60.0)
                        self._checkpoint_wal(self._conn)
                        self._conn.commit()
                    finally:
                        self._conn.close()
                        self._conn = None
                        self.logger.debug("Closed SQLite connection and deduplicated feedback")
                else:
                    self.logger.debug("No open SQLite connection to close")
        except Exception as e:
            self.logger.error(f"Error closing CacheSynchronizer: {e}")