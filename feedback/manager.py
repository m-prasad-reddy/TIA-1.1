# feedback/manager.py: Manages feedback for TableIdentifier-v1
# Original file, fully functional with validation and embeddings, ~167 lines

import os
import json
import logging
import logging.config
from typing import Dict, List, Optional
from config.cache_synchronizer import CacheSynchronizer
import numpy as np
from datetime import datetime
from sentence_transformers import util
from config.model_singleton import ModelSingleton

class FeedbackManager:
    """Manages feedback for query-table associations in TableIdentifier-v1."""
    
    def __init__(self, db_name: str, cache_synchronizer: Optional[CacheSynchronizer] = None):
        """Initialize with database name and optional cache synchronizer."""
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Configure logging
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
        
        self.logger = logging.getLogger("feedback")
        self.db_name = db_name
        self.cache_synchronizer = cache_synchronizer or CacheSynchronizer(db_name)
        self.feedback_cache_dir = os.path.join("feedback_cache", db_name)
        self.feedback = {}
        self.model = ModelSingleton().model
        self.logger.debug(f"Initialized FeedbackManager for {db_name}")

    def store_feedback(self, query: str, tables: List[str], schema_dict: Optional[Dict] = None):
        """Store feedback for a query and its identified tables."""
        try:
            if not query or not tables:
                self.logger.error("Empty query or tables provided")
                return

            # Validate tables if schema_dict is provided
            if schema_dict:
                valid_tables = []
                schema_map = {s.lower(): s for s in schema_dict.get('tables', {})}
                for table in tables:
                    if '.' not in table:
                        self.logger.warning(f"Invalid table format: {table}")
                        continue
                    schema, table_name = table.split('.')
                    schema_lower = schema.lower()
                    if (schema_lower in schema_map and 
                        table_name.lower() in {t.lower() for t in schema_dict['tables'].get(schema_map[schema_lower], [])}):
                        valid_tables.append(f"{schema_map[schema_lower]}.{table_name}")
                    else:
                        self.logger.warning(f"Invalid table: {table}")
                tables = valid_tables

            if not tables:
                self.logger.error("No valid tables after validation")
                return

            # Generate embedding
            try:
                embedding = self.model.encode(query, show_progress_bar=False)
            except Exception as e:
                self.logger.error(f"Error generating embedding for query '{query}': {e}")
                embedding = np.array([])

            # Store in SQLite
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.cache_synchronizer.write_feedback(timestamp, query, tables, embedding, count=1)
            self.feedback[timestamp] = {'query': query, 'tables': tables, 'count': 1}
            self.logger.debug(f"Stored feedback: {query} -> {tables}")
        except Exception as e:
            self.logger.error(f"Error storing feedback for query '{query}': {e}")

    def get_all_feedback(self) -> List[Dict]:
        """Retrieve all feedback entries."""
        try:
            feedback_data = self.cache_synchronizer.read_feedback()
            result = [
                {
                    'timestamp': ts,
                    'query': data['query'],
                    'tables': data['tables'],
                    'count': data['count']
                }
                for ts, data in feedback_data.items()
            ]
            self.logger.debug(f"Retrieved {len(result)} feedback entries")
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving feedback: {e}")
            return []

    def get_top_queries(self, limit: int = 5) -> List[str]:
        """Get top example queries based on feedback frequency."""
        try:
            feedback_data = self.cache_synchronizer.read_feedback()
            query_counts = {}
            for data in feedback_data.values():
                query = data['query']
                count = data.get('count', 1)
                query_counts[query] = query_counts.get(query, 0) + count
            
            sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
            top_queries = [query for query, _ in sorted_queries[:limit]]
            self.logger.debug(f"Retrieved top {len(top_queries)} queries")
            return top_queries
        except Exception as e:
            self.logger.error(f"Error retrieving top queries: {e}")
            return []

    def delete_feedback(self, query: str):
        """Delete feedback for a specific query."""
        try:
            feedback_data = self.cache_synchronizer.read_feedback()
            for ts, data in list(feedback_data.items()):
                if data['query'].lower() == query.lower():
                    del feedback_data[ts]
                    self.logger.debug(f"Deleted feedback for query: {query}")
            
            # Update SQLite
            with sqlite3.connect(self.cache_synchronizer.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM feedback WHERE query = ?", (query,))
                conn.commit()
                self.logger.debug(f"Removed feedback for query '{query}' from SQLite")
        except Exception as e:
            self.logger.error(f"Error deleting feedback for query '{query}': {e}")

    def clear_feedback(self):
        """Clear all feedback data."""
        try:
            self.feedback = {}
            with sqlite3.connect(self.cache_synchronizer.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM feedback")
                conn.commit()
                self.logger.debug("Cleared all feedback from SQLite")
        except Exception as e:
            self.logger.error(f"Error clearing feedback: {e}")