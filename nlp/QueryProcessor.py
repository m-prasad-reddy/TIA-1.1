# nlp/QueryProcessor.py: Processes natural language queries into SQL
# Updated to integrate is_query_ignored, enhance feedback handling, and preserve NLP functionality

import os
import logging
import logging.config
from typing import Dict, List, Tuple, Optional
from analysis.table_identifier import TableIdentifier
from analysis.name_match_manager import NameMatchManager
from analysis.processor import NLPPipeline
from config.patterns import PatternManager
from config.manager import DatabaseConnection
from config.cache_synchronizer import CacheSynchronizer
from config.model_singleton import ModelSingleton
import numpy as np
from datetime import datetime

class QueryProcessor:
    """Processes natural language queries into SQL."""
    
    def __init__(
        self,
        connection_manager: DatabaseConnection,
        schema_dict: Dict,
        nlp_pipeline: NLPPipeline,
        table_identifier: TableIdentifier,
        name_matcher: NameMatchManager,
        pattern_manager: PatternManager,
        db_name: str,
        cache_synchronizer: CacheSynchronizer
    ):
        """Initialize with required components."""
        # Ensure logs directory exists
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
                logging.warning(f"Logging config file not found: {logging_config_path}. Using default logging.")
        except Exception as e:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(os.path.join("logs", "bikestores_app.log")),
                    logging.StreamHandler()
                ]
            )
            logging.error(f"Error loading logging config: {e}. Using default logging.")
        
        self.logger = logging.getLogger("query_processor")
        self.connection_manager = connection_manager
        self.schema_dict = schema_dict
        self.nlp_pipeline = nlp_pipeline
        self.table_identifier = table_identifier
        self.name_matcher = name_matcher
        self.pattern_manager = pattern_manager
        self.db_name = db_name
        self.cache_synchronizer = cache_synchronizer
        self.model = ModelSingleton().model
        self.logger.debug(f"Initialized QueryProcessor for {db_name}")

    def process_query(self, query: str) -> Tuple[Optional[List[str]], bool]:
        """Process a natural language query."""
        self.logger.debug(f"Processing query: {query}")
        
        # Check for empty or short queries
        if not query or len(query.strip()) < 3:
            self.logger.debug(f"Ignoring query '{query}': too short or empty")
            return None, False
        
        # Check if query matches ignored queries
        ignored_reason = self.cache_synchronizer.is_query_ignored(query, threshold=0.85)
        if ignored_reason:
            self.logger.debug(f"Ignoring query '{query}': {ignored_reason}")
            return None, False
        
        # Identify tables
        tables, confidence = self.table_identifier.identify_tables(query)
        if not tables:
            self.logger.warning(f"No tables identified for query: {query}")
            normalized_query = self.cache_synchronizer.normalize_query(query)
            embedding = self.model.encode(normalized_query, show_progress_bar=False) if self.model else None
            self.cache_synchronizer.write_ignored_query(query, embedding, "No relevant tables found")
            return None, False

        # NLP analysis
        analysis = self.nlp_pipeline.analyze_query(query)
        tokens = analysis["tokens"]
        for token in tokens:
            for table in tables:
                schema, tbl = table.split('.')
                columns = self.schema_dict['columns'][schema][tbl]
                self.name_matcher.update_synonyms([token], self.name_matcher.get_token_embeddings([token]), columns)

        # Update feedback
        normalized_query = self.cache_synchronizer.normalize_query(query)
        timestamp = datetime.now().isoformat()
        embedding = self.model.encode(normalized_query, show_progress_bar=False) if self.model else np.zeros(384, dtype=np.float32)
        self.cache_synchronizer.write_feedback(timestamp, normalized_query, tables, embedding)
        
        # Update weights and synonyms
        self.table_identifier.update_weights_from_feedback(query, tables)
        self.name_matcher._load_dynamic()  # Reload synonym cache
        self.name_matcher.synonym_cache = self.name_matcher._merge_matches()
        
        self.logger.info(f"Identified tables: {tables}, Confidence: {confidence}")
        return tables, confidence