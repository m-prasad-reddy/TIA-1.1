# analysis/processor.py: Processes natural language queries with patterns and NLP
# Fixes pattern loading error, ~180 lines

import logging
import logging.config
import os
import re
from typing import Dict, List, Optional
import spacy
from config.patterns import PatternManager

class NLPPipeline:
    """
    Processes natural language queries using patterns and spaCy for tokenization.
    Matches queries to tables based on regex patterns and synonyms.
    """
    
    def __init__(self, pattern_manager: PatternManager, db_name: str):
        """
        Initialize NLPPipeline with pattern manager and database name.
        
        Args:
            pattern_manager: PatternManager instance for regex patterns.
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
        
        self.logger = logging.getLogger("nlp_pipeline")
        self.pattern_manager = pattern_manager
        self.db_name = db_name
        self.nlp = None
        self.patterns = []
        self._load_patterns()
        self._load_spacy()
        self.logger.debug("Initialized NLPPipeline")

    def _load_spacy(self):
        """
        Load spaCy model for tokenization.
        """
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.debug("Loaded spacy model: en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Error loading spaCy model: {e}")
            self.nlp = None

    def _load_patterns(self):
        """
        Load and compile patterns from PatternManager.
        """
        try:
            self.logger.debug("Loading patterns")
            patterns = self.pattern_manager.get_patterns()
            self.patterns = []
            for pattern, tables in patterns.items():
                try:
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                    self.patterns.append((compiled_pattern, tables))
                    self.logger.debug(f"Loaded pattern '{pattern}' with tables: {tables}")
                except re.error as e:
                    self.logger.error(f"Error compiling pattern '{pattern}': {e}")
            self.logger.debug(f"Loaded {len(self.patterns)} patterns")
        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")
            self.patterns = []

    def process_query(self, query: str) -> List[str]:
        """
        Process a natural language query to identify relevant tables.
        
        Args:
            query: Query string to process.
        
        Returns:
            List of table names matched to the query.
        """
        try:
            if not self.nlp:
                self.logger.error("spaCy model not loaded")
                return []
            
            doc = self.nlp(query)
            matched_tables = set()
            
            # Match patterns
            for pattern, tables in self.patterns:
                if pattern.search(query.lower()):
                    matched_tables.update(tables)
                    self.logger.debug(f"Pattern match: '{pattern.pattern}' -> {tables}")
            
            # Match tokens
            for token in doc:
                token_text = token.text.lower()
                for pattern, tables in self.patterns:
                    if pattern.match(token_text):
                        matched_tables.update(tables)
                        self.logger.debug(f"Token match: '{token_text}' -> {tables}")
            
            return list(matched_tables)
        except Exception as e:
            self.logger.error(f"Error processing query '{query}': {e}")
            return []

    def add_pattern(self, pattern: str, tables: List[str]):
        """
        Add a new pattern to the pipeline.
        
        Args:
            pattern: Regex pattern string.
            tables: List of table names.
        """
        try:
            self.pattern_manager.add_pattern(pattern, tables)
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            self.patterns.append((compiled_pattern, tables))
            self.logger.debug(f"Added pattern '{pattern}' with tables: {tables}")
        except Exception as e:
            self.logger.error(f"Error adding pattern '{pattern}': {e}")

    def clear_patterns(self):
        """
        Clear all patterns from the pipeline.
        """
        try:
            self.patterns = []
            self.pattern_manager.patterns = {}
            self.logger.debug("Cleared all patterns")
        except Exception as e:
            self.logger.error(f"Error clearing patterns: {e}")