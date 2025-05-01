# analysis/table_identifier.py: Identifies tables from queries for TableIdentifier-v1
# Enhances production.stocks for stock/availability, ~208 lines

import logging
import logging.config
import os
import re
from typing import Dict, List, Tuple, Set
import spacy
from sentence_transformers import util
from config.model_singleton import ModelSingleton
from config.patterns import PatternManager
from feedback.manager import FeedbackManager
from config.cache_synchronizer import CacheSynchronizer

class TableIdentifier:
    """
    Identifies relevant tables from natural language queries using patterns,
    semantic similarity, weights, and name matches.
    """
    
    def __init__(self, schema_dict: Dict, feedback_manager: FeedbackManager, 
                 pattern_manager: PatternManager, cache_synchronizer: CacheSynchronizer):
        """
        Initialize TableIdentifier with schema, feedback, patterns, and cache.
        
        Args:
            schema_dict: Dictionary of schema metadata (tables, columns, etc.).
            feedback_manager: Manages user feedback for queries.
            pattern_manager: Manages regex patterns for query matching.
            cache_synchronizer: Handles cache operations (weights, name matches).
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
        
        self.logger = logging.getLogger("table_identifier")
        self.schema_dict = schema_dict
        self.feedback_manager = feedback_manager
        self.pattern_manager = pattern_manager
        self.cache_synchronizer = cache_synchronizer
        self.model = ModelSingleton().model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.debug("Loaded spacy model: en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to load spacy model: {e}")
            raise RuntimeError(f"Spacy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
        self.weights = self.cache_synchronizer.load_weights() or {}
        self.name_matches = self.cache_synchronizer.load_name_matches() or {}
        self.similarity_threshold = 0.6
        self.max_confidence = 1.0
        self.logger.debug("Initialized TableIdentifier")

    def identify_tables(self, query: str, column_scores: Dict[str, float]) -> Tuple[List[str], float]:
        """
        Identify tables relevant to a query using multiple strategies and column scores.
        
        Args:
            query: Natural language query string.
            column_scores: Dictionary of column paths (schema.table.column) to scores.
        
        Returns:
            Tuple of (list of table names, confidence score).
        """
        try:
            self.logger.debug(f"Processing query: {query}")
            query_lower = query.lower().strip()
            doc = self.nlp(query_lower)
            tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
            matched_tables: Set[str] = set()
            confidence = 0.0

            # Pattern-based matching
            patterns = self.pattern_manager.get_patterns()
            for pattern, tables in patterns.items():
                if re.search(pattern, query_lower):
                    matched_tables.update(tables)
                    confidence = max(confidence, 0.7)
                    self.logger.debug(f"Pattern match: '{pattern}' -> {tables}")

            # Semantic matching with schema
            query_embedding = self.model.encode(query_lower, show_progress_bar=False)
            for schema in self.schema_dict['tables']:
                for table in self.schema_dict['tables'][schema]:
                    table_full = f"{schema}.{table}"
                    table_embedding = self.model.encode(table.lower(), show_progress_bar=False)
                    similarity = float(util.cos_sim(query_embedding, table_embedding)[0][0])
                    if similarity > self.similarity_threshold:
                        matched_tables.add(table_full)
                        confidence = max(confidence, similarity)
                        self.logger.debug(f"Semantic match: '{table}' (sim={similarity:.2f})")

            # Weight-based matching
            for table_full, cols in self.weights.items():
                for col, weight in cols.items():
                    if col in query_lower or any(token == col for token in tokens):
                        matched_tables.add(table_full)
                        confidence = max(confidence, weight)
                        self.logger.debug(f"Weight match: '{col}' in '{table_full}' (weight={weight})")
                    for synonym in self.name_matches.get(col, []):
                        if any(token == synonym for token in tokens):
                            matched_tables.add(table_full)
                            confidence = max(confidence, weight)
                            self.logger.debug(f"Weight match via synonym: '{synonym}' -> '{col}' in '{table_full}'")

            # Name match lookup
            for col, synonyms in self.name_matches.items():
                for synonym in synonyms:
                    if synonym in query_lower or any(token == synonym for token in tokens):
                        for schema in self.schema_dict['columns']:
                            for table, cols in self.schema_dict['columns'][schema].items():
                                if col in [c.lower() for c in cols]:
                                    table_full = f"{schema}.{table}"
                                    matched_tables.add(table_full)
                                    confidence = max(confidence, 0.8)
                                    self.logger.debug(f"Name match: '{synonym}' -> '{col}' in '{table_full}'")

            # Entity-based matching
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'GPE']:
                    for schema in self.schema_dict['tables']:
                        for table in self.schema_dict['tables'][schema]:
                            if ent.text.lower() in table.lower():
                                table_full = f"{schema}.{table}"
                                matched_tables.add(table_full)
                                confidence = max(confidence, 0.85)
                                self.logger.debug(f"Entity match: '{ent.text}' ({ent.label_}) -> '{table_full}'")

            # Column score-based matching
            for column_key, score in column_scores.items():
                if score > 0:
                    try:
                        schema, table, _ = column_key.split('.')
                        table_full = f"{schema}.{table}"
                        matched_tables.add(table_full)
                        confidence = max(confidence, min(score * 0.5, 0.9))
                        self.logger.debug(f"Column score match: '{column_key}' (score={score}) -> '{table_full}'")
                    except ValueError:
                        self.logger.debug(f"Invalid column key format: '{column_key}'")

            # Custom rule for stock/availability
            if any(token in ['stock', 'availability', 'stocks', 'quantities'] for token in tokens):
                matched_tables.add('production.stocks')
                confidence = max(confidence, 0.85)
                self.logger.debug("Custom rule match: 'stock/availability' -> 'production.stocks'")

            result = list(matched_tables)
            confidence = min(confidence, self.max_confidence)
            self.logger.debug(f"Identified tables: {result}, confidence={confidence:.2f}")
            return result, confidence
        except Exception as e:
            self.logger.error(f"Error identifying tables for query '{query}': {e}")
            return [], 0.0

    def update_weights_from_feedback(self, query: str, tables: List[str]):
        """
        Update weights based on user feedback for confirmed tables.
        
        Args:
            query: Natural language query.
            tables: List of confirmed tables.
        """
        try:
            for table in tables:
                if table not in self.weights:
                    self.weights[table] = {}
                for col in self.schema_dict['columns'].get(table.split('.')[0], {}).get(table.split('.')[1], []):
                    col_lower = col.lower()
                    self.weights[table][col_lower] = min(self.weights[table].get(col_lower, 0.05) + 0.01, 0.5)
                    self.logger.debug(f"Updated weight for '{col_lower}' in '{table}' to {self.weights[table][col_lower]}")
            self.cache_synchronizer.write_weights(self.weights)
            self.logger.debug("Weights updated from feedback")
        except Exception as e:
            self.logger.error(f"Error updating weights: {e}")

    def update_name_matches(self, column: str, synonyms: List[str]):
        """
        Update name matches with new synonyms for a column.
        
        Args:
            column: Column name to add synonyms for.
            synonyms: List of synonym strings.
        """
        try:
            column_lower = column.lower()
            if column_lower not in self.name_matches:
                self.name_matches[column_lower] = []
            self.name_matches[column_lower].extend(syn for syn in synonyms if syn not in self.name_matches[column_lower])
            self.name_matches[column_lower] = list(set(self.name_matches[column_lower]))
            self.cache_synchronizer.write_name_matches(self.name_matches)
            self.logger.debug(f"Updated name matches for '{column_lower}': {self.name_matches[column_lower]}")
        except Exception as e:
            self.logger.error(f"Error updating name matches for '{column}': {e}")

    def save_name_matches(self):
        """Save name matches to cache."""
        try:
            self.cache_synchronizer.write_name_matches(self.name_matches)
            self.logger.debug("Saved name matches to cache")
        except Exception as e:
            self.logger.error(f"Error saving name matches: {e}")

    def validate_tables(self, tables: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate if tables exist in the schema.
        
        Args:
            tables: List of table names (schema.table format).
        
        Returns:
            Tuple of (valid tables, invalid tables).
        """
        try:
            valid = []
            invalid = []
            for table in tables:
                schema, table_name = table.split('.') if '.' in table else (None, table)
                if schema and table_name and schema in self.schema_dict['tables']:
                    if table_name in self.schema_dict['tables'][schema]:
                        valid.append(table)
                    else:
                        invalid.append(table)
                else:
                    invalid.append(table)
            self.logger.debug(f"Validated tables: valid={valid}, invalid={invalid}")
            return valid, invalid
        except Exception as e:
            self.logger.error(f"Error validating tables: {e}")
            return [], tables

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query by normalizing and cleaning text.
        
        Args:
            query: Raw query string.
        
        Returns:
            Cleaned query string.
        """
        try:
            query_clean = re.sub(r'\s+', ' ', query.strip().lower())
            doc = self.nlp(query_clean)
            tokens = [token.lemma_ for token in doc if not token.is_punct]
            processed = ' '.join(tokens)
            self.logger.debug(f"Preprocessed query: '{query}' -> '{processed}'")
            return processed
        except Exception as e:
            self.logger.error(f"Error preprocessing query '{query}': {e}")
            return query.lower()

    def get_column_tables(self, column: str) -> List[str]:
        """
        Find tables containing a specific column.
        
        Args:
            column: Column name to search for.
        
        Returns:
            List of tables containing the column.
        """
        try:
            column_lower = column.lower()
            tables = []
            for schema in self.schema_dict['columns']:
                for table, cols in self.schema_dict['columns'][schema].items():
                    if column_lower in [c.lower() for c in cols]:
                        tables.append(f"{schema}.{table}")
            self.logger.debug(f"Found tables for column '{column_lower}': {tables}")
            return tables
        except Exception as e:
            self.logger.error(f"Error finding tables for column '{column}': {e}")
            return []