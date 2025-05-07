# analysis/table_identifier.py: Identifies tables from queries for TIA-1.1
# Prioritized similar feedback in table identification

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
from datetime import datetime, timedelta

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
            schema_dict: Dictionary of schema metadata.
            feedback_manager: Manages user feedback.
            pattern_manager: Manages regex patterns.
            cache_synchronizer: Handles cache operations.
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
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            self.logger.debug("Loaded spacy model: en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to load spacy model: {e}")
            raise RuntimeError(f"Spacy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
        self.weights = self.cache_synchronizer.load_weights() or {}
        self.name_matches = self.cache_synchronizer.read_name_matches('dynamic') or {}
        self.similarity_threshold = 0.7
        self.max_confidence = 0.95
        self.last_decay = datetime.now()
        self.logger.debug("Initialized TableIdentifier")

    def _decay_weights(self):
        """Decay weights for low-confidence matches weekly."""
        if datetime.now() - self.last_decay > timedelta(days=7):
            for table, cols in self.weights.items():
                for col, weight in cols.items():
                    self.weights[table][col] = max(0.05, weight * 0.9)
            self.cache_synchronizer.write_weights(self.weights)
            self.last_decay = datetime.now()
            self.logger.debug("Applied weight decay to low-confidence matches")

    def identify_tables(self, query: str, column_scores: Dict[str, float]) -> Tuple[List[str], float]:
        """
        Identify tables relevant to a query using multiple strategies with pruning.
        
        Args:
            query: Natural language query string.
            column_scores: Dictionary of column paths to scores.
        
        Returns:
            Tuple of (list of table names in schema.table format, confidence score).
        """
        try:
            self.logger.debug(f"Processing query: {query}")
            self._decay_weights()
            query_lower = self.preprocess_query(query)
            doc = self.nlp(query_lower)
            tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
            table_scores: Dict[str, float] = {}
            match_details = []

            # Check similar feedback first
            if self.feedback_manager:
                similar_feedback = self.cache_synchronizer.find_similar_feedback(query)
                if similar_feedback:
                    for fb_query, fb_tables, sim in similar_feedback:
                        for table_full in fb_tables:
                            table_scores[table_full] = table_scores.get(table_full, 0) + sim * 0.9
                            match_details.append(f"Feedback match: '{fb_query}' -> '{table_full}' (sim={sim:.2f}, score=+{sim*0.9:.2f})")
                    self.logger.debug(f"Found {len(similar_feedback)} similar feedback entries for query: {query}")

            # Exact table name matching
            for schema in self.schema_dict['tables']:
                for table in self.schema_dict['tables'][schema]:
                    if table.lower() in query_lower:
                        table_full = f"{schema}.{table}"
                        table_scores[table_full] = table_scores.get(table_full, 0) + 0.9
                        match_details.append(f"Exact match: '{table}' -> '{table_full}' (score=+0.9)")

            # Pattern-based matching
            patterns = self.pattern_manager.get_patterns()
            for pattern, tables in patterns.items():
                if re.search(pattern, query_lower):
                    for table_full in tables:
                        table_scores[table_full] = table_scores.get(table_full, 0) + 0.7
                        match_details.append(f"Pattern match: '{pattern}' -> '{table_full}' (score=+0.7)")

            # Semantic matching
            query_embedding = self.model.encode(query_lower, show_progress_bar=False)
            for schema in self.schema_dict['tables']:
                for table in self.schema_dict['tables'][schema]:
                    table_full = f"{schema}.{table}"
                    table_embedding = self.model.encode(table.lower(), show_progress_bar=False)
                    similarity = float(util.cos_sim(query_embedding, table_embedding)[0][0])
                    if similarity > self.similarity_threshold:
                        table_scores[table_full] = table_scores.get(table_full, 0) + similarity
                        match_details.append(f"Semantic match: '{table}' (sim={similarity:.2f}) -> '{table_full}' (score=+{similarity:.2f})")

            # Weight-based matching
            for table_full, cols in self.weights.items():
                if table_full not in [f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]]:
                    continue
                for col, weight in cols.items():
                    col_lower = col.lower()
                    if col_lower in query_lower or any(token == col_lower for token in tokens):
                        table_scores[table_full] = table_scores.get(table_full, 0) + weight
                        match_details.append(f"Weight match: '{col_lower}' in '{table_full}' (weight={weight})")
                    for synonym in self.name_matches.get(col_lower, []):
                        if any(token == synonym.lower() for token in tokens):
                            table_scores[table_full] = table_scores.get(table_full, 0) + weight
                            match_details.append(f"Weight match via synonym: '{synonym}' -> '{col_lower}' in '{table_full}' (weight={weight})")

            # Name match lookup
            for col, synonyms in self.name_matches.items():
                col_lower = col.lower()
                for synonym in synonyms:
                    if synonym.lower() in query_lower or any(token == synonym.lower() for token in tokens):
                        for schema in self.schema_dict['columns']:
                            for table, cols in self.schema_dict['columns'][schema].items():
                                if col_lower in [c.lower() for c in cols]:
                                    table_full = f"{schema}.{table}"
                                    table_scores[table_full] = table_scores.get(table_full, 0) + 0.8
                                    match_details.append(f"Name match: '{synonym}' -> '{col_lower}' in '{table_full}' (score=+0.8)")

            # Entity-based matching
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'GPE']:
                    for schema in self.schema_dict['tables']:
                        for table in self.schema_dict['tables'][schema]:
                            if ent.text.lower() in table.lower():
                                table_full = f"{schema}.{table}"
                                table_scores[table_full] = table_scores.get(table_full, 0) + 0.85
                                match_details.append(f"Entity match: '{ent.text}' ({ent.label_}) -> '{table_full}' (score=+0.85)")

            # Column score-based matching
            for column_key, score in column_scores.items():
                if score > 0:
                    try:
                        schema, table, col = column_key.split('.')
                        table_full = f"{schema}.{table}"
                        if table_full in [f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]]:
                            table_scores[table_full] = table_scores.get(table_full, 0) + score * 0.5
                            match_details.append(f"Column score match: '{column_key}' (score={score}) -> '{table_full}' (score=+{score*0.5:.2f})")
                    except ValueError:
                        self.logger.debug(f"Invalid column key format: '{column_key}'")

            # Custom rule for stock/availability
            if any(token in ['stock', 'availability', 'stocks', 'quantities'] for token in tokens):
                table_full = 'production.stocks'
                if table_full in [f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]]:
                    table_scores[table_full] = table_scores.get(table_full, 0) + 0.85
                    match_details.append("Custom rule match: 'stock/availability' -> 'production.stocks' (score=+0.85)")

            # Boost scores for foreign key relationships
            for table_full in table_scores:
                schema, table_name = table_full.split('.')
                foreign_keys = self.schema_dict.get('foreign_keys', {}).get(schema, {}).get(table_name, [])
                for fk in foreign_keys:
                    ref_table = fk.get('ref_table', fk.get('referenced_table'))
                    ref_schema = fk.get('ref_schema', schema)
                    ref_table_full = f"{ref_schema}.{ref_table}"
                    if ref_table_full in table_scores:
                        table_scores[table_full] += 0.1
                        table_scores[ref_table_full] += 0.1
                        match_details.append(f"Foreign key boost: '{table_full}' <-> '{ref_table_full}' (score=+0.1)")

            # Prune to top 3-4 tables
            ranked_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
            selected_tables = [table for table, score in ranked_tables if score > 0.3][:4]
            confidence = min(sum(score for _, score in ranked_tables[:4]) / 4, self.max_confidence) if ranked_tables else 0.0

            # Fallback: Check feedback history
            if not selected_tables and self.feedback_manager:
                feedback = self.feedback_manager.get_all_feedback()
                for entry in feedback:
                    if query_lower == entry['query'].lower():
                        selected_tables = entry['tables']
                        confidence = max(confidence, 0.9)
                        match_details.append(f"Feedback match: '{query}' -> {selected_tables} (score=+0.9)")
                        break

            # Log scoring details
            for detail in match_details:
                self.logger.debug(detail)
            if selected_tables:
                self.logger.debug(f"Final tables: {selected_tables}, confidence={confidence:.2f}")
            else:
                self.logger.debug(f"No tables identified for query: {query}")
            return selected_tables, confidence
        except Exception as e:
            self.logger.error(f"Error identifying tables for query '{query}': {e}")
            return [], 0.0

    def update_weights_from_feedback(self, query: str, tables: List[str]):
        """
        Update weights based on user feedback and remove from ignored queries.
        
        Args:
            query: Natural language query.
            tables: List of confirmed tables (schema.table format).
        """
        try:
            valid_tables = set(tables)
            all_tables = {f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]}
            for table_full in all_tables:
                schema, table_name = table_full.split('.')
                weight_change = 0.01 if table_full in valid_tables else -0.01
                if table_full not in self.weights:
                    self.weights[table_full] = {}
                for col in self.schema_dict['columns'].get(schema, {}).get(table_name, []):
                    col_lower = col.lower()
                    self.weights[table_full][col_lower] = min(max(0.05, self.weights[table_full].get(col_lower, 0.05) + weight_change), 0.5)
                    self.logger.debug(f"{'Increased' if weight_change > 0 else 'Decreased'} weight for '{col_lower}' in '{table_full}' to {self.weights[table_full][col_lower]:.2f}")
            self.cache_synchronizer.write_weights(self.weights)
            self.cache_synchronizer.delete_ignored_query(query)
            self.logger.debug(f"Weights updated and query '{query}' removed from ignored_queries")
        except Exception as e:
            self.logger.error(f"Error updating weights for query '{query}': {e}")

    def update_name_matches(self, column: str, synonyms: List[str]):
        """
        Update name matches with new synonyms for a column.
        
        Args:
            column: Column name.
            synonyms: List of synonym strings.
        """
        try:
            column_lower = column.lower()
            if column_lower not in self.name_matches:
                self.name_matches[column_lower] = []
            self.name_matches[column_lower].extend(syn.lower() for syn in synonyms if syn.lower() not in self.name_matches[column_lower])
            self.name_matches[column_lower] = list(set(self.name_matches[column_lower]))
            self.cache_synchronizer.write_name_matches(self.name_matches, 'dynamic')
            self.logger.debug(f"Updated name matches for '{column_lower}': {self.name_matches[column_lower]}")
        except Exception as e:
            self.logger.error(f"Error updating name matches for '{column}': {e}")

    def save_name_matches(self):
        """
        Save name matches to cache.
        """
        try:
            self.cache_synchronizer.write_name_matches(self.name_matches, 'dynamic')
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
        Preprocess query by normalizing and lemmatizing text.
        
        Args:
            query: Raw query string.
        
        Returns:
            Cleaned and lemmatized query string.
        """
        try:
            query_clean = re.sub(r'\s+', ' ', query.strip().lower())
            self.logger.debug(f"Normalized query: {query_clean}")
            
            doc = self.nlp(query_clean)
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            self.logger.debug(f"Lemmatized tokens: {tokens}")
            
            processed_query = ' '.join(tokens)
            self.logger.debug(f"Preprocessed query: {query} -> {processed_query}")
            
            return processed_query if processed_query else query_clean
        except Exception as e:
            self.logger.error(f"Error preprocessing query '{query}': {e}")
            return query.lower()