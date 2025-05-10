# analysis/table_identifier.py: Identifies tables from queries for TableIdentifier-v2.1
# Enhanced feedback deduplication, query generalization, and custom rules

import logging
import logging.config
import os
import re
from typing import Dict, List, Tuple, Optional
import spacy
from sentence_transformers import util
from config.model_singleton import ModelSingleton
from config.patterns import PatternManager
from feedback.manager import FeedbackManager
from config.cache_synchronizer import CacheSynchronizer
from datetime import datetime, timedelta

class TableIdentifier:
    """Identifies relevant tables from queries using patterns, similarity, weights, and feedback."""
    
    def __init__(self, schema_dict: Dict, feedback_manager: FeedbackManager, 
                 pattern_manager: PatternManager, cache_synchronizer: CacheSynchronizer):
        os.makedirs("logs", exist_ok=True)
        logging_config_path = "app-config/logging_config.ini"
        try:
            if os.path.exists(logging_config_path):
                logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            else:
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                    handlers=[logging.FileHandler(os.path.join("logs", "bikestores_app.log")), logging.StreamHandler()])
                logging.warning(f"Logging config file not found: {logging_config_path}")
        except Exception as e:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                handlers=[logging.FileHandler(os.path.join("logs", "bikestores_app.log")), logging.StreamHandler()])
            logging.error(f"Error loading logging config: {e}")
        
        self.logger = logging.getLogger("table_identifier")
        self.schema_dict = schema_dict
        self.feedback_manager = feedback_manager
        self.pattern_manager = pattern_manager
        self.cache_synchronizer = cache_synchronizer
        self.model = ModelSingleton().model
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            self.logger.debug("Loaded spacy model")
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
        """Decay weights weekly for low-confidence matches."""
        if datetime.now() - self.last_decay > timedelta(days=7):
            for table, cols in self.weights.items():
                for col, weight in cols.items():
                    self.weights[table][col] = max(0.05, weight * 0.9)
            self.cache_synchronizer.write_weights(self.weights)
            self.last_decay = datetime.now()
            self.logger.debug("Applied weight decay")

    def check_duplicate_feedback(self, query: str, tables: List[str]) -> Tuple[bool, Optional[str], Optional[List[str]]]:
        """Check for duplicate feedback; return (is_duplicate, feedback_id, existing_tables)."""
        try:
            normalized_query = self.cache_synchronizer.normalize_query(query)
            similar_feedback = self.cache_synchronizer.find_similar_feedback(normalized_query, threshold=0.95)
            for fb_query, fb_tables, sim in similar_feedback:
                if sim > 0.95:
                    fb_id = next(ts for ts, data in self.cache_synchronizer.read_feedback().items() if data['query'] == fb_query)
                    self.logger.debug(f"Found duplicate: '{normalized_query}' ~ '{fb_query}' (sim={sim:.2f})")
                    return True, fb_id, fb_tables
            return False, None, None
        except Exception as e:
            self.logger.error(f"Error checking duplicate feedback: {e}")
            return False, None, None

    def identify_tables(self, query: str, column_scores: Dict[str, float]) -> Tuple[List[str], float]:
        """Identify tables using feedback, patterns, similarity, weights, and custom rules."""
        try:
            self.logger.debug(f"Processing query: {query}")
            self._decay_weights()
            query_lower = self.preprocess_query(query)
            doc = self.nlp(query_lower)
            tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
            table_scores: Dict[str, float] = {}
            match_details = []

            # Short-circuit for high-similarity feedback
            if self.feedback_manager:
                similar_feedback = self.cache_synchronizer.find_similar_feedback(query, threshold=0.7)
                for fb_query, fb_tables, sim in similar_feedback:
                    if sim > 0.95:
                        self.logger.debug(f"Using feedback: '{fb_query}' -> {fb_tables} (sim={sim:.2f})")
                        return fb_tables, 0.95
                    if sim > 0.7:
                        for table_full in fb_tables:
                            table_scores[table_full] = table_scores.get(table_full, 0) + sim * 4.0
                            match_details.append(f"Feedback: '{fb_query}' -> '{table_full}' (sim={sim:.2f}, score=+{sim*4.0:.2f})")
                if similar_feedback:
                    self.logger.debug(f"Found {len(similar_feedback)} similar feedback entries")

            # Exact table name matching
            for schema in self.schema_dict['tables']:
                for table in self.schema_dict['tables'][schema]:
                    if table.lower() in query_lower:
                        table_full = f"{schema}.{table}"
                        table_scores[table_full] = table_scores.get(table_full, 0) + 0.9
                        match_details.append(f"Exact match: '{table}' -> '{table_full}' (+0.9)")

            # Pattern-based matching
            patterns = self.pattern_manager.get_patterns()
            for pattern, tables in patterns.items():
                if re.search(pattern, query_lower):
                    for table_full in tables:
                        table_scores[table_full] = table_scores.get(table_full, 0) + 0.7
                        match_details.append(f"Pattern: '{pattern}' -> '{table_full}' (+0.7)")

            # Semantic matching
            query_embedding = self.model.encode(query_lower, show_progress_bar=False)
            for schema in self.schema_dict['tables']:
                for table in self.schema_dict['tables'][schema]:
                    table_full = f"{schema}.{table}"
                    table_embedding = self.model.encode(table.lower(), show_progress_bar=False)
                    similarity = float(util.cos_sim(query_embedding, table_embedding)[0][0])
                    if similarity > self.similarity_threshold:
                        table_scores[table_full] = table_scores.get(table_full, 0) + similarity
                        match_details.append(f"Semantic: '{table}' (sim={similarity:.2f}) -> '{table_full}' (+{similarity:.2f})")

            # Weight-based matching
            for table_full, cols in self.weights.items():
                if table_full not in [f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]]:
                    continue
                for col, weight in cols.items():
                    col_lower = col.lower()
                    if col_lower in query_lower or any(token == col_lower for token in tokens):
                        table_scores[table_full] = table_scores.get(table_full, 0) + weight
                        match_details.append(f"Weight: '{col_lower}' in '{table_full}' (weight={weight})")
                    for synonym in self.name_matches.get(col_lower, []):
                        if any(token == synonym.lower() for token in tokens):
                            table_scores[table_full] = table_scores.get(table_full, 0) + weight
                            match_details.append(f"Synonym: '{synonym}' -> '{col_lower}' in '{table_full}' (weight={weight})")

            # Name match lookup
            for col, synonyms in self.name_matches.items():
                col_lower = col.lower()
                for synonym in synonyms:
                    if synonym.lower() in query_lower or any(token == synonym.lower() for token in tokens):
                        for schema in self.schema_dict['columns']:
                            for table, cols in self.schema_dict['columns'][schema].items():
                                if col_lower in [c.lower() for c in cols]:
                                    table_full = f"{schema}.{table}"
                                    table_scores[table_full] = table_scores.get(table_full, 0) + 0.6
                                    match_details.append(f"Name match: '{synonym}' -> '{col_lower}' in '{table_full}' (+0.6)")

            # Column score-based matching
            for column_key, score in column_scores.items():
                if score > 0:
                    try:
                        schema, table, col = column_key.split('.')
                        table_full = f"{schema}.{table}"
                        if table_full in [f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]]:
                            table_scores[table_full] = table_scores.get(table_full, 0) + score * 0.5
                            match_details.append(f"Column: '{column_key}' (score={score}) -> '{table_full}' (+{score*0.5:.2f})")
                    except ValueError:
                        self.logger.debug(f"Invalid column key: '{column_key}'")

            # Custom rule for stock/availability
            if any(token in ['stock', 'availability', 'stocks', 'quantities'] for token in tokens):
                table_full = 'production.stocks'
                if table_full in [f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]]:
                    table_scores[table_full] = table_scores.get(table_full, 0) + 0.85
                    match_details.append("Custom: 'stock/availability' -> 'production.stocks' (+0.85)")
                table_full = 'sales.stores'
                if table_full in [f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]]:
                    table_scores[table_full] = table_scores.get(table_full, 0) + 0.75
                    match_details.append("Custom: 'stock/availability' -> 'sales.stores' (+0.75)")

            # Prune to top tables
            ranked_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
            selected_tables = [table for table, score in ranked_tables if score > 0.7][:4]
            confidence = min(sum(score for _, score in ranked_tables[:4]) / 4, self.max_confidence) if ranked_tables else 0.0

            # Fallback to feedback history
            if not selected_tables and self.feedback_manager:
                feedback = self.feedback_manager.get_all_feedback()
                normalized_query = self.cache_synchronizer.normalize_query(query_lower)
                for entry in feedback:
                    if normalized_query == entry['query'].lower():
                        selected_tables = entry['tables']
                        confidence = max(confidence, 0.9)
                        match_details.append(f"Feedback: '{query}' -> {selected_tables} (+3.0)")
                        break

            for detail in match_details:
                self.logger.debug(detail)
            if selected_tables:
                self.logger.debug(f"Final tables: {selected_tables}, confidence={confidence:.2f}")
            else:
                self.logger.debug(f"No tables identified for query")
            return selected_tables, confidence
        except Exception as e:
            self.logger.error(f"Error identifying tables: {e}")
            return [], 0.0

    def update_weights_from_feedback(self, query: str, tables: List[str]):
        """Update weights based on feedback, skip or merge duplicates."""
        try:
            is_duplicate, _, _ = self.check_duplicate_feedback(query, tables)
            if is_duplicate:
                self.logger.debug(f"Skipped weight update for duplicate feedback")
                return

            valid_tables = set(tables)
            all_tables = {f"{s}.{t}" for s in self.schema_dict['tables'] for t in self.schema_dict['tables'][s]}
            for table_full in all_tables:
                schema, table_name = table_full.split('.')
                weight_change = 0.02 if table_full in valid_tables else -0.15
                if table_full not in self.weights:
                    self.weights[table_full] = {}
                for col in self.schema_dict['columns'].get(schema, {}).get(table_name, []):
                    col_lower = col.lower()
                    self.weights[table_full][col_lower] = min(max(0.05, self.weights[table_full].get(col_lower, 0.05) + weight_change), 0.8)
                    self.logger.debug(f"{'Increased' if weight_change > 0 else 'Decreased'} weight for '{col_lower}' in '{table_full}' to {self.weights[table_full][col_lower]:.2f}")
            self.cache_synchronizer.write_weights(self.weights)
            self.cache_synchronizer.delete_ignored_query(query)
            self.logger.debug(f"Weights updated and query removed from ignored_queries")
        except Exception as e:
            self.logger.error(f"Error updating weights: {e}")

    def update_name_matches(self, column: str, synonyms: List[str]):
        """Update name matches with new synonyms."""
        try:
            column_lower = column.lower()
            if column_lower not in self.name_matches:
                self.name_matches[column_lower] = []
            self.name_matches[column_lower].extend(syn.lower() for syn in synonyms if syn.lower() not in self.name_matches[column_lower])
            self.name_matches[column_lower] = list(set(self.name_matches[column_lower]))
            self.cache_synchronizer.write_name_matches(self.name_matches, 'dynamic')
            self.logger.debug(f"Updated name matches for '{column_lower}': {self.name_matches[column_lower]}")
        except Exception as e:
            self.logger.error(f"Error updating name matches: {e}")

    def save_name_matches(self):
        """Save name matches to cache."""
        try:
            self.cache_synchronizer.write_name_matches(self.name_matches, 'dynamic')
            self.logger.debug("Saved name matches")
        except Exception as e:
            self.logger.error(f"Error saving name matches: {e}")

    def validate_tables(self, tables: List[str]) -> Tuple[List[str], List[str]]:
        """Validate if tables exist in the schema."""
        try:
            valid, invalid = [], []
            for table in tables:
                schema, table_name = table.split('.') if '.' in table else (None, table)
                if schema and table_name and schema in self.schema_dict['tables'] and table_name in self.schema_dict['tables'][schema]:
                    valid.append(table)
                else:
                    invalid.append(table)
                    self.logger.debug(f"Invalid table: '{table}'")
            self.logger.debug(f"Validated tables: valid={valid}, invalid={invalid}")
            return valid, invalid
        except Exception as e:
            self.logger.error(f"Error validating tables: {e}")
            return [], tables

    def preprocess_query(self, query: str) -> str:
        """Preprocess query by normalizing and lemmatizing."""
        try:
            query_clean = re.sub(r'\s+', ' ', query.strip().lower())
            doc = self.nlp(query_clean)
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            processed_query = ' '.join(tokens)
            self.logger.debug(f"Preprocessed query: {query} -> {processed_query}")
            return processed_query if processed_query else query_clean
        except Exception as e:
            self.logger.error(f"Error preprocessing query: {e}")
            return query.lower()