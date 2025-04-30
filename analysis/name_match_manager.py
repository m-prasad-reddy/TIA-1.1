# analysis/name_match_manager.py: Manages name matches and synonyms for TableIdentifier-v1
# Removes read_processed_queries, preserves ~225 lines, fixes indentation

import os
import json
import logging
import logging.config
from typing import Dict, List, Optional
import spacy
import numpy as np
from sentence_transformers import util
from config.model_singleton import ModelSingleton
from config.cache_synchronizer import CacheSynchronizer

class NameMatchManager:
    """Manages name matches and synonyms for query processing in TableIdentifier-v1."""
    
    def __init__(self, db_name: str):
        """Initialize with database name."""
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
        
        self.logger = logging.getLogger("name_match_manager")
        self.db_name = db_name
        self.nlp = spacy.load("en_core_web_sm")
        self.model = ModelSingleton().model
        self.cache_synchronizer = CacheSynchronizer(db_name)
        self.default_matches = self.cache_synchronizer.read_name_matches('default')
        self.dynamic_matches = self.cache_synchronizer.read_name_matches('dynamic')
        self.synonym_threshold = 0.7  # Similarity threshold for synonym detection
        self.logger.debug(f"Initialized NameMatchManager for {db_name}")

    def process_query(self, query: str, schema_dict: Dict) -> Dict[str, float]:
        """Process query to identify column matches and return scores."""
        try:
            if not schema_dict or 'tables' not in schema_dict or not schema_dict['tables']:
                self.logger.error("Invalid or empty schema dictionary provided")
                return {}

            # Tokenize and extract relevant tokens
            doc = self.nlp(query.lower())
            tokens = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'VERB', 'ADJ') and not token.is_stop]
            self.logger.debug(f"Extracted tokens: {tokens}")
            if not tokens:
                self.logger.debug("No relevant tokens found in query")
                return {}
            
            # Generate embeddings for tokens
            token_embeddings = self.model.encode(tokens, show_progress_bar=False) if tokens else []
            self.logger.debug(f"Generated embeddings for {len(tokens)} tokens")
            
            # Initialize synonym dictionary
            synonyms = {}
            unmatched_tokens = tokens.copy()
            
            # Iterate through schema to find matches
            system_schemas = ['dbo', 'sys', 'information_schema']
            for schema in schema_dict['tables']:
                if schema.lower() in system_schemas:
                    self.logger.debug(f"Skipping system schema: {schema}")
                    continue
                for table in schema_dict['tables'][schema]:
                    for col in schema_dict['columns'][schema][table]:
                        col_lower = col.lower()
                        col_embedding = self.model.encode(col_lower, show_progress_bar=False).reshape(1, -1)
                        
                        # Check existing matches (default and dynamic)
                        if col_lower in self.default_matches:
                            for syn in self.default_matches[col_lower]:
                                if syn in tokens and syn in unmatched_tokens:
                                    unmatched_tokens.remove(syn)
                                    synonyms.setdefault(col_lower, []).append(syn)
                                    self.logger.debug(f"Matched default synonym '{syn}' for '{col_lower}'")
                        if col_lower in self.dynamic_matches:
                            for syn in self.dynamic_matches[col_lower]:
                                if syn in tokens and syn in unmatched_tokens:
                                    unmatched_tokens.remove(syn)
                                    synonyms.setdefault(col_lower, []).append(syn)
                                    self.logger.debug(f"Matched dynamic synonym '{syn}' for '{col_lower}'")
                        
                        # Check token similarity for new synonyms
                        for token, token_emb in zip(tokens, token_embeddings):
                            if token not in unmatched_tokens:
                                continue
                            token_emb = token_emb.reshape(1, -1)
                            similarity = util.cos_sim(token_emb, col_embedding)[0][0]
                            if similarity > self.synonym_threshold:
                                # Prompt user for confirmation
                                print(f"\nIs '{token}' a synonym for column '{col}' in table '{schema}.{table}'? (Y/N)")
                                response = input().strip().lower()
                                if response == 'y':
                                    if col_lower in self.dynamic_matches and token in self.dynamic_matches[col_lower]:
                                        self.logger.debug(f"Synonym '{token}' already exists for '{col_lower}'")
                                        continue
                                    synonyms.setdefault(col_lower, []).append(token)
                                    unmatched_tokens.remove(token)
                                    self.dynamic_matches.setdefault(col_lower, []).append(token)
                                    self.logger.info(f"User confirmed synonym '{token}' for '{col_lower}'")
                                    self._save_dynamic_matches()
                                elif response == 'n':
                                    # Check for conflicts with other columns
                                    conflict_col = self._find_conflict_column(
                                        schema_dict, schema, table, col_lower, token, token_emb
                                    )
                                    if conflict_col:
                                        self.logger.debug(f"Synonym conflict for '{token}' with '{conflict_col}'")
                                        synonyms.setdefault(conflict_col.lower(), []).append(token)
                                        unmatched_tokens.remove(token)
                                        self.dynamic_matches.setdefault(conflict_col.lower(), []).append(token)
                                        self._save_dynamic_matches()
            
            # Log results
            for col, syn_list in synonyms.items():
                self.logger.debug(f"Synonyms for '{col}': {syn_list}")
            if unmatched_tokens:
                self.logger.debug(f"Unmatched tokens: {unmatched_tokens}")
            
            # Score columns based on synonyms
            scores = self._score_columns(schema_dict, synonyms)
            return scores
        except Exception as e:
            self.logger.error(f"Error processing query '{query}': {e}")
            return {}

    def _find_conflict_column(self, schema_dict: Dict, schema: str, table: str, current_col: str, token: str, token_emb: np.ndarray) -> Optional[str]:
        """Find a conflicting column with higher similarity to the token."""
        try:
            for other_col in schema_dict['columns'][schema][table]:
                if other_col.lower() != current_col:
                    other_embedding = self.model.encode(other_col.lower(), show_progress_bar=False).reshape(1, -1)
                    other_similarity = util.cos_sim(token_emb, other_embedding)[0][0]
                    if other_similarity > self.synonym_threshold:
                        return other_col
            return None
        except Exception as e:
            self.logger.error(f"Error finding conflict column for token '{token}': {e}")
            return None

    def _score_columns(self, schema_dict: Dict, synonyms: Dict[str, List[str]]) -> Dict[str, float]:
        """Score columns based on matched synonyms."""
        try:
            scores = {}
            system_schemas = ['dbo', 'sys', 'information_schema']
            for schema in schema_dict['tables']:
                if schema.lower() in system_schemas:
                    continue
                for table in schema_dict['tables'][schema]:
                    for col in schema_dict['columns'][schema][table]:
                        col_lower = col.lower()
                        score = 0.0
                        if col_lower in synonyms:
                            score += 0.1 * len(synonyms[col_lower])  # Weight by number of synonyms
                        scores[f"{schema}.{table}.{col_lower}"] = score
                        self.logger.debug(f"Column score for '{col_lower}': {score}")
            return scores
        except Exception as e:
            self.logger.error(f"Error scoring columns: {e}")
            return {}

    def _save_dynamic_matches(self):
        """Save dynamic matches to SQLite."""
        try:
            # Remove duplicates while preserving order
            for col in self.dynamic_matches:
                self.dynamic_matches[col] = list(dict.fromkeys(self.dynamic_matches[col]))
            self.cache_synchronizer.write_name_matches(self.dynamic_matches, 'dynamic')
            self.logger.debug("Saved dynamic matches to SQLite")
        except Exception as e:
            self.logger.error(f"Error saving dynamic matches: {e}")

    def save_matches(self):
        """Save all matches to SQLite."""
        try:
            # Save default matches
            for col in self.default_matches:
                self.default_matches[col] = list(dict.fromkeys(self.default_matches[col]))
            self.cache_synchronizer.write_name_matches(self.default_matches, 'default')
            
            # Save dynamic matches
            self._save_dynamic_matches()
            
            self.logger.debug("Saved all name matches to SQLite")
        except Exception as e:
            self.logger.error(f"Error saving name matches: {e}")