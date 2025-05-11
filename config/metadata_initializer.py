# config/metadata_initializer.py: Initializes metadata caches for TableIdentifier-v2.1
# Enhanced with single-table processing and weight existence checks

import os
import json
import logging
import logging.config
from typing import Dict, List
from datetime import datetime
import numpy as np
import spacy
from sentence_transformers import util
from config.model_singleton import ModelSingleton
from schema.manager import SchemaManager
from config.manager import DatabaseConnection
from config.cache_synchronizer import CacheSynchronizer
import signal
import sys
import time

class MetadataInitializer:
    """Initializes metadata caches for TableIdentifier-v2.1 first launch."""
    
    def __init__(self, db_name: str, schema_manager: SchemaManager, connection_manager: DatabaseConnection):
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
        
        self.logger = logging.getLogger("metadata_initializer")
        self.db_name = db_name
        self.schema_manager = schema_manager
        self.connection_manager = connection_manager
        self.model = ModelSingleton().model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to load spacy model: {e}")
            raise RuntimeError(f"Spacy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
        self.cache_synchronizer = CacheSynchronizer(db_name)
        self.schema_cache_dir = os.path.join("schema_cache", db_name)
        self.feedback_cache_dir = os.path.join("feedback_cache", db_name)
        self.config_dir = os.path.join("app-config", db_name)
        self.query_synonyms_file = os.path.join(self.config_dir, "query_synonyms.json")
        self.default_name_matches_file = os.path.join(self.config_dir, "default_name_matches.json")
        self.synonym_threshold = 0.7
        self.system_schemas = ['dbo', 'sys', 'information_schema']
        self.logger.debug(f"Initialized MetadataInitializer for {db_name}")
        
        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Deduplicate feedback to reduce cache size
        self.logger.info("Deduplicating feedback cache")
        removed = self.cache_synchronizer.deduplicate_feedback()
        self.logger.debug(f"Removed {removed} duplicate feedback entries")
        
        # Log feedback count
        feedback_count = self.cache_synchronizer.count_feedback()
        self.logger.debug(f"Feedback table contains {feedback_count} entries")

    def _signal_handler(self, sig, frame):
        """Handle SIGINT (Ctrl+C) to gracefully exit."""
        self.logger.info("Received SIGINT, closing connections and exiting")
        self.cache_synchronizer.close()
        self.connection_manager.close()
        sys.exit(0)

    def initialize(self) -> bool:
        try:
            self.logger.info("Checking metadata caches")
            
            if not self.connection_manager.is_connected():
                self.logger.error("No active database connection")
                return False
            
            self.logger.info("Building schema cache")
            schema_dict = self._build_schema_cache()
            if not schema_dict:
                self.logger.error("Failed to build schema dictionary")
                return False
            
            self.logger.info("Building weights cache")
            weights = self._build_weights(schema_dict)
            if not weights:
                self.logger.error("Failed to build weights")
                return False
            self.cache_synchronizer.write_weights(weights)
            self.logger.debug("Saved weights to SQLite")
            
            self.logger.info("Building default name matches")
            default_matches = self._build_default_name_matches(schema_dict)
            if not default_matches:
                self.logger.error("Failed to build default name matches")
                return False
            self.cache_synchronizer.write_name_matches(default_matches, 'default')
            self.logger.debug("Saved default name matches to SQLite")
            
            self.logger.info("Generating dynamic query synonyms")
            query_synonyms = self._generate_query_synonyms(schema_dict)
            if not query_synonyms:
                self.logger.error("Failed to generate query synonyms")
                return False
            os.makedirs(self.config_dir, exist_ok=True)
            try:
                with open(self.query_synonyms_file, 'w') as f:
                    json.dump(query_synonyms, f, indent=2)
                self.logger.debug(f"Saved query synonyms to {self.query_synonyms_file}")
            except Exception as e:
                self.logger.error(f"Error saving query synonyms: {e}")
                return False
            
            self.logger.info("Building synthetic feedback")
            if not self._build_synthetic_feedback(schema_dict):
                self.logger.error("Failed to build synthetic feedback")
                return False
            
            self.logger.info("Initializing ignored queries")
            if not self._initialize_ignored_queries():
                self.logger.error("Failed to initialize ignored queries")
                return False
            
            self.logger.info("Migrating file caches")
            if not self._migrate_file_caches():
                self.logger.error("Failed to migrate file caches")
                return False
            
            self.logger.info("Metadata initialization successful")
            return True
        except Exception as e:
            self.logger.error(f"Metadata initialization failed: {e}")
            return False

    def _build_schema_cache(self) -> Dict:
        try:
            schema_dict = self.schema_manager.build_schema_dictionary(self.connection_manager.connection)
            if not schema_dict or 'tables' not in schema_dict:
                self.logger.error("Empty or invalid schema dictionary")
                return {}
            os.makedirs(self.schema_cache_dir, exist_ok=True)
            self.schema_manager.save_schema(os.path.join(self.schema_cache_dir, "schema.json"), schema_dict)
            self.logger.debug(f"Saved schema to {os.path.join(self.schema_cache_dir, 'schema.json')}")
            return schema_dict
        except Exception as e:
            self.logger.error(f"Error building schema cache: {e}")
            return {}

    def _build_weights(self, schema_dict: Dict, batch_size: int = 1) -> Dict[str, Dict[str, float]]:
        try:
            weights = {}
            existing_weights = self.cache_synchronizer.load_weights()
            all_tables = [(schema, table) for schema in schema_dict['tables'] for table in schema_dict['tables'][schema]]
            total_tables = len(all_tables)
            self.logger.debug(f"Processing {total_tables} tables for weights")
            
            for i, (schema, table) in enumerate(all_tables):
                if schema.lower() in self.system_schemas:
                    self.logger.debug(f"Skipping system schema for weights: {schema}")
                    continue
                table_full = f"{schema}.{table}"
                if table_full in existing_weights:
                    self.logger.debug(f"Skipping existing weights for {table_full}")
                    weights[table_full] = existing_weights[table_full]
                    continue
                
                self.logger.debug(f"Processing weights for table {i+1}/{total_tables}: {table_full}")
                weights[table_full] = {}
                for col in schema_dict['columns'][schema][table]:
                    weights[table_full][col.lower()] = 0.05
                if schema in schema_dict['primary_keys'] and table in schema_dict['primary_keys'][schema]:
                    for pk in schema_dict['primary_keys'][schema][table]:
                        weights[table_full][pk.lower()] = 0.1
                if schema in schema_dict['foreign_keys'] and table in schema_dict['foreign_keys'][schema]:
                    for fk in schema_dict['foreign_keys'][schema][table]:
                        weights[table_full][fk['column'].lower()] = 0.08
                
                self.cache_synchronizer.write_weights({table_full: weights[table_full]}, batch_size=10)
                self.logger.debug(f"Saved weights for {table_full}")
                time.sleep(0.01)
            
            return weights
        except Exception as e:
            self.logger.error(f"Error building weights: {e}")
            return {}

    def _build_default_name_matches(self, schema_dict: Dict) -> Dict[str, List[str]]:
        try:
            matches = {}
            for schema in schema_dict['tables']:
                if schema.lower() in self.system_schemas:
                    self.logger.debug(f"Skipping system schema for name matches: {schema}")
                    continue
                for table in schema_dict['tables'][schema]:
                    for col in schema_dict['columns'][schema][table]:
                        col_lower = col.lower()
                        col_embedding = self.model.encode(col_lower, show_progress_bar=False).reshape(1, -1)
                        synonyms = []
                        variations = [
                            col_lower,
                            col_lower.replace('_', ' '),
                            col_lower.replace('_', ''),
                            ' '.join(token.lemma_ for token in self.nlp(col_lower)),
                            col_lower.replace('id', 'identifier'),
                            col_lower.replace('date', 'time')
                        ]
                        for var in variations:
                            if not var or var == col_lower:
                                continue
                            var_embedding = self.model.encode(var, show_progress_bar=False).reshape(1, -1)
                            similarity = util.cos_sim(col_embedding, var_embedding)[0][0]
                            if similarity > self.synonym_threshold:
                                synonyms.append(var)
                                self.logger.debug(f"Added synonym '{var}' for '{col_lower}' (sim={similarity:.2f})")
                        table_embedding = self.model.encode(table.lower(), show_progress_bar=False).reshape(1, -1)
                        table_similarity = util.cos_sim(col_embedding, table_embedding)[0][0]
                        if table_similarity > self.synonym_threshold - 0.1:
                            synonyms.append(table.lower())
                            self.logger.debug(f"Added table synonym '{table.lower()}' for '{col_lower}' (sim={table_similarity:.2f})")
                        if col_lower.endswith('_id'):
                            synonyms.append(col_lower[:-3])
                            synonyms.append(table.lower() + '_name')
                            self.logger.debug(f"Added ID synonyms for '{col_lower}'")
                        if synonyms:
                            matches[col_lower] = list(set(synonyms))
            return matches
        except Exception as e:
            self.logger.error(f"Error building default name matches: {e}")
            return {}

    def _generate_query_synonyms(self, schema_dict: Dict) -> Dict:
        try:
            queries = []
            synonyms = {}
            for schema in schema_dict['tables']:
                if schema.lower() in self.system_schemas:
                    self.logger.debug(f"Skipping system schema for queries: {schema}")
                    continue
                for table in schema_dict['tables'][schema]:
                    table_name = table.lower()
                    queries.extend([
                        f"List all {table_name}",
                        f"Show all {table_name}",
                        f"Get {table_name} details",
                        f"Count {table_name}",
                        f"Find {table_name} records"
                    ])
                    for col in schema_dict['columns'][schema][table]:
                        col_lower = col.lower()
                        queries.extend([
                            f"Show {col_lower} from {table_name}",
                            f"List {table_name} with {col_lower}",
                            f"Find {table_name} by {col_lower}",
                            f"Get {table_name} where {col_lower}",
                            f"Select {col_lower} in {table_name}",
                            f"Count {table_name} by {col_lower}",
                            f"Sum {col_lower} in {table_name}"
                        ])
                        self.logger.debug(f"Generated query: {queries[-1]}")
                        col_embedding = self.model.encode(col_lower, show_progress_bar=False).reshape(1, -1)
                        col_synonyms = []
                        variations = [
                            col_lower.replace('_', ' '),
                            col_lower.replace('_', ''),
                            ' '.join(token.lemma_ for token in self.nlp(col_lower)),
                            col_lower.replace('id', 'identifier'),
                            col_lower.replace('date', 'time'),
                            table_name + '_' + col_lower
                        ]
                        for var in variations:
                            if not var or var == col_lower:
                                continue
                            var_embedding = self.model.encode(var, show_progress_bar=False).reshape(1, -1)
                            similarity = util.cos_sim(col_embedding, var_embedding)[0][0]
                            if similarity > self.synonym_threshold:
                                col_synonyms.append(var)
                                self.logger.debug(f"Added query token synonym '{var}' for '{col_lower}' (sim={similarity:.2f})")
                        table_embedding = self.model.encode(table_name, show_progress_bar=False).reshape(1, -1)
                        table_similarity = util.cos_sim(col_embedding, table_embedding)[0][0]
                        if table_similarity > self.synonym_threshold - 0.1:
                            col_synonyms.append(table_name)
                            self.logger.debug(f"Added table synonym '{table_name}' for '{col_lower}' (sim={table_similarity:.2f})")
                        if col_lower.endswith('_id'):
                            col_synonyms.append(col_lower[:-3])
                            col_synonyms.append(table_name + '_name')
                        if col_synonyms:
                            synonyms[col_lower] = list(set(col_synonyms))
            
            return {
                "example_queries": list(set(queries)),
                "synonyms": synonyms
            }
        except Exception as e:
            self.logger.error(f"Error generating query synonyms: {e}")
            return {"example_queries": [], "synonyms": {}}

    def _build_synthetic_feedback(self, schema_dict: Dict, batch_size: int = 50) -> bool:
        try:
            all_queries = []
            for schema in schema_dict['tables']:
                if schema.lower() in self.system_schemas:
                    self.logger.debug(f"Skipping system schema for feedback: {schema}")
                    continue
                for table in schema_dict['tables'][schema]:
                    table_full = f"{schema}.{table}"
                    table_name = table.lower()
                    queries = [
                        f"List all {table_name}",
                        f"Show all {table_name}",
                        f"Get {table_name} details",
                        f"Count {table_name}",
                        f"Find {table_name} records"
                    ]
                    for col in schema_dict['columns'][schema][table]:
                        col_lower = col.lower()
                        queries.extend([
                            f"{table_name} with {col_lower}",
                            f"{table_name} in {col_lower}",
                            f"Find {table_name} by {col_lower}",
                            f"Count {table_name} by {col_lower}",
                            f"Sum {col_lower} in {table_name}"
                        ])
                    all_queries.extend([(query, [table_full]) for query in queries])
                    related_tables = self._find_related_tables(schema_dict, schema, table)
                    if related_tables:
                        query = f"Join {table_name} with {related_tables[0].split('.')[-1]}"
                        all_queries.append((query, [table_full, related_tables[0]]))
            
            total_queries = len(all_queries)
            self.logger.debug(f"Generating {total_queries} synthetic feedback entries")
            
            for i in range(0, total_queries, batch_size):
                batch = all_queries[i:i + batch_size]
                self.logger.debug(f"Processing feedback batch {i//batch_size + 1} ({len(batch)} entries)")
                
                for query, tables in batch:
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        embedding = self.model.encode(query, show_progress_bar=False)
                        self.cache_synchronizer.write_feedback(timestamp, query, tables, embedding, count=1)
                        self.logger.debug(f"Created feedback: {query} -> {tables}")
                    except Exception as e:
                        self.logger.error(f"Error creating feedback for query '{query}': {e}")
                        continue
                
                time.sleep(0.01)
            
            return True
        except Exception as e:
            self.logger.error(f"Error building synthetic feedback: {e}")
            return False

    def _find_related_tables(self, schema_dict: Dict, schema: str, table: str) -> List[str]:
        try:
            related = []
            foreign_keys = schema_dict['foreign_keys'].get(schema, {}).get(table, [])
            for fk in foreign_keys:
                ref_table = f"{fk['referenced_schema']}.{fk['referenced_table']}"
                if ref_table != f"{schema}.{table}":
                    related.append(ref_table)
            return related
        except Exception as e:
            self.logger.error(f"Error finding related tables for {schema}.{table}: {e}")
            return []

    def _initialize_ignored_queries(self) -> bool:
        try:
            default_ignored = [
                {"query": "chitti emchestunnav", "reason": "non-English"},
                {"query": "what you are doing now?", "reason": "irrelevant"},
                {"query": "hello how are you", "reason": "irrelevant"},
                {"query": "hi there", "reason": "irrelevant"},
                {"query": "test query", "reason": "test"}
            ]
            for item in default_ignored:
                query = item["query"]
                reason = item["reason"]
                try:
                    embedding = self.model.encode(query, show_progress_bar=False)
                    self.cache_synchronizer.write_ignored_query(query, embedding, reason)
                    self.logger.debug(f"Initialized ignored query: {query}, reason: {reason}")
                except Exception as e:
                    self.logger.error(f"Error initializing ignored query '{query}': {e}")
                    continue
            return True
        except Exception as e:
            self.logger.error(f"Error initializing ignored queries: {e}")
            return False

    def _migrate_file_caches(self) -> bool:
        try:
            if os.path.exists(self.query_synonyms_file):
                try:
                    with open(self.query_synonyms_file) as f:
                        query_synonyms = json.load(f)
                    schema_dict = self.schema_manager.build_schema_dictionary(self.connection_manager.connection)
                    for query in query_synonyms.get("example_queries", []):
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        embedding = self.model.encode(query, show_progress_bar=False)
                        table = None
                        for schema in schema_dict['tables']:
                            for t in schema_dict['tables'][schema]:
                                if t.lower() in query.lower():
                                    table = f"{schema}.{t}"
                                    break
                            if table:
                                break
                        if table:
                            self.cache_synchronizer.write_feedback(timestamp, query, [table], embedding)
                            self.logger.debug(f"Migrated query synonym as feedback: {query} -> {table}")
                    bak_file = self.query_synonyms_file + ".bak"
                    if os.path.exists(bak_file):
                        self.logger.debug(f"Removing existing backup file: {bak_file}")
                        os.remove(bak_file)
                    os.rename(self.query_synonyms_file, bak_file)
                    self.logger.debug(f"Backed up {self.query_synonyms_file} to {bak_file}")
                except Exception as e:
                    self.logger.warning(f"Error migrating query synonyms: {e}")
                    return False
            
            if os.path.exists(self.default_name_matches_file):
                try:
                    with open(self.default_name_matches_file) as f:
                        name_matches = json.load(f)
                    self.cache_synchronizer.write_name_matches(name_matches, 'default')
                    self.logger.debug(f"Migrated default name matches from {self.default_name_matches_file}")
                except Exception as e:
                    self.logger.warning(f"Error migrating default name matches: {e}")
                    return False
            
            self.cache_synchronizer.migrate_file_caches()
            self.logger.debug("File cache migration completed")
            return True
        except Exception as e:
            self.logger.error(f"Error migrating file caches: {e}")
            return False