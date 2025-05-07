# cli/interface.py: CLI for TIA-1.1
# Relaxed query validation and added similar feedback table display

import os
import json
import logging
import logging.config
from typing import Dict, List, Optional, Tuple
import langdetect
from datetime import datetime
import re
from config.cache_synchronizer import CacheSynchronizer
from feedback.manager import FeedbackManager
from analysis.name_match_manager import NameMatchManager
from analysis.table_identifier import TableIdentifier
from config.patterns import PatternManager
from config.manager import DatabaseConnection
from schema.manager import SchemaManager
from config.model_singleton import ModelSingleton
import spacy
import numpy as np

class DatabaseAnalyzerCLI:
    """CLI for TIA-1.1, providing database interaction and query processing."""
    
    def __init__(self, db_name: str, schema_dict: Dict = None, feedback_manager: Optional[FeedbackManager] = None,
                 schemas: List[str] = None, tables: List[str] = None):
        """Initialize with database name, schema dictionary, feedback manager, and schemas/tables."""
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
        
        self.logger = logging.getLogger("interface")
        self.db_name = db_name
        self.schema_dict = schema_dict or {}
        self.feedback_manager = feedback_manager
        self.cache_synchronizer = CacheSynchronizer(db_name)
        self.schemas = schemas or []
        self.tables = tables or []
        self.pattern_manager = PatternManager(self.schema_dict, self.schemas, self.tables)
        self.name_match_manager = NameMatchManager(db_name)
        self.table_identifier = TableIdentifier(
            self.schema_dict,
            self.feedback_manager,
            self.pattern_manager,
            self.cache_synchronizer
        )
        self.query_synonyms_file = os.path.join("app-config", db_name, "query_synonyms.json")
        self.config_path = os.path.join("app-config", "database_configurations.json")
        self.query_history = []
        self.max_history = 50
        self.connection_manager = None
        self.schema_manager = None
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            self.logger.debug("Loaded spacy model: en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to load spacy model: {e}")
            raise RuntimeError(f"Spacy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
        self.model = ModelSingleton().model
        self.logger.debug(f"Initialized DatabaseAnalyzerCLI for {db_name}")

    def run(self):
        """Run the CLI with main menu."""
        self.logger.info(f"Started CLI for {self.db_name}")
        while True:
            print("\n=== BikeStores Schema Analyzer ===\n")
            print("Main Menu:")
            print("1. Connect to Database")
            print("2. Query Mode")
            print("3. Reload Configurations")
            print("4. Manage Feedback")
            print("5. Manage Ignored Queries")
            print("6. View Schema")
            print("7. View Query History")
            print("8. Generate Synthetic Feedback")
            print("9. Exit")
            choice = input("Select option: ").strip()
            self.logger.debug(f"User selected option: {choice}")
            
            if choice == '1':
                self._connect_to_database()
            elif choice == '2':
                self._query_mode()
            elif choice == '3':
                self._reload_configurations()
            elif choice == '4':
                self._manage_feedback()
            elif choice == '5':
                self._manage_ignored_queries()
            elif choice == '6':
                self._view_schema()
            elif choice == '7':
                self._view_query_history()
            elif choice == '8':
                self._generate_synthetic_feedback()
            elif choice == '9':
                print("Exiting...")
                self._cleanup()
                break
            else:
                print("Invalid option. Please try again.")

    def _connect_to_database(self):
        """Connect to a database using configuration."""
        try:
            print(f"\nConfig path [default: {self.config_path}]:")
            config_path = input().strip() or self.config_path
            self.logger.debug(f"Using config path: {config_path}")
            
            if not os.path.exists(config_path):
                print(f"Configuration file not found: {config_path}")
                self.logger.error(f"Configuration file not found: {config_path}")
                return
            
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            configs = [
                {**config, 'name': name}
                for name, config in config_dict.items()
                if isinstance(config, dict)
            ]
            self.logger.debug(f"Loaded {len(configs)} configurations")
            
            print("\nAvailable Configurations:")
            for i, config in enumerate(configs, 1):
                print(f"{i}. {config.get('name', 'Unknown')}")
            print(f"{len(configs) + 1}. Cancel")
            
            choice = input("Select configuration: ").strip()
            self.logger.debug(f"Configuration choice: {choice}")
            
            if not choice.isdigit() or int(choice) < 1 or int(choice) > len(configs) + 1:
                print("Invalid choice.")
                return
            
            if int(choice) == len(configs) + 1:
                print("Connection cancelled.")
                return
            
            config = configs[int(choice) - 1]
            db_name = config.get('name')
            required_keys = {'server', 'database', 'username', 'password', 'driver'}
            if not all(key in config for key in required_keys):
                print("Invalid configuration: missing required fields.")
                self.logger.error(f"Invalid configuration: {db_name}")
                return
            
            self.schemas = config.get('schemas', [])
            self.tables = config.get('tables', [])
            for table in self.tables[:]:
                if '.' not in table:
                    self.logger.warning(f"Invalid table format: {table} in {db_name}")
                    self.tables.remove(table)
            
            self.connection_manager = DatabaseConnection()
            self.connection_manager.connect(config)
            self.schema_manager = SchemaManager(db_name, self.schemas, self.tables)
            self.schema_dict = self.schema_manager.build_schema_dictionary(self.connection_manager.connection)
            
            if not self.schema_dict or 'tables' not in self.schema_dict:
                print("Failed to build schema dictionary.")
                self.logger.error("Failed to build schema dictionary")
                self.connection_manager = None
                self.schema_manager = None
                self.schema_dict = {}
                return
            
            self.feedback_manager = FeedbackManager(db_name, self.cache_synchronizer)
            self.name_match_manager = NameMatchManager(db_name)
            self.pattern_manager = PatternManager(self.schema_dict, self.schemas, self.tables)
            self.table_identifier = TableIdentifier(
                self.schema_dict,
                self.feedback_manager,
                self.pattern_manager,
                self.cache_synchronizer
            )
            
            print(f"Connected to {db_name}")
            self.logger.info(f"Selected configuration: {db_name}")
            
            if not os.path.exists(self.query_synonyms_file):
                print(f"No query synonyms file for {db_name}. Generating {self.query_synonyms_file} dynamically.")
                self.logger.info(f"No query synonyms file for {db_name}. Will generate {self.query_synonyms_file}")
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            print(f"Connection failed: {e}")
            self.connection_manager = None
            self.schema_manager = None
            self.schema_dict = {}

    def _query_mode(self):
        """Handle query mode with advanced processing and synonym caching."""
        try:
            example_queries = []
            if self.feedback_manager:
                try:
                    example_queries = self.feedback_manager.get_top_queries(limit=5)
                    if example_queries:
                        print("\nExample Queries:")
                        for i, q in enumerate(example_queries, 1):
                            print(f"{i}. {q['query']} -> {', '.join(q['tables'])}")
                    else:
                        print("\nNo example queries available.")
                except Exception as e:
                    self.logger.error(f"Error loading example queries: {e}")
                    print("Could not load example queries.")
            else:
                self.logger.warning("Feedback manager not initialized, skipping example queries")
                print("Feedback manager not initialized.")
            
            while True:
                query = input("\nEnter query (or 'back'): ").strip()
                self.logger.debug(f"Received query: {query}")
                if query.lower() == 'back':
                    break
                if not query:
                    print("Query cannot be empty.")
                    continue
                
                is_valid, reason = self._validate_query(query)
                if not is_valid:
                    print(f"Query rejected: {reason}")
                    print("Example queries:")
                    print("- Show me customer names")
                    print("- How many orders were delivered in 2016?")
                    print("- What production categories are available?")
                    continue
                
                processed_query = self._expand_query_with_synonyms(query)
                column_scores = self.name_match_manager.process_query(processed_query, self.schema_dict)
                tables, confidence = self.table_identifier.identify_tables(processed_query, column_scores)
                
                # Check similar feedback
                similar_feedback = self.cache_synchronizer.find_similar_feedback(query)
                feedback_tables = []
                if similar_feedback:
                    print("\nSimilar queries found in feedback:")
                    for fb_query, fb_tables, sim in similar_feedback[:3]:
                        print(f"- {fb_query} -> {', '.join(fb_tables)} (similarity: {sim:.2f})")
                        feedback_tables.extend(fb_tables)
                    feedback_tables = list(set(feedback_tables))  # Deduplicate
                
                # Combine tables from identification and feedback
                if not tables and feedback_tables:
                    tables = feedback_tables
                    confidence = max(confidence, 0.9)
                
                if not tables:
                    print("No tables identified for the query.")
                    if similar_feedback:
                        print("Using feedback tables:", ', '.join(feedback_tables))
                    embedding = self.model.encode(query, show_progress_bar=False) if self.model else None
                    self.cache_synchronizer.write_ignored_query(query, embedding, "no_tables_identified")
                    self.query_history.append({
                        'query': query,
                        'tables': [],
                        'confidence': 0.0,
                        'timestamp': datetime.now()
                    })
                    continue
                
                print(f"\nIdentified Tables: {', '.join(tables)}")
                print(f"Confidence: {'High' if confidence > 0.5 else 'Low'}")
                
                feedback_ok = input("Are these tables correct? (y/n): ").strip().lower()
                embedding = self.model.encode(query, show_progress_bar=False) if self.model else None
                if feedback_ok == 'y' and self.feedback_manager:
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        self.feedback_manager.store_feedback(query, tables, self.schema_dict, confidence=confidence)
                        self.table_identifier.update_weights_from_feedback(query, tables)
                        self.table_identifier.save_name_matches()
                        self.cache_synchronizer.write_feedback(timestamp, query, tables, embedding)
                        self.logger.debug(f"Stored feedback for {query} -> {tables}")
                        self.query_history.append({
                            'query': query,
                            'tables': tables,
                            'confidence': confidence,
                            'timestamp': datetime.now()
                        })
                    except Exception as e:
                        self.logger.error(f"Error storing feedback: {e}")
                        print("Failed to store feedback.")
                else:
                    print("Please enter the correct tables (comma-separated, e.g., production.stocks,sales.stores) or 'skip':")
                    correct_tables = input().strip()
                    if correct_tables.lower() != 'skip' and correct_tables:
                        try:
                            correct_tables_list = [t.strip() for t in correct_tables.split(',') if t.strip()]
                            valid_tables, invalid_tables = self.table_identifier.validate_tables(correct_tables_list)
                            if invalid_tables:
                                print(f"Invalid tables: {', '.join(invalid_tables)}")
                                self.logger.debug(f"Invalid tables provided: {invalid_tables}")
                            if valid_tables and self.feedback_manager:
                                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                                self.feedback_manager.store_feedback(query, valid_tables, self.schema_dict, confidence=confidence)
                                self.table_identifier.update_weights_from_feedback(query, valid_tables)
                                self.table_identifier.save_name_matches()
                                self.cache_synchronizer.write_feedback(timestamp, query, valid_tables, embedding)
                                self.logger.debug(f"Stored feedback for correct tables: {valid_tables}")
                                # Cache synonyms from feedback
                                query_tokens = self.name_match_manager.extract_tokens(query)
                                for table in valid_tables:
                                    schema, table_name = table.split('.')
                                    columns = self.schema_dict['columns'].get(schema, {}).get(table_name, [])
                                    for token in query_tokens:
                                        for col in columns:
                                            col_lower = col.lower()
                                            similarity = self.name_match_manager.compute_similarity(token, col_lower)
                                            if similarity > 0.65 and not self.name_match_manager._suggest_synonym(token, col, schema, table_name):
                                                self.name_match_manager.dynamic_matches.setdefault(col_lower, []).append(token)
                                                self.name_match_manager._save_dynamic_matches()
                                                self.logger.debug(f"Cached synonym '{token}' for '{col_lower}' in '{table}' (sim={similarity:.2f})")
                                self.query_history.append({
                                    'query': query,
                                    'tables': valid_tables,
                                    'confidence': confidence,
                                    'timestamp': datetime.now()
                                })
                            else:
                                self.cache_synchronizer.write_ignored_query(query, embedding, "user_rejected_no_valid_tables")
                                self.query_history.append({
                                    'query': query,
                                    'tables': tables,
                                    'confidence': confidence,
                                    'timestamp': datetime.now(),
                                    'rejected': True
                                })
                        except Exception as e:
                            self.logger.error(f"Error processing correct tables: {e}")
                            print(f"Error processing correct tables: {e}")
                            self.cache_synchronizer.write_ignored_query(query, embedding, "user_rejected_error")
                    else:
                        self.cache_synchronizer.write_ignored_query(query, embedding, "user_rejected")
                        self.query_history.append({
                            'query': query,
                            'tables': tables,
                            'confidence': confidence,
                            'timestamp': datetime.now(),
                            'rejected': True
                        })
                
                if len(self.query_history) > self.max_history:
                    self.query_history = self.query_history[-self.max_history:]
        except Exception as e:
            self.logger.error(f"Error in query mode: {e}")
            print("An error occurred in query mode.")

    def _validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate query for length, language, format, and relevance with relaxed intent checks."""
        try:
            if not query or len(query.strip()) < 3:
                self.logger.debug("Query too short or empty")
                return False, "Query too short or empty"
            
            query_lower = query.lower()
            
            # Normalize dates and numbers
            query_clean = re.sub(r'\b[jJ]an\b-(\d{1,2})-(\d{4})\b', r'January \1, \2', query_lower)
            query_clean = re.sub(r'\b(\d+)\s*(items?|products?|orders?)\b', r'\1 \2', query_clean)
            doc = self.nlp(query_clean)
            cleaned_query = ' '.join(token.text for token in doc if token.ent_type_ not in ('PERSON', 'ORG', 'GPE'))
            self.logger.debug(f"Cleaned query for validation: {cleaned_query}")
            if not cleaned_query.strip():
                cleaned_query = query_lower
            
            # Check feedback first
            if self.feedback_manager:
                feedback = self.feedback_manager.get_all_feedback()
                for entry in feedback:
                    if query_lower == entry['query'].lower():
                        self.logger.debug(f"Query matched feedback: {query}")
                        return True, ""
            
            # Check for schema and synonym relevance
            schema_match = False
            if self.schema_dict and 'tables' in self.schema_dict:
                # Load cached synonyms
                default_matches = self.name_match_manager.default_matches
                dynamic_matches = self.name_match_manager.dynamic_matches
                tokens = cleaned_query.split()
                
                for token in tokens:
                    # Exact table/column match
                    for schema in self.schema_dict['tables']:
                        for table in self.schema_dict['tables'][schema]:
                            if token in table.lower():
                                schema_match = True
                                self.logger.debug(f"Schema match: table '{token}' in query")
                                break
                            for column in self.schema_dict['columns'].get(schema, {}).get(table, []):
                                if token in column.lower():
                                    schema_match = True
                                    self.logger.debug(f"Schema match: column '{token}' in query")
                                    break
                        if schema_match:
                            break
                    # Synonym match
                    for col, synonyms in {**default_matches, **dynamic_matches}.items():
                        if token in synonyms:
                            schema_match = True
                            self.logger.debug(f"Synonym match: '{token}' for column '{col}'")
                            break
                    # Embedding-based similarity
                    if not schema_match:
                        token_embedding = self.model.encode(token, show_progress_bar=False)
                        for schema in self.schema_dict['tables']:
                            for table in self.schema_dict['tables'][schema]:
                                for column in self.schema_dict['columns'].get(schema, {}).get(table, []):
                                    col_embedding = self.model.encode(column.lower(), show_progress_bar=False)
                                    similarity = np.dot(token_embedding, col_embedding) / (
                                        np.linalg.norm(token_embedding) * np.linalg.norm(col_embedding)
                                    )
                                    if similarity > 0.65:
                                        schema_match = True
                                        self.logger.debug(f"Embedding match: '{token}' similar to '{column}' (sim={similarity:.2f})")
                                        break
                            if schema_match:
                                break
                    if schema_match:
                        break
            
            # Language detection with override
            try:
                lang_result = langdetect.detect_langs(cleaned_query)
                lang, confidence = lang_result[0].lang, lang_result[0].prob
                self.logger.debug(f"Language detection: lang={lang}, confidence={confidence:.2f}")
                if lang != 'en' and confidence > 0.9:
                    if schema_match:
                        self.logger.debug(f"Non-English detected (lang: {lang}, conf: {confidence:.2f}), but schema match found")
                        return True, ""
                    print(f"Non-English query detected (language: {lang}, confidence: {confidence:.2f})")
                    override = input("Force English processing? (y/n): ").strip().lower()
                    if override == 'y':
                        self.logger.debug(f"User forced English for query: {query}")
                        return True, ""
                    self.cache_synchronizer.write_ignored_query(query, None, f"non_english_lang_{lang}")
                    return False, f"Non-English query (detected: {lang})"
            except langdetect.lang_detect_exception.LangDetectException as e:
                self.logger.warning(f"Language detection failed for '{query}': {e}")
                if schema_match:
                    self.logger.debug(f"Language detection failed, but schema match found")
                    return True, ""
                self.cache_synchronizer.write_ignored_query(query, None, "language_detection_failed")
                return False, "Language detection failed"
            
            # Check ignored queries
            ignored_queries = self.cache_synchronizer.read_ignored_queries()
            for iq, info in ignored_queries.items():
                if query_lower == iq.lower():
                    self.logger.debug(f"Ignored query matched: {query}")
                    return False, f"Ignored query (reason: {info['reason']})"
            
            # Relaxed intent check
            intent_patterns = [
                r'^(how\s+many|what|which|list|show|get|find|count|sum|select)\b',
                r'\b(names|categories|products|orders|sales|customers|stores)\b'
            ]
            has_intent = any(re.search(pattern, query_lower) for pattern in intent_patterns)
            if not has_intent:
                self.logger.debug(f"No actionable intent in query: {query}")
                return False, "Query lacks actionable intent"
            
            # Ensure schema is initialized
            if not self.schema_dict or 'tables' not in self.schema_dict:
                self.logger.warning("Schema dictionary empty or invalid")
                return False, "Schema not initialized"
            
            if schema_match or has_intent:
                self.logger.debug(f"Query validated: {query} (schema_match={schema_match}, has_intent={has_intent})")
                return True, ""
            
            self.logger.debug(f"No relevant tables, columns, or synonyms found in query: {query}")
            self.cache_synchronizer.write_ignored_query(query, None, "irrelevant")
            return False, "No relevant tables or columns found"
        except Exception as e:
            self.logger.error(f"Error validating query: {e}")
            return False, f"Validation error: {e}"

    def _expand_query_with_synonyms(self, query: str) -> str:
        """Expand query with synonyms from query_synonyms.json."""
        try:
            for path in [self.query_synonyms_file, os.path.join("app-config", "BikeStores", "query_synonyms.json")]:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            synonyms_data = json.load(f)
                        self.logger.debug(f"Loaded query synonyms from {path}")
                        break
                    except Exception as e:
                        self.logger.error(f"Error loading synonyms from {path}: {e}")
                        continue
            else:
                self.logger.debug(f"No query synonyms file found")
                return query
            
            synonyms = synonyms_data.get('synonyms', {})
            
            query_words = query.lower().split()
            expanded_words = []
            for word in query_words:
                for key, syn_list in synonyms.items():
                    if word == key.lower():
                        expanded_words.extend(syn_list)
                        self.logger.debug(f"Expanded '{word}' to {syn_list}")
                        break
                else:
                    expanded_words.append(word)
            
            expanded_query = ' '.join(expanded_words)
            self.logger.debug(f"Expanded query: {query} -> {expanded_query}")
            return expanded_query
        except Exception as e:
            self.logger.error(f"Error expanding query with synonyms: {e}")
            return query

    def _reload_configurations(self):
        """Reload database configurations and reinitialize managers."""
        try:
            print(f"\nReloading configurations from {self.config_path}")
            if not os.path.exists(self.config_path):
                print(f"Configuration file not found: {self.config_path}")
                self.logger.error(f"Configuration file not found: {self.config_path}")
                return
            
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
            configs = [
                {**config, 'name': name}
                for name, config in config_dict.items()
                if isinstance(config, dict)
            ]
            self.logger.debug(f"Reloaded {len(configs)} configurations")
            
            print("\nAvailable Configurations:")
            for i, config in enumerate(configs, 1):
                print(f"{i}. {config.get('name', 'Unknown')}")
            print(f"{len(configs) + 1}. Cancel")
            
            choice = input("Select configuration to reload: ").strip()
            self.logger.debug(f"Reload configuration choice: {choice}")
            
            if not choice.isdigit() or int(choice) < 1 or int(choice) > len(configs) + 1:
                print("Invalid choice.")
                return
            
            if int(choice) == len(configs) + 1:
                print("Reload cancelled.")
                return
            
            config = configs[int(choice) - 1]
            db_name = config.get('name')
            required_keys = {'server', 'database', 'username', 'password', 'driver'}
            if not all(key in config for key in required_keys):
                print("Invalid configuration: missing required fields.")
                self.logger.error(f"Invalid configuration: {db_name}")
                return
            
            self.db_name = db_name
            self.schemas = config.get('schemas', [])
            self.tables = config.get('tables', [])
            for table in self.tables[:]:
                if '.' not in table:
                    self.logger.warning(f"Invalid table format: {table} in {db_name}")
                    self.tables.remove(table)
            
            self.connection_manager = DatabaseConnection()
            self.connection_manager.connect(config)
            self.schema_manager = SchemaManager(db_name, self.schemas, self.tables)
            self.schema_dict = self.schema_manager.build_schema_dictionary(self.connection_manager.connection)
            
            if not self.schema_dict or 'tables' not in self.schema_dict:
                print("Failed to reload schema dictionary.")
                self.logger.error("Failed to reload schema dictionary")
                self.connection_manager = None
                self.schema_manager = None
                self.schema_dict = {}
                return
            
            self.feedback_manager = FeedbackManager(db_name, self.cache_synchronizer)
            self.name_match_manager = NameMatchManager(db_name)
            self.pattern_manager = PatternManager(self.schema_dict, self.schemas, self.tables)
            self.table_identifier = TableIdentifier(
                self.schema_dict,
                self.feedback_manager,
                self.pattern_manager,
                self.cache_synchronizer
            )
            self.query_synonyms_file = os.path.join("app-config", db_name, "query_synonyms.json")
            self.cache_synchronizer.reload_caches(self.schema_manager, self.feedback_manager, self.name_match_manager)
            
            print(f"Configurations reloaded for {db_name}")
            self.logger.info(f"Reloaded configurations for {db_name}")
        except Exception as e:
            self.logger.error(f"Error reloading configurations: {e}")
            print(f"Error reloading configurations: {e}")

    def _manage_feedback(self):
        """Manage feedback entries with similarity search."""
        try:
            if not self.feedback_manager:
                print("Feedback manager not initialized. Please connect to a database.")
                self.logger.warning("Feedback manager not initialized")
                return
            
            while True:
                print("\nManage Feedback:")
                print("1. List Feedback")
                print("2. Add Feedback")
                print("3. Remove Feedback")
                print("4. Clear All Feedback")
                print("5. Find Similar Feedback")
                print("6. Back")
                choice = input("Select option: ").strip()
                self.logger.debug(f"Feedback option: {choice}")
                
                if choice == '1':
                    feedback = self.feedback_manager.get_all_feedback()
                    if not feedback:
                        print("No feedback found.")
                    else:
                        print("\nFeedback Entries:")
                        for entry in feedback:
                            print(f"- Query: {entry['query']}, Tables: {', '.join(entry['tables'])}, Count: {entry['count']}, Confidence: {entry['confidence']:.2f}")
                elif choice == '2':
                    query = input("Enter query: ").strip()
                    tables = input("Enter tables (comma-separated): ").strip().split(',')
                    tables = [t.strip() for t in tables if t.strip()]
                    if query and tables:
                        self.feedback_manager.store_feedback(query, tables, self.schema_dict)
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        embedding = self.model.encode(query, show_progress_bar=False) if self.model else None
                        self.cache_synchronizer.write_feedback(timestamp, query, tables, embedding)
                        print(f"Feedback added: {query} -> {tables}")
                        self.logger.debug(f"Added feedback: {query}")
                    else:
                        print("Query and tables cannot be empty.")
                elif choice == '3':
                    query = input("Enter query to remove: ").strip()
                    feedback = self.feedback_manager.get_all_feedback()
                    if any(entry['query'].lower() == query.lower() for entry in feedback):
                        self.feedback_manager.delete_feedback(query)
                        self.cache_synchronizer.delete_feedback(query)
                        print(f"Feedback removed: {query}")
                        self.logger.debug(f"Removed feedback: {query}")
                    else:
                        print("Feedback not found.")
                elif choice == '4':
                    confirm = input("Clear all feedback? (y/n): ").strip().lower()
                    if confirm == 'y':
                        self.feedback_manager.clear_feedback()
                        self.cache_synchronizer.clear_cache(table='feedback')
                        print("All feedback cleared.")
                        self.logger.debug("Cleared all feedback")
                    else:
                        print("Clear cancelled.")
                elif choice == '5':
                    query = input("Enter query to find similar feedback: ").strip()
                    similar = self.cache_synchronizer.find_similar_feedback(query)
                    if not similar:
                        print("No similar feedback found.")
                    else:
                        print("\nSimilar Feedback:")
                        for fb_query, tables, sim in similar:
                            print(f"- {fb_query} -> {', '.join(tables)} (similarity: {sim:.2f})")
                elif choice == '6':
                    break
                else:
                    print("Invalid option.")
        except Exception as e:
            self.logger.error(f"Error managing feedback: {e}")
            print("An error occurred while managing feedback.")

    def _manage_ignored_queries(self):
        """Manage ignored queries with validation."""
        try:
            while True:
                print("\nManage Ignored Queries:")
                print("1. List Ignored Queries")
                print("2. Add Ignored Query")
                print("3. Remove Ignored Query")
                print("4. Clear All Ignored Queries")
                print("5. Back")
                choice = input("Select option: ").strip()
                self.logger.debug(f"Ignored queries option: {choice}")
                
                if choice == '1':
                    ignored_queries = self.cache_synchronizer.read_ignored_queries()
                    if not ignored_queries:
                        print("No ignored queries found.")
                    else:
                        print("\nIgnored Queries:")
                        for query, info in ignored_queries.items():
                            print(f"- {query} (Reason: {info['reason']})")
                elif choice == '2':
                    query = input("Enter query to ignore: ").strip()
                    if not query or len(query.strip()) < 3:
                        print("Query must be at least 3 characters long.")
                        continue
                    ignored_queries = self.cache_synchronizer.read_ignored_queries()
                    if query.lower() in [q.lower() for q in ignored_queries]:
                        print(f"Query '{query}' is already ignored.")
                        continue
                    reason = input("Enter reason: ").strip()
                    if not reason:
                        print("Reason cannot be empty.")
                        continue
                    embedding = self.model.encode(query, show_progress_bar=False) if self.model else None
                    self.cache_synchronizer.write_ignored_query(query, embedding, reason)
                    print(f"Ignored query added: {query}")
                    self.logger.debug(f"Added ignored query: {query}")
                elif choice == '3':
                    query = input("Enter query to remove: ").strip()
                    ignored_queries = self.cache_synchronizer.read_ignored_queries()
                    if query.lower() in [q.lower() for q in ignored_queries]:
                        self.cache_synchronizer.delete_ignored_query(query)
                        print(f"Ignored query removed: {query}")
                        self.logger.debug(f"Removed ignored query: {query}")
                    else:
                        print("Query not found in ignored list.")
                elif choice == '4':
                    confirm = input("Clear all ignored queries? (y/n): ").strip().lower()
                    if confirm == 'y':
                        self.cache_synchronizer.clear_ignored_queries()
                        print("All ignored queries cleared.")
                        self.logger.debug("Cleared all ignored queries")
                    else:
                        print("Clear cancelled.")
                elif choice == '5':
                    break
                else:
                    print("Invalid option.")
        except Exception as e:
            self.logger.error(f"Error managing ignored queries: {e}")
            print("An error occurred while managing ignored queries.")

    def _view_schema(self):
        """Display schema details."""
        try:
            if not self.schema_dict or 'tables' not in self.schema_dict:
                print("Schema not initialized. Please connect to a database.")
                self.logger.warning("Schema not initialized")
                return
            
            print("\nDatabase Schema:")
            for schema in self.schema_dict['tables']:
                print(f"\nSchema: {schema}")
                for table in self.schema_dict['tables'][schema]:
                    print(f"  Table: {table}")
                    columns = self.schema_dict['columns'].get(schema, {}).get(table, [])
                    if columns:
                        print("    Columns:")
                        for col in columns:
                            col_info = self.schema_dict['columns'][schema][table][col]
                            print(f"      - {col} ({col_info['type']}, Nullable: {col_info['nullable']})")
                    primary_keys = self.schema_dict['primary_keys'].get(schema, {}).get(table, [])
                    if primary_keys:
                        print("    Primary Keys:")
                        for pk in primary_keys:
                            print(f"      - {pk}")
                    foreign_keys = self.schema_dict['foreign_keys'].get(schema, {}).get(table, [])
                    if foreign_keys:
                        print("    Foreign Keys:")
                        for fk in foreign_keys:
                            try:
                                ref_table = fk.get('ref_table', fk.get('referenced_table', 'unknown'))
                                ref_column = fk.get('ref_column', fk.get('referenced_column', 'unknown'))
                                print(f"      - {fk['column']} -> {ref_table}.{ref_column}")
                            except KeyError as e:
                                self.logger.error(f"Invalid foreign key structure for {schema}.{table}: {fk}")
                                print(f"      - {fk['column']} -> [invalid: missing {e}]")
            self.logger.debug("Displayed schema details")
        except Exception as e:
            self.logger.error(f"Error displaying schema: {e}")
            print(f"An error occurred while displaying schema: {e}")

    def _view_query_history(self):
        """Display query history with detailed status."""
        try:
            if not self.query_history:
                print("No query history available.")
                self.logger.debug("No query history")
                return
            
            print("\nQuery History:")
            for entry in self.query_history:
                status = "Rejected" if entry.get('rejected') else "Accepted"
                print(f"- Query: {entry['query']}")
                print(f"  Tables: {', '.join(entry['tables']) if entry['tables'] else 'None'}")
                print(f"  Status: {status}")
                print(f"  Confidence: {entry['confidence']:.2f}")
                print(f"  Timestamp: {entry['timestamp']}")
            self.logger.debug("Displayed query history")
        except Exception as e:
            self.logger.error(f"Error displaying query history: {e}")
            print("An error occurred while displaying query history.")

    def _generate_synthetic_feedback(self):
        """Generate synthetic feedback for training."""
        try:
            if not self.schema_dict or not self.feedback_manager:
                print("Schema or feedback manager not initialized. Please connect to a database.")
                self.logger.warning("Schema or feedback manager not initialized")
                return
            
            print("\nGenerating synthetic feedback...")
            count = 0
            for schema in self.schema_dict['tables']:
                for table in self.schema_dict['tables'][schema]:
                    table_full = f"{schema}.{table}"
                    table_name = table.lower()
                    queries = [
                        f"List all {table_name}",
                        f"Show {table_name} details",
                        f"Count {table_name}"
                    ]
                    for col in self.schema_dict['columns'][schema][table]:
                        col_lower = col.lower()
                        queries.extend([
                            f"Find {table_name} by {col_lower}",
                            f"List {table_name} with {col_lower}",
                            f"Sum {col_lower} in {table_name}"
                        ])
                    for query in queries:
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        embedding = self.model.encode(query, show_progress_bar=False) if self.model else None
                        self.feedback_manager.store_feedback(query, [table_full], self.schema_dict)
                        self.cache_synchronizer.write_feedback(timestamp, query, [table_full], embedding)
                        count += 1
            print(f"Generated {count} synthetic feedback entries.")
            self.logger.debug(f"Generated {count} synthetic feedback entries")
        except Exception as e:
            self.logger.error(f"Error generating synthetic feedback: {e}")
            print("An error occurred while generating synthetic feedback.")

    def _cleanup(self):
        """Clean up resources before exiting."""
        try:
            if self.connection_manager:
                self.connection_manager.disconnect()
                self.logger.debug("Disconnected database connection")
            if self.cache_synchronizer:
                self.cache_synchronizer.close()
                self.logger.debug("Closed cache synchronizer")
            self.query_history = []
            self.logger.debug("Cleared query history")
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")