# main.py: Entry point for Database Schema Analyzer
# Updated to remove ensure_model call to fix AttributeError while preserving all functionality

import logging
import logging.config
import os
from typing import Dict, List, Tuple
from config.manager import DBConfigManager, DatabaseConnection
from config.patterns import PatternManager
from config.metadata_initializer import MetadataInitializer
from config.cache_synchronizer import CacheSynchronizer
from schema.manager import SchemaManager
from feedback.manager import FeedbackManager
from analysis.table_identifier import TableIdentifier
from analysis.name_match_manager import NameMatchManager
from analysis.processor import NLPPipeline
from nlp.QueryProcessor import QueryProcessor
from cli.interface import DatabaseAnalyzerCLI
import cProfile

class DatabaseAnalyzer:
    """Main class for database schema analysis and query processing."""
    
    def __init__(self):
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
        
        self.logger = logging.getLogger("analyzer")
        self.connection_manager = DatabaseConnection()
        self.config_manager = DBConfigManager()
        self.schema_manager = None
        self.schema_dict = {}
        self.pattern_manager = None
        self.feedback_manager = None
        self.nlp_pipeline = None
        self.name_matcher = None
        self.table_identifier = None
        self.query_processor = None
        self.cache_synchronizer = None
        self.current_config = None
        self.logger.debug("Initialized DatabaseAnalyzer")

    def run(self):
        try:
            configs = self.load_configs()
            if not configs:
                print("No configurations loaded.")
                return
            
            print("\nAvailable Configurations:")
            for i, config in enumerate(configs, 1):
                print(f"{i}. {config.get('name', config.get('database', 'Unknown'))}")
            print(f"{len(configs) + 1}. Cancel")
            
            choice = input("Select configuration: ").strip()
            self.logger.debug(f"Configuration choice: {choice}")
            
            if not choice.isdigit() or int(choice) < 1 or int(choice) > len(configs) + 1:
                print("Invalid choice.")
                return
            
            if int(choice) == len(configs) + 1:
                print("Cancelled.")
                return
            
            self.set_current_config(configs[int(choice) - 1])
            if not self.connect_to_database():
                print(f"Failed to connect to {self.current_config['database']}")
                return
            
            cli = DatabaseAnalyzerCLI(
                self.current_config['database'],
                schema_dict=self.schema_dict,
                feedback_manager=self.feedback_manager,
                schemas=self.current_config.get('schemas', []),
                tables=self.current_config.get('tables', [])
            )
            cli.run()
        except Exception as e:
            self.logger.error(f"Error running application: {e}")
            print(f"Error: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up resources on shutdown."""
        try:
            if self.table_identifier:
                self.table_identifier.save_name_matches()
                self.logger.debug("Saved name matches")
            if self.cache_synchronizer:
                self.cache_synchronizer.close()
                self.logger.debug("Closed cache synchronizer")
            if self.connection_manager:
                self.connection_manager.close()
                self.logger.debug("Closed database connection")
            self.logger.info("Application shutdown")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def load_configs(self, config_path: str = "app-config/database_configurations.json") -> List[Dict]:
        try:
            if not os.path.exists(config_path):
                self.logger.warning(f"Config file not found at {config_path}")
                config_path = input("Enter config file path: ").strip()
            configs = self.config_manager.load_configs(config_path)
            for config in configs:
                config['schemas'] = config.get('schemas', [])
                config['tables'] = config.get('tables', [])
                if not isinstance(config['schemas'], list):
                    self.logger.warning(f"Invalid schemas format in {config['name']}: expected list")
                    config['schemas'] = []
                if not isinstance(config['tables'], list):
                    self.logger.warning(f"Invalid tables format in {config['name']}: expected list")
                    config['tables'] = []
                for table in config['tables']:
                    if '.' not in table:
                        self.logger.warning(f"Invalid table format in {config['name']}: {table} (expected schema.table)")
                        config['tables'].remove(table)
            self.logger.debug(f"Loaded {len(configs)} configurations")
            return configs
        except Exception as e:
            self.logger.error(f"Error loading configs: {e}")
            return []

    def set_current_config(self, config: Dict):
        self.current_config = config
        self.logger.debug(f"Set config: {config.get('database')}")

    def connect_to_database(self) -> bool:
        if not self.current_config:
            self.logger.error("No configuration selected")
            return False
            
        if not self.connection_manager.connect(self.current_config):
            self.logger.error("Database connection failed")
            return False
            
        if not self.connection_manager.is_connected():
            self.logger.error("Connection is not valid")
            return False
            
        try:
            self._initialize_managers()
            self.logger.info(f"Connected to {self.current_config['database']}")
            return True
        except Exception as e:
            self.logger.error(f"Initialization error: {e}", exc_info=True)
            return False

    def _initialize_managers(self):
        db_name = self.current_config['database']
        self.logger.debug(f"Initializing managers for {db_name}")
        try:
            self.schema_manager = SchemaManager(db_name, self.current_config.get('schemas', []), self.current_config.get('tables', []))
            
            metadata_initializer = MetadataInitializer(db_name, self.schema_manager, self.connection_manager)
            if not metadata_initializer.initialize():
                raise RuntimeError("Metadata initialization failed")
            
            if self.schema_manager.needs_refresh(self.connection_manager.connection):
                self.logger.debug("Building fresh schema")
                self.schema_dict = self.schema_manager.build_data_dict(self.connection_manager.connection)
            else:
                self.logger.debug("Loading schema from cache")
                self.schema_dict = self.schema_manager.load_from_cache()
            
            self.cache_synchronizer = CacheSynchronizer(db_name)
            # Removed ModelSingleton().ensure_model() as model is already initialized
            self.pattern_manager = PatternManager(self.schema_dict, self.current_config.get('schemas', []), self.current_config.get('tables', []))
            self.feedback_manager = FeedbackManager(db_name, self.cache_synchronizer)
            self.nlp_pipeline = NLPPipeline(self.pattern_manager, db_name)
            self.name_matcher = NameMatchManager(db_name)
            self.table_identifier = TableIdentifier(
                self.schema_dict,
                self.feedback_manager,
                self.pattern_manager,
                self.cache_synchronizer
            )
            self.query_processor = QueryProcessor(
                self.connection_manager,
                self.schema_dict,
                self.nlp_pipeline,
                self.table_identifier,
                self.name_matcher,
                self.pattern_manager,
                db_name,
                self.cache_synchronizer
            )
            self.logger.debug("Managers initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize managers: {e}", exc_info=True)
            raise RuntimeError(f"Manager initialization failed: {e}")

    def reload_all_configurations(self) -> bool:
        if not self.connection_manager.is_connected():
            self.logger.error("Not connected to database")
            print("Not connected to database!")
            return False
            
        try:
            self.logger.debug("Rebuilding schema")
            self.schema_dict = self.schema_manager.build_data_dict(self.connection_manager.connection)
            self.pattern_manager = PatternManager(self.schema_dict, self.current_config.get('schemas', []), self.current_config.get('tables', []))
            self.feedback_manager = FeedbackManager(self.current_config['database'], self.cache_synchronizer)
            self.nlp_pipeline = NLPPipeline(self.pattern_manager, self.current_config['database'])
            self.name_matcher = NameMatchManager(self.current_config['database'])
            self.table_identifier = TableIdentifier(
                self.schema_dict,
                self.feedback_manager,
                self.pattern_manager,
                self.cache_synchronizer
            )
            self.query_processor = QueryProcessor(
                self.connection_manager,
                self.schema_dict,
                self.nlp_pipeline,
                self.table_identifier,
                self.name_matcher,
                self.pattern_manager,
                self.current_config['database'],
                self.cache_synchronizer
            )
            self.cache_synchronizer.reload_caches(self.schema_manager, self.feedback_manager, self.name_matcher)
            self.logger.info("Configurations reloaded")
            return True
        except Exception as e:
            self.logger.error(f"Reload failed: {e}")
            print(f"Reload failed: {e}")
            return False

    def process_query(self, query: str) -> Tuple[List[str], bool]:
        if not self.connection_manager.is_connected():
            self.logger.error("Not connected to database")
            print("Not connected to database!")
            return None, False
            
        if self.query_processor is None:
            self.logger.error("Query processor not initialized")
            print("Query processor not initialized. Please connect to the database.")
            return None, False
            
        try:
            tables, confidence = self.query_processor.process_query(query)
            self.logger.debug(f"Query: {query}, Tables: {tables}, Confidence: {confidence}")
            return tables, confidence
        except Exception as e:
            self.logger.error(f"Query processing error: {e}")
            print(f"Query processing error: {e}")
            return None, False

    def validate_tables_exist(self, tables: List[str]) -> Tuple[List[str], List[str]]:
        valid = []
        invalid = []
        schema_map = {s.lower(): s for s in self.schema_dict['tables']}
        
        for table in tables:
            parts = table.split('.')
            if len(parts) != 2:
                invalid.append(table)
                continue
                
            schema, table_name = parts
            schema_lower = schema.lower()
            
            if (schema_lower in schema_map and 
                table_name.lower() in {t.lower() for t in self.schema_dict['tables'][schema_map[schema_lower]]}):
                valid.append(f"{schema_map[schema_lower]}.{table_name}")
            else:
                invalid.append(table)
                
        self.logger.debug(f"Validated tables: Valid={valid}, Invalid={invalid}")
        return valid, invalid

    def generate_ddl(self, tables: List[str]):
        for table in tables:
            if '.' not in table:
                print(f"Invalid format: {table}")
                continue
                
            schema, table_name = table.split('.')
            if schema not in self.schema_dict['tables']:
                print(f"Schema not found: {schema}")
                continue
                
            if table_name not in self.schema_dict['tables'][schema]:
                print(f"Table not found: {table_name} in schema {schema}")
                continue
                
            self.logger.debug(f"Generating DDL for {schema}.{table_name}")
            metadata = self.schema_manager.get_table_metadata(schema, table_name)
            columns = metadata.get('columns', {})
            col_defs = []
            
            for col_name, col_info in columns.items():
                col_def = f"    [{col_name}] {col_info['type']}"
                if col_info.get('is_primary_key'):
                    col_def += " PRIMARY KEY"
                if not col_info.get('nullable'):
                    col_def += " NOT NULL"
                col_defs.append(col_def)
                
            print("CREATE TABLE [{}].[{}] (\n{}\n);".format(
                schema, table_name, ",\n".join(col_defs)
            ))

    def close_connection(self):
        self.connection_manager.close()
        self.logger.info("Database connection closed")

    def is_connected(self) -> bool:
        return self.connection_manager.is_connected()

    def get_all_tables(self) -> List[str]:
        tables = []
        for schema in self.schema_dict['tables']:
            tables.extend(f"{schema}.{table}" for table in self.schema_dict['tables'][schema])
        self.logger.debug(f"All tables: {tables}")
        return tables

    def confirm_tables(self, query: str, tables: List[str]):
        if self.feedback_manager:
            self.feedback_manager.store_feedback(query, tables, self.schema_dict)
            if self.table_identifier:
                self.table_identifier.update_weights_from_feedback(query, tables)
            self.logger.info(f"Confirmed tables for query: {query}")

    def update_feedback(self, query: str, tables: List[str]):
        if self.feedback_manager:
            self.feedback_manager.store_feedback(query, tables, self.schema_dict)
            if self.table_identifier:
                self.table_identifier.update_weights_from_feedback(query, tables)
            self.logger.info(f"Updated feedback for query: {query}")

    def clear_feedback(self):
        if self.feedback_manager:
            self.feedback_manager.clear_feedback()
            print("Feedback cleared")
            self.logger.info("Feedback cleared")
        else:
            self.logger.error("Feedback manager not initialized")
            print("Feedback manager not initialized. Please connect to the database.")

if __name__ == "__main__":
    analyzer = DatabaseAnalyzer()
    analyzer.run()

# Comment above 3 lines and run the below code with cProfile
# to profile the performance of the application.
# def main():
#     """Main function to run the Database Analyzer."""
#     analyzer = DatabaseAnalyzer()
#     analyzer.run()
    
# if __name__ == "__main__":
#     cProfile.run("main()", "profile.out")