# config/patterns.py: Manages regex patterns for query matching
# Requires schema_dict, ~120 lines

import logging
import logging.config
import os
import re
from typing import Dict, List

class PatternManager:
    """
    Manages regex patterns for matching queries to tables and columns.
    Uses schema dictionary to generate patterns dynamically.
    """
    
    def __init__(self, schema_dict: Dict):
        """
        Initialize PatternManager with schema dictionary.
        
        Args:
            schema_dict: Dictionary of schema metadata (tables, columns, etc.).
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
        
        self.logger = logging.getLogger("pattern_manager")
        self.schema_dict = schema_dict
        self.patterns = {}
        self._generate_patterns()
        self.logger.debug("Initialized PatternManager")

    def _generate_patterns(self):
        """
        Generate regex patterns based on schema tables and columns.
        """
        try:
            self.patterns = {}
            for schema in self.schema_dict['tables']:
                for table in self.schema_dict['tables'][schema]:
                    table_lower = table.lower()
                    table_full = f"{schema}.{table}"
                    # Pattern for table name
                    pattern = rf"\b{re.escape(table_lower)}\b"
                    if pattern not in self.patterns:
                        self.patterns[pattern] = []
                    self.patterns[pattern].append(table_full)
                    self.logger.debug(f"Added pattern '{pattern}' for table '{table_full}'")
                    
                    # Patterns for columns
                    for col in self.schema_dict['columns'].get(schema, {}).get(table, []):
                        col_lower = col.lower()
                        col_pattern = rf"\b{re.escape(col_lower)}\b"
                        if col_pattern not in self.patterns:
                            self.patterns[col_pattern] = []
                        self.patterns[col_pattern].append(table_full)
                        self.logger.debug(f"Added pattern '{col_pattern}' for column '{col_lower}' in '{table_full}'")
            
            # Generic patterns
            generic_patterns = {
                r"\blist\s+all\b": [],
                r"\bshow\s+all\b": [],
                r"\bget\s+details\b": []
            }
            for pattern in generic_patterns:
                for schema in self.schema_dict['tables']:
                    for table in self.schema_dict['tables'][schema]:
                        table_full = f"{schema}.{table}"
                        generic_patterns[pattern].append(table_full)
                self.patterns[pattern] = generic_patterns[pattern]
                self.logger.debug(f"Added generic pattern '{pattern}' for all tables")
        except Exception as e:
            self.logger.error(f"Error generating patterns: {e}")
            self.patterns = {}

    def get_patterns(self) -> Dict[str, List[str]]:
        """
        Retrieve all patterns and their associated tables.
        
        Returns:
            Dictionary mapping patterns to lists of table names.
        """
        try:
            self.logger.debug(f"Returning {len(self.patterns)} patterns")
            return self.patterns
        except Exception as e:
            self.logger.error(f"Error retrieving patterns: {e}")
            return {}

    def add_pattern(self, pattern: str, tables: List[str]):
        """
        Add a custom pattern with associated tables.
        
        Args:
            pattern: Regex pattern string.
            tables: List of table names (schema.table format).
        """
        try:
            if pattern not in self.patterns:
                self.patterns[pattern] = []
            self.patterns[pattern].extend(tables)
            self.patterns[pattern] = list(set(self.patterns[pattern]))
            self.logger.debug(f"Added custom pattern '{pattern}' for tables: {tables}")
        except Exception as e:
            self.logger.error(f"Error adding pattern '{pattern}': {e}")

    def remove_pattern(self, pattern: str):
        """
        Remove a pattern from the manager.
        
        Args:
            pattern: Regex pattern string to remove.
        """
        try:
            if pattern in self.patterns:
                del self.patterns[pattern]
                self.logger.debug(f"Removed pattern '{pattern}'")
        except Exception as e:
            self.logger.error(f"Error removing pattern '{pattern}': {e}")