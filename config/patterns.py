# config/patterns.py: Generates regex patterns for TableIdentifier-v1
# Updated to support schema/table filtering

import logging
import logging.config
import os
import re
from typing import Dict, List, Optional

class PatternManager:
    """
    Manages regex patterns for matching tables and columns in queries.
    Supports filtering by schemas and tables to reduce noise.
    """
    
    def __init__(self, schema_dict: Dict, schemas: List[str] = None, tables: List[str] = None):
        """
        Initialize PatternManager with schema dictionary and optional schema/table filters.

        Args:
            schema_dict: Dictionary containing schema metadata (tables, columns, etc.).
            schemas: List of schemas to include (e.g., ['hr', 'sales']). If empty, uses all schemas.
            tables: List of tables to include (e.g., ['dbo.DimAccount']). Takes precedence over schemas.
        """
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
        
        self.logger = logging.getLogger("pattern_manager")
        self.schema_dict = schema_dict
        self.schemas = schemas or []
        self.tables = tables or []
        # Validate table format
        for table in self.tables:
            if '.' not in table:
                self.logger.warning(f"Invalid table format: {table} (expected schema.table)")
                self.tables.remove(table)
        self.patterns = {}
        self._generate_patterns()
        self.logger.debug(f"Initialized PatternManager with schemas={self.schemas}, tables={self.tables}")

    def _generate_patterns(self):
        """
        Generate regex patterns for tables and columns based on schema dictionary.
        Filters patterns to specified schemas or tables.
        """
        try:
            self.patterns.clear()
            if not self.schema_dict or 'tables' not in self.schema_dict:
                self.logger.error("Invalid or empty schema dictionary")
                return

            # Determine which schemas/tables to process
            if self.tables:
                # Use only specified tables
                target_tables = [(table.split('.')[0], table.split('.')[1]) for table in self.tables]
                self.logger.debug(f"Generating patterns for specified tables: {self.tables}")
            else:
                # Use specified schemas or all schemas
                target_schemas = self.schemas if self.schemas else self.schema_dict['tables'].keys()
                target_tables = []
                for schema in target_schemas:
                    if schema in self.schema_dict['tables']:
                        for table in self.schema_dict['tables'][schema]:
                            target_tables.append((schema, table))
                self.logger.debug(f"Generating patterns for schemas: {target_schemas}")

            for schema, table in target_tables:
                if schema not in self.schema_dict['tables'] or table not in self.schema_dict['tables'][schema]:
                    self.logger.warning(f"Skipping invalid table: {schema}.{table}")
                    continue
                
                table_full = f"{schema}.{table}"
                # Pattern for table name
                table_pattern = r'\b' + re.escape(table.lower()) + r'\b'
                self.patterns[table_pattern] = [table_full]
                self.logger.debug(f"Generated pattern: '{table_pattern}' -> {table_full}")

                # Patterns for columns
                columns = self.schema_dict['columns'].get(schema, {}).get(table, {})
                for column in columns:
                    col_pattern = r'\b' + re.escape(column.lower()) + r'\b'
                    if col_pattern not in self.patterns:
                        self.patterns[col_pattern] = []
                    self.patterns[col_pattern].append(table_full)
                    self.logger.debug(f"Generated pattern: '{col_pattern}' -> {table_full}")

            self.logger.debug(f"Generated {len(self.patterns)} patterns")
        except Exception as e:
            self.logger.error(f"Error generating patterns: {e}")

    def get_patterns(self) -> Dict[str, List[str]]:
        """
        Retrieve the generated patterns.

        Returns:
            Dictionary mapping regex patterns to lists of table names (schema.table).
        """
        try:
            return self.patterns
        except Exception as e:
            self.logger.error(f"Error retrieving patterns: {e}")
            return {}

    def update_patterns(self, schema_dict: Dict):
        """
        Update patterns with a new schema dictionary.

        Args:
            schema_dict: Updated schema dictionary.
        """
        try:
            self.schema_dict = schema_dict
            self._generate_patterns()
            self.logger.debug("Updated patterns with new schema dictionary")
        except Exception as e:
            self.logger.error(f"Error updating patterns: {e}")