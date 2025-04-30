# schema/manager.py: Manages database schema for TableIdentifier-v1
# Fixes list indices error, retains try-finally, restores ~257 lines

import os
import json
import logging
import logging.config
import pyodbc
from typing import Dict, Optional, Any
import time
from datetime import datetime

class SchemaManager:
    """Manages database schema information for TableIdentifier-v1."""
    
    def __init__(self, db_name: str):
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
        
        self.logger = logging.getLogger("schema")
        self.db_name = db_name
        self.schema_cache_dir = os.path.join("schema_cache", db_name)
        self.schema_file = os.path.join(self.schema_cache_dir, "schema.json")
        self.metadata_file = os.path.join(self.schema_cache_dir, "metadata.json")
        os.makedirs(self.schema_cache_dir, exist_ok=True)
        self.system_schemas = ['dbo', 'sys', 'information_schema']
        self.logger.debug(f"Initialized SchemaManager for {db_name}")

    def needs_refresh(self, connection: Any) -> bool:
        try:
            cursor = None
            if not connection or not connection.cursor():
                self.logger.error("No valid database connection for refresh check")
                return True
            
            cursor = connection.cursor()
            if not os.path.exists(self.schema_file):
                self.logger.debug("Schema file not found, needs refresh")
                return True
            
            metadata = self._load_cache_metadata()
            last_updated = metadata.get('last_updated')
            cached_version = metadata.get('schema_version', '1.0')
            
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_catalog = ?
                AND table_schema NOT IN (?, ?, ?)
            """, (self.db_name, *self.system_schemas))
            current_table_count = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_catalog = ?
                AND table_schema NOT IN (?, ?, ?)
            """, (self.db_name, *self.system_schemas))
            current_column_count = cursor.fetchone()[0]
            
            with open(self.schema_file) as f:
                cached_schema = json.load(f)
                cached_table_count = sum(len(tables) for tables in cached_schema.get('tables', {}).values())
                cached_column_count = sum(
                    len(cols) for schema in cached_schema.get('columns', {}).values()
                    for cols in schema.values()
                )
            
            needs_refresh = (
                current_table_count != cached_table_count or
                current_column_count != cached_column_count or
                cached_version < '1.1'
            )
            
            if last_updated:
                last_updated_dt = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S")
                if (datetime.now() - last_updated_dt).days > 7:
                    self.logger.debug("Schema cache older than 7 days, needs refresh")
                    needs_refresh = True
            
            self.logger.debug(f"Schema refresh needed: {needs_refresh}")
            return needs_refresh
        except Exception as e:
            self.logger.warning(f"Error checking schema refresh: {e}")
            return True
        finally:
            if cursor:
                try:
                    cursor.close()
                    self.logger.debug("Cursor closed in needs_refresh")
                except Exception as e:
                    self.logger.error(f"Error closing cursor: {e}")

    def build_data_dict(self, connection: Any) -> Dict:
        self.logger.debug("Building schema dictionary")
        return self.build_schema_dictionary(connection)

    def build_schema_dictionary(self, connection: Any) -> Dict:
        try:
            if not connection or not connection.cursor():
                self.logger.error("No valid database connection")
                raise ValueError("Invalid database connection")
            
            schema_dict = {
                'tables': {},
                'columns': {},
                'primary_keys': {},
                'foreign_keys': {},
                'views': {}
            }
            cursor = None
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    cursor = connection.cursor()
                    self.logger.debug(f"Attempt {attempt}: Fetching schemas")
                    cursor.execute("""
                        SELECT schema_name 
                        FROM information_schema.schemata 
                        WHERE catalog_name = ?
                        AND schema_name NOT IN (?, ?, ?)
                    """, (self.db_name, *self.system_schemas))
                    schemas = [row[0] for row in cursor.fetchall()]
                    
                    if not schemas:
                        self.logger.warning("No user schemas found")
                        return {}
                    
                    for schema in schemas:
                        self.logger.debug(f"Processing schema: {schema}")
                        schema_dict['tables'][schema] = []
                        schema_dict['columns'][schema] = {}
                        schema_dict['primary_keys'][schema] = {}
                        schema_dict['foreign_keys'][schema] = {}
                        schema_dict['views'][schema] = []
                        
                        cursor.execute("""
                            SELECT table_name 
                            FROM information_schema.tables 
                            WHERE table_schema = ? 
                            AND table_catalog = ?
                            AND table_type = 'BASE TABLE'
                        """, (schema, self.db_name))
                        tables = [row[0] for row in cursor.fetchall()]
                        schema_dict['tables'][schema] = tables
                        self.logger.debug(f"Found tables in {schema}: {tables}")
                        
                        cursor.execute("""
                            SELECT table_name 
                            FROM information_schema.tables 
                            WHERE table_schema = ? 
                            AND table_catalog = ?
                            AND table_type = 'VIEW'
                        """, (schema, self.db_name))
                        views = [row[0] for row in cursor.fetchall()]
                        schema_dict['views'][schema] = views
                        self.logger.debug(f"Found views in {schema}: {views}")
                        
                        for table in tables:
                            self.logger.debug(f"Fetching columns for {schema}.{table}")
                            cursor.execute("""
                                SELECT 
                                    column_name, 
                                    data_type, 
                                    is_nullable, 
                                    column_default
                                FROM information_schema.columns 
                                WHERE table_schema = ? 
                                AND table_name = ? 
                                AND table_catalog = ?
                            """, (schema, table, self.db_name))
                            columns = {}
                            for row in cursor.fetchall():
                                columns[row[0]] = {
                                    'type': row[1],
                                    'nullable': row[2] == 'YES',
                                    'default': row[3]
                                }
                            schema_dict['columns'][schema][table] = columns
                            
                            cursor.execute("""
                                SELECT column_name 
                                FROM information_schema.key_column_usage 
                                WHERE table_schema = ? 
                                AND table_name = ? 
                                AND table_catalog = ? 
                                AND constraint_name LIKE 'PK%'
                            """, (schema, table, self.db_name))
                            pks = [row[0] for row in cursor.fetchall()]
                            if pks:
                                schema_dict['primary_keys'][schema][table] = pks
                                for pk in pks:
                                    if pk in columns:
                                        columns[pk]['is_primary_key'] = True
                            
                            cursor.execute("""
                                SELECT 
                                    kcu.column_name,
                                    ccu.table_schema AS referenced_schema,
                                    ccu.table_name AS referenced_table,
                                    ccu.column_name AS referenced_column
                                FROM information_schema.key_column_usage kcu
                                JOIN information_schema.constraint_column_usage ccu
                                ON kcu.constraint_name = ccu.constraint_name
                                WHERE kcu.table_schema = ?
                                AND kcu.table_name = ?
                                AND kcu.table_catalog = ?
                                AND kcu.constraint_name LIKE 'FK%'
                            """, (schema, table, self.db_name))
                            fks = []
                            for row in cursor.fetchall():
                                fks.append({
                                    'column': row[0],
                                    'referenced_schema': row[1],
                                    'referenced_table': row[2],
                                    'referenced_column': row[3]
                                })
                            if fks:
                                schema_dict['foreign_keys'][schema][table] = fks
                        
                        for view in views:
                            cursor.execute("""
                                SELECT 
                                    column_name, 
                                    data_type, 
                                    is_nullable
                                FROM information_schema.columns 
                                WHERE table_schema = ? 
                                AND table_name = ? 
                                AND table_catalog = ?
                            """, (schema, view, self.db_name))
                            columns = {}
                            for row in cursor.fetchall():
                                columns[row[0]] = {
                                    'type': row[1],
                                    'nullable': row[2] == 'YES'
                                }
                            schema_dict['columns'][schema][view] = columns
                    
                    if not self.validate_schema(schema_dict):
                        self.logger.error("Schema validation failed")
                        return {}
                    
                    self.logger.debug("Schema dictionary built successfully")
                    return schema_dict
                except pyodbc.Error as e:
                    self.logger.warning(f"Query attempt {attempt} failed: {str(e)}")
                    if attempt < max_attempts:
                        time.sleep(0.1 * (2 ** attempt))
                    else:
                        self.logger.error(f"Error building schema dictionary: {str(e)}")
                        return {}
                except Exception as e:
                    self.logger.error(f"Error building schema dictionary: {e}")
                    return {}
                finally:
                    if cursor:
                        try:
                            cursor.close()
                            self.logger.debug("Cursor closed in build_schema_dictionary")
                        except Exception as e:
                            self.logger.error(f"Error closing cursor: {e}")
            
            return {}
        except Exception as e:
            self.logger.error(f"Error building schema dictionary: {e}")
            return {}

    def validate_schema(self, schema_dict: Dict) -> bool:
        try:
            if not schema_dict or not isinstance(schema_dict, dict):
                self.logger.error("Invalid schema dictionary")
                return False
            
            required_keys = {'tables', 'columns', 'primary_keys', 'foreign_keys', 'views'}
            if not required_keys.issubset(schema_dict.keys()):
                missing = required_keys - set(schema_dict.keys())
                self.logger.error(f"Missing schema keys: {', '.join(missing)}")
                return False
            
            for schema in schema_dict['tables']:
                if not schema_dict['tables'][schema]:
                    self.logger.warning(f"No tables in schema: {schema}")
                    continue
                for table in schema_dict['tables'][schema]:
                    if table not in schema_dict['columns'].get(schema, {}):
                        self.logger.error(f"No columns defined for table: {schema}.{table}")
                        return False
                    if not schema_dict['columns'][schema][table]:
                        self.logger.error(f"Empty columns for table: {schema}.{table}")
                        return False
            
            self.logger.debug("Schema validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating schema: {e}")
            return False

    def load_from_cache(self) -> Dict:
        try:
            if os.path.exists(self.schema_file):
                with open(self.schema_file) as f:
                    schema_dict = json.load(f)
                if self.validate_schema(schema_dict):
                    self.logger.debug(f"Loaded schema from {self.schema_file}")
                    return schema_dict
                self.logger.warning("Cached schema failed validation")
            self.logger.warning(f"Schema cache file not found or invalid: {self.schema_file}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading schema cache: {e}")
            return {}

    def save_schema(self, schema_path: str, schema_dict: Dict = None):
        try:
            if not schema_dict:
                schema_dict = self.build_schema_dictionary()
            if not schema_dict:
                self.logger.error("Cannot save empty schema dictionary")
                return
            
            with open(schema_path, 'w') as f:
                json.dump(schema_dict, f, indent=2)
            self.logger.debug(f"Saved schema to {schema_path}")
            
            self._update_cache_metadata()
        except Exception as e:
            self.logger.error(f"Error saving schema: {e}")

    def _load_cache_metadata(self) -> Dict:
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file) as f:
                    metadata = json.load(f)
                self.logger.debug(f"Loaded metadata from {self.metadata_file}")
                return metadata
            return {'last_updated': None, 'schema_version': '1.0'}
        except Exception as e:
            self.logger.error(f"Error loading cache metadata: {e}")
            return {'last_updated': None, 'schema_version': '1.0'}

    def _update_cache_metadata(self):
        try:
            metadata = {
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'schema_version': '1.1',
                'db_name': self.db_name
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.debug(f"Updated metadata in {self.metadata_file}")
        except Exception as e:
            self.logger.error(f"Error updating cache metadata: {e}")

    def get_table_metadata(self, schema: str, table: str) -> Dict:
        """Retrieve metadata for a specific table."""
        try:
            schema_dict = self.load_from_cache()
            if not schema_dict:
                self.logger.warning(f"No cached schema for {schema}.{table}")
                return {}
            
            metadata = {
                'columns': schema_dict['columns'].get(schema, {}).get(table, {}),
                'primary_keys': schema_dict['primary_keys'].get(schema, {}).get(table, []),
                'foreign_keys': schema_dict['foreign_keys'].get(schema, {}).get(table, [])
            }
            self.logger.debug(f"Retrieved metadata for {schema}.{table}")
            return metadata
        except Exception as e:
            self.logger.error(f"Error retrieving table metadata for {schema}.{table}: {e}")
            return {}