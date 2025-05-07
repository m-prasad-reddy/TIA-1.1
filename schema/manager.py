# schema/manager.py: Manages database schema for TableIdentifier-v2.1
# Enhanced connection validation, cursor handling, and cache fallback

import os
import json
import logging
import logging.config
from typing import Dict, Optional, Any, List
import pyodbc
from contextlib import contextmanager
from datetime import datetime
import time

class SchemaManager:
    """Manages database schema information for TableIdentifier-v2.1."""
    
    def __init__(self, db_name: str, schemas: List[str] = None, tables: List[str] = None):
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
        self.system_schemas = [
            'dbo', 'sys', 'information_schema',
            'db_accessadmin', 'db_backupoperator', 'db_datareader', 'db_datawriter',
            'db_ddladmin', 'db_denydatareader', 'db_denydatawriter', 'db_owner',
            'db_securityadmin', 'guest'
        ]
        self.schemas = schemas or []
        self.tables = tables or []
        for table in self.tables[:]:
            if '.' not in table:
                self.logger.warning(f"Invalid table format: {table} (expected schema.table)")
                self.tables.remove(table)
        self.logger.debug(f"Initialized SchemaManager for {db_name}, schemas={self.schemas}, tables={self.tables}")

    @contextmanager
    def _get_cursor(self, connection: pyodbc.Connection):
        """Context manager for cursor handling."""
        cursor = None
        try:
            cursor = connection.cursor()
            yield cursor
        except Exception as e:
            self.logger.error(f"Cursor error: {e}")
            raise
        finally:
            if cursor:
                try:
                    cursor.close()
                    self.logger.debug("Cursor closed")
                except Exception as e:
                    self.logger.error(f"Error closing cursor: {e}")

    def needs_refresh(self, connection: Any) -> bool:
        """Check if schema cache needs refreshing."""
        try:
            if not connection or not hasattr(connection, 'cursor'):
                self.logger.error("No valid database connection for refresh check")
                return True
            
            with self._get_cursor(connection) as cursor:
                if not os.path.exists(self.schema_file):
                    self.logger.debug("Schema file not found, needs refresh")
                    return True
                
                metadata = self._load_cache_metadata()
                last_updated = metadata.get('last_updated')
                cached_version = metadata.get('schema_version', '1.0')
                cached_schemas = metadata.get('schemas', [])
                cached_tables = metadata.get('tables', [])
                
                if cached_schemas != self.schemas or cached_tables != self.tables:
                    self.logger.debug("Schema or table configuration changed, needs refresh")
                    return True
                
                if self.tables:
                    table_count = len(self.tables)
                    column_count = 0
                    index_count = 0
                    for table in self.tables:
                        schema, table_name = table.split('.')
                        cursor.execute("""
                            SELECT COUNT(*)
                            FROM information_schema.columns
                            WHERE table_schema = ? AND table_name = ? AND table_catalog = ?
                        """, (schema, table_name, self.db_name))
                        column_count += cursor.fetchone()[0]
                        cursor.execute("""
                            SELECT COUNT(*)
                            FROM sys.indexes
                            WHERE object_id = OBJECT_ID(?)
                        """, (f"{schema}.{table_name}",))
                        index_count += cursor.fetchone()[0]
                else:
                    schemas = self.schemas if self.schemas else []
                    if not schemas:
                        cursor.execute("""
                            SELECT schema_name
                            FROM information_schema.schemata
                            WHERE catalog_name = ?
                            AND schema_name NOT IN ({})
                        """.format(','.join('?' * len(self.system_schemas))), (self.db_name, *self.system_schemas))
                        schemas = [row[0] for row in cursor.fetchall()]
                    
                    table_count = 0
                    column_count = 0
                    index_count = 0
                    for schema in schemas:
                        cursor.execute("""
                            SELECT COUNT(*)
                            FROM information_schema.tables
                            WHERE table_schema = ? AND table_catalog = ? AND table_type = 'BASE TABLE'
                        """, (schema, self.db_name))
                        table_count += cursor.fetchone()[0]
                        cursor.execute("""
                            SELECT COUNT(*)
                            FROM information_schema.columns
                            WHERE table_schema = ? AND table_catalog = ?
                        """, (schema, self.db_name))
                        column_count += cursor.fetchone()[0]
                        cursor.execute("""
                            SELECT COUNT(*)
                            FROM sys.indexes i
                            JOIN sys.tables t ON i.object_id = t.object_id
                            JOIN sys.schemas s ON t.schema_id = s.schema_id
                            WHERE s.name = ?
                        """, (schema,))
                        index_count += cursor.fetchone()[0]
                
                with open(self.schema_file) as f:
                    cached_schema = json.load(f)
                    cached_table_count = sum(len(tables) for tables in cached_schema.get('tables', {}).values())
                    cached_column_count = sum(
                        len(cols) for schema in cached_schema.get('columns', {}).values()
                        for cols in schema.values()
                    )
                    cached_index_count = sum(
                        len(cached_schema.get('indexes', {}).get(schema, {}).get(table, []))
                        for schema in cached_schema.get('tables', {})
                        for table in cached_schema['tables'][schema]
                    )
                
                needs_refresh = (
                    table_count != cached_table_count or
                    column_count != cached_column_count or
                    index_count != cached_index_count or
                    cached_version < '1.2'
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

    def build_data_dict(self, connection: Any) -> Dict:
        """Alias for build_schema_dictionary."""
        self.logger.debug("Building schema dictionary")
        return self.build_schema_dictionary(connection)

    def build_schema_dictionary(self, connection: Any) -> Dict:
        """Build a comprehensive schema dictionary."""
        try:
            if not connection or not hasattr(connection, 'cursor'):
                self.logger.error("No valid database connection")
                raise ValueError("Invalid database connection")
            
            schema_dict = {
                'tables': {},
                'columns': {},
                'primary_keys': {},
                'foreign_keys': {},
                'views': {},
                'indexes': {}
            }
            max_attempts = 3
            
            for attempt in range(1, max_attempts + 1):
                try:
                    with self._get_cursor(connection) as cursor:
                        self.logger.debug(f"Attempt {attempt}: Fetching schemas")
                        
                        if self.tables:
                            schemas = list(set(table.split('.')[0] for table in self.tables))
                            self.logger.debug(f"Using schemas from tables: {schemas}")
                        elif self.schemas:
                            schemas = self.schemas
                            self.logger.debug(f"Using specified schemas: {schemas}")
                        else:
                            cursor.execute("""
                                SELECT schema_name 
                                FROM information_schema.schemata 
                                WHERE catalog_name = ?
                                AND schema_name NOT IN ({})
                            """.format(','.join('?' * len(self.system_schemas))), (self.db_name, *self.system_schemas))
                            schemas = [row[0] for row in cursor.fetchall()]
                            self.logger.debug(f"Using all non-system schemas: {schemas}")
                        
                        if not schemas:
                            self.logger.warning("No schemas found")
                            break
                        
                        for schema in schemas:
                            self.logger.debug(f"Processing schema: {schema}")
                            schema_dict['tables'][schema] = []
                            schema_dict['columns'][schema] = {}
                            schema_dict['primary_keys'][schema] = {}
                            schema_dict['foreign_keys'][schema] = {}
                            schema_dict['views'][schema] = []
                            schema_dict['indexes'][schema] = {}
                            
                            if self.tables:
                                tables = [table.split('.')[1] for table in self.tables if table.split('.')[0] == schema]
                                valid_tables = []
                                for table in tables:
                                    cursor.execute("""
                                        SELECT COUNT(*)
                                        FROM information_schema.tables
                                        WHERE table_schema = ? AND table_name = ? AND table_catalog = ?
                                        AND table_type = 'BASE TABLE'
                                    """, (schema, table, self.db_name))
                                    if cursor.fetchone()[0] > 0:
                                        valid_tables.append(table)
                                    else:
                                        self.logger.warning(f"Table not found: {schema}.{table}")
                                tables = valid_tables
                            else:
                                cursor.execute("""
                                    SELECT table_name 
                                    FROM information_schema.tables 
                                    WHERE table_schema = ? 
                                    AND table_catalog = ?
                                    AND table_type = 'BASE TABLE'
                                """, (schema, self.db_name))
                                tables = [row[0] for row in cursor.fetchall()]
                            
                            schema_dict['tables'][schema] = tables
                            self.logger.debug(f"Found {len(tables)} tables in {schema}: {tables}")
                            
                            cursor.execute("""
                                SELECT table_name 
                                FROM information_schema.tables 
                                WHERE table_schema = ? 
                                AND table_catalog = ?
                                AND table_type = 'VIEW'
                            """, (schema, self.db_name))
                            views = [row[0] for row in cursor.fetchall()]
                            schema_dict['views'][schema] = views
                            self.logger.debug(f"Found {len(views)} views in {schema}: {views}")
                            
                            # Batch fetch columns for all tables in schema
                            if tables:
                                table_placeholders = ','.join('?' * len(tables))
                                cursor.execute("""
                                    SELECT table_name, column_name, data_type, is_nullable, column_default
                                    FROM information_schema.columns 
                                    WHERE table_schema = ? 
                                    AND table_name IN ({})
                                    AND table_catalog = ?
                                """.format(table_placeholders), (schema, *tables, self.db_name))
                                columns_by_table = {}
                                for row in cursor.fetchall():
                                    table = row[0]
                                    columns_by_table.setdefault(table, {})[row[1]] = {
                                        'type': row[2],
                                        'nullable': row[3] == 'YES',
                                        'default': row[4]
                                    }
                                for table in tables:
                                    schema_dict['columns'][schema][table] = columns_by_table.get(table, {})
                            
                            # Fetch constraints and indexes
                            for table in tables:
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
                                        if pk in schema_dict['columns'][schema][table]:
                                            schema_dict['columns'][schema][table][pk]['is_primary_key'] = True
                                
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
                                fks = [
                                    {
                                        'column': row[0],
                                        'referenced_schema': row[1],
                                        'referenced_table': row[2],
                                        'referenced_column': row[3]
                                    }
                                    for row in cursor.fetchall()
                                ]
                                if fks:
                                    schema_dict['foreign_keys'][schema][table] = fks
                                
                                cursor.execute("""
                                    SELECT i.name
                                    FROM sys.indexes i
                                    JOIN sys.tables t ON i.object_id = t.object_id
                                    JOIN sys.schemas s ON t.schema_id = s.schema_id
                                    WHERE s.name = ? AND t.name = ?
                                """, (schema, table))
                                indexes = [row[0] for row in cursor.fetchall()]
                                if indexes:
                                    schema_dict['indexes'][schema][table] = indexes
                            
                            for view in views:
                                cursor.execute("""
                                    SELECT column_name, data_type, is_nullable
                                    FROM information_schema.columns 
                                    WHERE table_schema = ? 
                                    AND table_name = ? 
                                    AND table_catalog = ?
                                """, (schema, view, self.db_name))
                                columns = {
                                    row[0]: {
                                        'type': row[1],
                                        'nullable': row[2] == 'YES'
                                    }
                                    for row in cursor.fetchall()
                                }
                                schema_dict['columns'][schema][view] = columns
                        
                        if self.validate_schema(schema_dict):
                            self.save_schema(self.schema_file, schema_dict)
                            self.logger.debug("Schema dictionary built successfully")
                            return schema_dict
                        else:
                            self.logger.warning(f"Schema validation failed on attempt {attempt}")
                            if attempt == max_attempts:
                                self.logger.error("Schema validation failed after all attempts")
                                break
                except pyodbc.Error as e:
                    self.logger.warning(f"Query attempt {attempt} failed: {str(e)}")
                    if attempt < max_attempts:
                        time.sleep(0.1 * (2 ** attempt))
                    else:
                        self.logger.error(f"Failed to build schema dictionary after {max_attempts} attempts: {str(e)}")
                        break
            
            # Fallback to cache if live fetching fails
            cached_schema = self.load_from_cache()
            if cached_schema:
                self.logger.info("Using cached schema as fallback")
                return cached_schema
            else:
                self.logger.error("No valid schema available (live fetch failed and no valid cache)")
                return {}
        except Exception as e:
            self.logger.error(f"Error building schema dictionary: {e}")
            cached_schema = self.load_from_cache()
            if cached_schema:
                self.logger.info("Using cached schema as fallback")
                return cached_schema
            return {}

    def validate_schema(self, schema_dict: Dict) -> bool:
        """Validate schema dictionary for completeness."""
        try:
            if not schema_dict or not isinstance(schema_dict, dict):
                self.logger.error("Invalid schema dictionary")
                return False
            
            required_keys = {'tables', 'columns', 'primary_keys', 'foreign_keys', 'views', 'indexes'}
            if not required_keys.issubset(schema_dict.keys()):
                missing = required_keys - set(schema_dict.keys())
                self.logger.error(f"Missing schema keys: {', '.join(missing)}")
                return False
            
            has_tables = False
            for schema in schema_dict['tables']:
                if schema_dict['tables'][schema]:
                    has_tables = True
                    for table in schema_dict['tables'][schema]:
                        if table not in schema_dict['columns'].get(schema, {}):
                            self.logger.error(f"No columns defined for table: {schema}.{table}")
                            return False
                        if not schema_dict['columns'][schema][table]:
                            self.logger.error(f"Empty columns for table: {schema}.{table}")
                            return False
                        if table in schema_dict['primary_keys'].get(schema, {}) and not schema_dict['primary_keys'][schema][table]:
                            self.logger.warning(f"Empty primary keys for table: {schema}.{table}")
            
            if not has_tables:
                self.logger.warning("No tables found in any schema")
                return False
            
            self.logger.debug("Schema validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating schema: {e}")
            return False

    def load_from_cache(self) -> Dict:
        """Load schema from cache."""
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
        """Save schema to file."""
        try:
            if not schema_dict:
                self.logger.error("Cannot save empty schema dictionary")
                return
            
            if not self.validate_schema(schema_dict):
                self.logger.error("Schema validation failed, not saving")
                return
            
            with open(schema_path, 'w') as f:
                json.dump(schema_dict, f, indent=2)
            self.logger.debug(f"Saved schema to {schema_path}")
            
            self._update_cache_metadata()
        except Exception as e:
            self.logger.error(f"Error saving schema: {e}")

    def _load_cache_metadata(self) -> Dict:
        """Load cache metadata."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file) as f:
                    metadata = json.load(f)
                self.logger.debug(f"Loaded metadata from {self.metadata_file}")
                return metadata
            return {'last_updated': None, 'schema_version': '1.2', 'schemas': [], 'tables': []}
        except Exception as e:
            self.logger.error(f"Error loading cache metadata: {e}")
            return {'last_updated': None, 'schema_version': '1.2', 'schemas': [], 'tables': []}

    def _update_cache_metadata(self):
        """Update cache metadata."""
        try:
            metadata = {
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'schema_version': '1.2',
                'db_name': self.db_name,
                'schemas': self.schemas,
                'tables': self.tables
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
                'foreign_keys': schema_dict['foreign_keys'].get(schema, {}).get(table, []),
                'indexes': schema_dict['indexes'].get(schema, {}).get(table, [])
            }
            self.logger.debug(f"Retrieved metadata for {schema}.{table}")
            return metadata
        except Exception as e:
            self.logger.error(f"Error retrieving table metadata for {schema}.{table}: {e}")
            return {}

    def validate_cache_consistency(self, connection: Any) -> bool:
        """Validate cache consistency with database."""
        try:
            cached_schema = self.load_from_cache()
            if not cached_schema:
                self.logger.warning("No cached schema to validate")
                return False
            
            live_schema = self.build_schema_dictionary(connection)
            if not live_schema:
                self.logger.error("Failed to build live schema for validation")
                return False
            
            for schema in cached_schema['tables']:
                if schema not in live_schema['tables']:
                    self.logger.error(f"Schema {schema} missing in live database")
                    return False
                for table in cached_schema['tables'][schema]:
                    if table not in live_schema['tables'][schema]:
                        self.logger.error(f"Table {schema}.{table} missing in live database")
                        return False
                    cached_columns = set(cached_schema['columns'][schema][table].keys())
                    live_columns = set(live_schema['columns'][schema][table].keys())
                    if not cached_columns.issubset(live_columns):
                        self.logger.error(f"Column mismatch for {schema}.{table}: missing {cached_columns - live_columns}")
                        return False
            
            self.logger.debug("Cache consistency validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating cache consistency: {e}")
            return False