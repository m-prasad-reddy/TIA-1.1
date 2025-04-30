# config/manager.py: Manages database configurations and connections
# Enhances connection validation with detailed logging

import os
import json
import pyodbc
from typing import List, Dict, Optional
import logging

class DatabaseConnection:
    """Handles database connections using pyodbc."""
    
    def __init__(self):
        self.connection = None
        self.current_config = None
        self.logger = logging.getLogger("database_connection")

    def connect(self, config: Dict) -> bool:
        try:
            conn_str = (
                f"DRIVER={config['driver']};"
                f"SERVER={config['server']};"
                f"DATABASE={config['database']};"
                f"UID={config['username']};"
                f"PWD={config['password']}"
            )
            self.logger.debug(f"Attempting connection: DRIVER={config['driver']}, SERVER={config['server']}, DATABASE={config['database']}")
            self.connection = pyodbc.connect(conn_str, timeout=5)
            if self.connection is None or not self.connection.cursor():
                self.logger.error("Connection established but connection object or cursor is invalid")
                return False
            self.current_config = config
            self.logger.info(f"Connected to {config['database']}")
            return True
        except pyodbc.Error as e:
            self.logger.error(f"Connection failed: {str(e)}")
            if 'IM002' in str(e):
                self.logger.warning("ODBC driver not found. Check if 'ODBC Driver 17 for SQL Server' is installed.")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected connection error: {str(e)}")
            return False

    def close(self):
        if self.connection:
            try:
                self.connection.close()
                self.logger.debug("Connection closed")
            except Exception as e:
                self.logger.error(f"Error closing connection: {str(e)}")
            self.connection = None
            self.current_config = None

    def is_connected(self) -> bool:
        try:
            if self.connection is None:
                return False
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception as e:
            self.logger.error(f"Connection validation failed: {str(e)}")
            return False

    def get_cursor(self) -> Optional[pyodbc.Cursor]:
        try:
            return self.connection.cursor() if self.connection else None
        except Exception as e:
            self.logger.error(f"Error getting cursor: {str(e)}")
            return None

class DBConfigManager:
    """Manages loading and validation of database configurations."""
    
    def load_configs(self, config_path: str) -> List[Dict]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        try:
            with open(config_path) as f:
                config_dict = json.load(f)
            
            configs = self._validate_and_convert_configs(config_dict)
            return configs
        except Exception as e:
            self.logger.error(f"Error loading configs: {str(e)}")
            return []

    def _validate_and_convert_configs(self, config_dict: Dict) -> List[Dict]:
        if not isinstance(config_dict, dict):
            raise ValueError("Config file should contain a dictionary of configurations")
        
        configs = []
        required_keys = {'server', 'database', 'username', 'password', 'driver'}
        
        for config_name, config in config_dict.items():
            if not isinstance(config, dict):
                raise ValueError(f"Configuration '{config_name}' must be a dictionary")
            if not required_keys.issubset(config.keys()):
                missing = required_keys - set(config.keys())
                raise ValueError(f"Missing keys in config '{config_name}': {', '.join(missing)}")
            
            config_with_name = config.copy()
            config_with_name['name'] = config_name
            configs.append(config_with_name)
        
        if not configs:
            raise ValueError("No valid configurations found")
        
        return configs