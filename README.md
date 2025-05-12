# Table Identifier Agent

## Overview
The **Table Identifier Agent** is a Python-based application designed to analyze database schemas and process natural language queries to identify relevant tables in a relational database. It leverages natural language processing (NLP), machine learning (ML), and database metadata to map user queries to database tables, making it valuable for data analysts, database administrators, and developers who need to explore or query complex databases without deep SQL knowledge.

The application is built to be modular, extensible, and configurable, supporting various database systems (e.g., SQL Server) and providing features like schema caching, feedback-driven learning, and synonym generation. It includes an interactive command-line interface (CLI) for user interaction and supports performance profiling for optimization.

This document provides an overview of the application's architecture, details its modules and components, explains the data flow, and offers guidance for developers to contribute to the project.

## Features
- **Natural Language Query Processing**: Converts user queries (e.g., "total sales amount at storename=Baldwin Bikes") into relevant database tables (e.g., `sales.orders`, `sales.stores`).
- **Schema Analysis**: Extracts and caches database schema metadata (tables, columns, relationships).
- **Feedback Learning**: Stores user feedback to improve table identification accuracy over time.
- **Synonym Generation**: Dynamically generates synonyms for column names and queries to enhance query matching.
- **Interactive CLI**: Provides a user-friendly interface to process queries, view schemas, manage feedback, and generate DDL statements.
- **Extensibility**: Supports multiple database configurations and modular components for easy customization.
- **Performance Profiling**: Includes optional `cProfile` support to identify bottlenecks.

## Architecture
The Table Identifier Agent follows a modular architecture with loosely coupled components that interact through well-defined interfaces. The application is divided into layers: **Configuration**, **Core Processing**, **Data Management**, **Analysis**, **NLP**, and **User Interface**. Each layer contains specific modules that handle distinct responsibilities, ensuring maintainability and scalability.

### High-Level Architecture
```
+-------------------+
|   User Interface  |  (CLI: DatabaseAnalyzerCLI)
|   (main.py, cli/) |
+-------------------+
          |
          v
+-------------------+
| Core Processing   |  (DatabaseAnalyzer, QueryProcessor)
|   (main.py, nlp/) |
+-------------------+
          |
          v
+-------------------+
|    Analysis       |  (TableIdentifier, NameMatchManager, NLPPipeline)
| (analysis/)       |
+-------------------+
          |
          v
+-------------------+
|  Data Management  |  (SchemaManager, FeedbackManager, CacheSynchronizer)
| (schema/, feedback/, config/) |
+-------------------+
          |
          v
+-------------------+
|   Configuration   |  (DBConfigManager, DatabaseConnection, ModelSingleton)
|   (config/)       |
+-------------------+
          |
          v
+-------------------+
| Database & Cache  |  (SQL Server, SQLite cache.db)
| (app-config/)     |
+-------------------+
```

### Layers and Responsibilities
1. **Configuration Layer** (`config/`):
   - Manages database connections, configurations, and shared resources (e.g., ML models).
   - Key components: `DBConfigManager`, `DatabaseConnection`, `ModelSingleton`, `MetadataInitializer`, `CacheSynchronizer`.

2. **Data Management Layer** (`schema/`, `feedback/`, `config/`):
   - Handles schema extraction, feedback storage, and caching of metadata and embeddings.
   - Key components: `SchemaManager`, `FeedbackManager`, `CacheSynchronizer`.

3. **Analysis Layer** (`analysis/`):
   - Performs table identification, name matching, and NLP preprocessing.
   - Key components: `TableIdentifier`, `NameMatchManager`, `NLPPipeline`, `PatternManager`.

4. **Core Processing Layer** (`main.py`, `nlp/`):
   - Orchestrates query processing and application flow.
   - Key components: `DatabaseAnalyzer`, `QueryProcessor`.

5. **User Interface Layer** (`cli/`, `main.py`):
   - Provides an interactive CLI for user input and output.
   - Key component: `DatabaseAnalyzerCLI`.

## Modules and Components
Below is a detailed explanation of each module and its components, including their purpose, key functionalities, and file locations.

### 1. Configuration Layer (`config/`)
Handles setup, database connections, and shared resources.

- **DBConfigManager** (`config/manager.py`):
  - **Purpose**: Loads and validates database configurations from `app-config/database_configurations.json`.
  - **Functionality**:
    - Reads JSON configuration (e.g., database name, driver, server, schemas, tables).
    - Validates schema and table formats (e.g., `schema.table`).
    - Provides configuration options for multiple databases.
  - **Key Methods**:
    - `load_configs(config_path)`: Loads and validates configurations.
  - **Dependencies**: Python `json`, `os`.

- **DatabaseConnection** (`config/manager.py`):
  - **Purpose**: Manages connections to the target database (e.g., SQL Server).
  - **Functionality**:
    - Establishes connections using `pyodbc`.
    - Supports ODBC drivers (e.g., `ODBC Driver 17 for SQL Server`).
    - Provides methods to check connection status and close connections.
  - **Key Methods**:
    - `connect(config)`: Connects to the database using config parameters.
    - `is_connected()`: Checks if the connection is active.
    - `close()`: Closes the connection.
  - **Dependencies**: `pyodbc`.

- **ModelSingleton** (`config/model_singleton.py`):
  - **Purpose**: Manages a single instance of the `SentenceTransformer` model (`all-MiniLM-L6-v2`) for embedding generation.
  - **Functionality**:
    - Implements the singleton pattern to ensure one model instance.
    - Loads the model on initialization, supporting CUDA if available.
    - Provides access to the model for encoding queries and column names.
  - **Key Methods**:
    - `__new__()`: Enforces singleton behavior.
    - `_initialize()`: Loads the model and sets up logging.
  - **Dependencies**: `sentence_transformers`, `torch`, `logging`.

- **MetadataInitializer** (`config/metadata_initializer.py`):
  - **Purpose**: Initializes metadata caches (schema, weights, name matches, synonyms, feedback).
  - **Functionality**:
    - Builds and caches schema metadata from the database.
    - Generates initial weights for tables and columns.
    - Creates default name matches and query synonyms using NLP.
    - Manages feedback deduplication and migration of file-based caches (e.g., `query_synonyms.json`).
  - **Key Methods**:
    - `initialize()`: Orchestrates metadata initialization.
    - `_build_schema_cache()`: Extracts schema metadata.
    - `_build_weights()`: Assigns initial weights to tables/columns.
    - `_build_default_name_matches()`: Generates column synonyms.
    - `_generate_query_synonyms()`: Creates query-based synonyms.
    - `_build_synthetic_feedback()`: Generates synthetic feedback.
  - **Dependencies**: `spacy`, `sentence_transformers`, `sqlite3`, `json`.

- **CacheSynchronizer** (`config/cache_synchronizer.py`):
  - **Purpose**: Manages the SQLite cache (`cache.db`) for storing metadata, feedback, and embeddings.
  - **Functionality**:
    - Creates and maintains tables (`weights`, `name_matches`, `feedback`, `ignored_queries`).
    - Handles CRUD operations for cached data.
    - Supports indexing for efficient queries (e.g., `idx_feedback_query`).
    - Uses WAL mode for concurrent access.
  - **Key Methods**:
    - `__init__(db_name)`: Initializes the SQLite database.
    - `write_name_matches(matches)`: Writes name match entries.
    - `write_feedback(feedback)`: Stores feedback entries.
    - `get_similar_feedback(query, threshold)`: Retrieves similar feedback based on embeddings.
  - **Dependencies**: `sqlite3`, `numpy`.

### 2. Data Management Layer (`schema/`, `feedback/`, `config/`)
Manages schema metadata, feedback, and cached data.

- **SchemaManager** (`schema/manager.py`):
  - **Purpose**: Extracts and caches database schema metadata.
  - **Functionality**:
    - Queries the database for schemas, tables, columns, and relationships.
    - Caches schema data in `schema_cache/<db_name>/schema.json`.
    - Validates schema and table names.
    - Supports schema refreshes when metadata changes.
  - **Key Methods**:
    - `build_data_dict(connection)`: Builds the schema dictionary.
    - `load_from_cache()`: Loads cached schema.
    - `needs_refresh(connection)`: Checks if schema needs updating.
    - `get_table_metadata(schema, table)`: Retrieves metadata for a table.
  - **Dependencies**: `pyodbc`, `json`.

- **FeedbackManager** (`feedback/manager.py`):
  - **Purpose**: Manages user feedback to improve table identification.
  - **Functionality**:
    - Stores query-table mappings with embeddings and confidence scores.
    - Updates feedback counts for repeated queries.
    - Supports clearing feedback for reset scenarios.
  - **Key Methods**:
    - `store_feedback(query, tables, schema_dict)`: Saves feedback with embeddings.
    - `clear_feedback()`: Deletes all feedback entries.
    - `get_feedback()`: Retrieves feedback entries.
  - **Dependencies**: `CacheSynchronizer`, `sentence_transformers`.

### 3. Analysis Layer (`analysis/`)
Performs table identification, name matching, and query preprocessing.

- **TableIdentifier** (`analysis/table_identifier.py`):
  - **Purpose**: Identifies relevant tables for a given query.
  - **Functionality**:
    - Combines feedback, semantic similarity, synonyms, and name matches to score tables.
    - Validates identified tables against the schema.
    - Updates weights based on feedback.
  - **Key Methods**:
    - `process_query(query)`: Returns a list of tables and confidence score.
    - `update_weights_from_feedback(query, tables)`: Adjusts table weights.
    - `save_name_matches()`: Saves dynamic name matches.
  - **Dependencies**: `FeedbackManager`, `NameMatchManager`, `PatternManager`, `CacheSynchronizer`.

- **NameMatchManager** (`analysis/name_match_manager.py`):
  - **Purpose**: Generates and manages column name synonyms for query matching.
  - **Functionality**:
    - Computes similarity between query tokens and column names using embeddings.
    - Caches synonyms in `name_matches` table.
    - Supports dynamic synonym generation during query processing.
  - **Key Methods**:
    - `generate_matches(query_tokens)`: Creates synonyms for query tokens.
    - `save_dynamic_matches(matches)`: Saves new matches to cache.
  - **Dependencies**: `CacheSynchronizer`, `sentence_transformers`.

- **NLPPipeline** (`analysis/processor.py`):
  - **Purpose**: Preprocesses queries for analysis.
  - **Functionality**:
    - Tokenizes queries using `spacy`.
    - Removes stop words and normalizes tokens.
    - Extracts key terms for matching.
  - **Key Methods**:
    - `preprocess_query(query)`: Returns preprocessed tokens.
  - **Dependencies**: `spacy`.

- **PatternManager** (`config/patterns.py`):
  - **Purpose**: Manages patterns for query and schema matching.
  - **Functionality**:
    - Defines patterns for identifying entities (e.g., column names, table names) in queries.
    - Supports schema-specific patterns for filtering tables.
  - **Key Methods**:
    - `match_patterns(query, schema_dict)`: Matches query tokens to schema elements.
  - **Dependencies**: `SchemaManager`.

### 4. Core Processing Layer (`main.py`, `nlp/`)
Orchestrates the application flow and query processing.

- **DatabaseAnalyzer** (`main.py`):
  - **Purpose**: Main application controller.
  - **Functionality**:
    - Initializes all components and managers.
    - Handles configuration selection and database connection.
    - Processes queries and delegates to `QueryProcessor`.
    - Provides methods for schema validation, DDL generation, and feedback management.
    - Launches the CLI (`DatabaseAnalyzerCLI`).
  - **Key Methods**:
    - `run()`: Starts the application and CLI.
    - `connect_to_database()`: Establishes database connection.
    - `_initialize_managers()`: Initializes all managers.
    - `process_query(query)`: Processes a natural language query.
    - `generate_ddl(tables)`: Generates DDL for specified tables.
    - `validate_tables_exist(tables)`: Validates table names.
  - **Dependencies**: All other modules.

- **QueryProcessor** (`nlp/QueryProcessor.py`):
  - **Purpose**: Processes natural language queries to identify tables.
  - **Functionality**:
    - Coordinates preprocessing (`NLPPipeline`), table identification (`TableIdentifier`), and name matching (`NameMatchManager`).
    - Returns a list of tables and a confidence score.
  - **Key Methods**:
    - `process_query(query)`: Processes a query and returns tables and confidence.
  - **Dependencies**: `NLPPipeline`, `TableIdentifier`, `NameMatchManager`, `PatternManager`, `CacheSynchronizer`.

### 5. User Interface Layer (`cli/`)
Provides an interactive interface for users.

- **DatabaseAnalyzerCLI** (`cli/interface.py`):
  - **Purpose**: Implements the command-line interface.
  - **Functionality**:
    - Displays a menu for query processing, schema viewing, feedback management, and DDL generation.
    - Handles user input and displays results.
    - Supports commands like "back" and exit.
  - **Key Methods**:
    - `run()`: Starts the CLI loop.
    - `display_menu()`: Shows available options.
  - **Dependencies**: `DatabaseAnalyzer`, `FeedbackManager`, `SchemaManager`.

## Data Flow
The data flow describes how information moves through the components during key operations, such as application startup, query processing, and feedback storage.

### 1. Application Startup
1. **User Input** (`DatabaseAnalyzerCLI`):
   - User runs `main.py`, which instantiates `DatabaseAnalyzer`.
   - `DatabaseAnalyzer.run()` prompts for configuration selection (e.g., `BikeStores`).
2. **Configuration Loading** (`DBConfigManager`):
   - Loads `database_configurations.json` and validates schemas/tables.
   - Sets the selected configuration in `DatabaseAnalyzer.current_config`.
3. **Database Connection** (`DatabaseConnection`):
   - Connects to the database using `pyodbc` and config parameters.
4. **Manager Initialization** (`DatabaseAnalyzer._initialize_managers`):
   - Initializes `SchemaManager` to extract or load schema.
   - Runs `MetadataInitializer` to build caches (schema, weights, name matches, synonyms).
   - Initializes `CacheSynchronizer` to create `cache.db`.
   - Sets up `ModelSingleton` for the `SentenceTransformer` model.
   - Initializes `PatternManager`, `FeedbackManager`, `NLPPipeline`, `NameMatchManager`, `TableIdentifier`, and `QueryProcessor`.
5. **CLI Launch** (`DatabaseAnalyzerCLI`):
   - Starts the CLI loop, ready for user queries.

### 2. Query Processing
1. **User Query** (`DatabaseAnalyzerCLI`):
   - User enters a query (e.g., "how much staff works for all stores").
   - CLI passes the query to `DatabaseAnalyzer.process_query`.
2. **Preprocessing** (`QueryProcessor`, `NLPPipeline`):
   - `QueryProcessor` delegates to `NLPPipeline` to tokenize and normalize the query (e.g., `staff work store`).
3. **Table Identification** (`TableIdentifier`):
   - `TableIdentifier` retrieves feedback from `FeedbackManager` via `CacheSynchronizer`.
   - Matches query tokens to columns using `NameMatchManager` and synonyms.
   - Uses `PatternManager` to apply schema-specific patterns.
   - Combines feedback, semantic similarity (via `ModelSingleton`), and name matches to score tables.
   - Validates tables against `SchemaManager`.
   - Returns tables (e.g., `sales.staffs`, `sales.stores`) and confidence (e.g., 0.95).
4. **Result Display** (`DatabaseAnalyzerCLI`):
   - CLI displays the tables and confidence to the user.

### 3. Feedback Storage
1. **User Confirmation** (`DatabaseAnalyzerCLI`):
   - User confirms or modifies the identified tables.
   - CLI calls `DatabaseAnalyzer.confirm_tables` or `update_feedback`.
2. **Feedback Storage** (`FeedbackManager`):
   - `FeedbackManager` generates an embedding for the query using `ModelSingleton`.
   - Stores the query, tables, embedding, and count in `cache.db` via `CacheSynchronizer`.
3. **Weight Update** (`TableIdentifier`):
   - Updates table weights based on feedback to improve future identifications.
4. **Synonym Update** (`NameMatchManager`):
   - Caches new synonyms (e.g., `staff` for `staff_id`) in `name_matches`.

### 4. Shutdown
1. **User Exit** (`DatabaseAnalyzerCLI`):
   - User selects the exit option (e.g., option 9).
2. **Resource Cleanup** (`DatabaseAnalyzer._cleanup`):
   - `TableIdentifier` saves name matches.
   - `CacheSynchronizer` closes SQLite connections.
   - `DatabaseConnection` closes the database connection.

### Data Flow Diagram
```
User Query -> DatabaseAnalyzerCLI -> DatabaseAnalyzer -> QueryProcessor
   |                                                  |
   v                                                  v
NLPPipeline                                         TableIdentifier
   |                                                  |
   v                                                  v
PatternManager                                  FeedbackManager
   |                                                  |
   v                                                  v
SchemaManager                                    CacheSynchronizer
   |                                                  |
   v                                                  v
DatabaseConnection                               ModelSingleton
   |                                                  |
   v                                                  v
SQL Server Database                             SQLite cache.db
```

## Developer Guide
To collaborate on the Table Identifier Agent, follow these steps to set up, understand, and contribute to the project.

### Prerequisites
- **Python**: 3.8+.
- **Dependencies**:
  ```bash
  pip install sentence-transformers numpy sqlite3 pyodbc spacy
  python -m spacy download en_core_web_sm
  ```
- **Database**: SQL Server or compatible DBMS with ODBC driver (e.g., `ODBC Driver 17 for SQL Server`).
- **Git**: To clone the repository.

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/m-prasad-reddy/TIA-1.1.git
   cd TIA-1.1
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Database**:
   - Update `app-config/database_configurations.json` with your database details:
     ```json
     [
       {
         "name": "BikeStores",
         "driver": "ODBC Driver 17 for SQL Server",
         "server": "localhost",
         "database": "BikeStores",
         "schemas": ["hr", "production", "sales"],
         "tables": []
       }
     ]
     ```
4. **Run the Application**:
   ```bash
   python src/main.py
   ```

### Project Structure
```
TIA-1.1/
├── app-config/                 # Configuration files and cache
│   ├── database_configurations.json
│   ├── logging_config.ini
│   ├── BikeStores/            # Cache directory
│   │   ├── cache.db
│   │   ├── schema_cache/
├── logs/                      # Log files
│   ├── bikestores_app.log
├── src/                       # Source code
│   ├── main.py               # Main entry point
│   ├── cli/                  # CLI module
│   │   ├── interface.py
│   ├── config/               # Configuration modules
│   │   ├── manager.py
│   │   ├── model_singleton.py
│   │   ├── metadata_initializer.py
│   │   ├── cache_synchronizer.py
│   │   ├── patterns.py
│   ├── schema/               # Schema management
│   │   ├── manager.py
│   ├── feedback/             # Feedback management
│   │   ├── manager.py
│   ├── analysis/             # Analysis modules
│   │   ├── table_identifier.py
│   │   ├── name_match_manager.py
│   │   ├── processor.py
│   ├── nlp/                  # NLP modules
│   │   ├── QueryProcessor.py
├── requirements.txt           # Dependencies
```

### Contributing
1. **Fork and Branch**:
   - Fork the repository and create a feature branch:
     ```bash
     git checkout -b feature/your-feature
     ```
2. **Code Style**:
   - Follow PEP 8 guidelines.
   - Use descriptive variable names and docstrings.
   - Add logging for debugging (`logging.debug`, `logging.info`).
3. **Testing**:
   - Test changes with the `BikeStores` database.
   - Verify functionality using the CLI:
     - Process queries (e.g., "total sales amount at storename=Baldwin Bikes").
     - Check feedback (`SELECT * FROM feedback` in `cache.db`).
     - Generate DDL for tables.
   - Run with `cProfile` to profile performance:
     ```python
     # Uncomment cProfile code in main.py
     python src/main.py
     python -m pstats profile.out
     ```
4. **Logging**:
   - Use the existing logging setup (`logging_config.ini`).
   - Log key operations and errors to `logs/bikestores_app.log`.
5. **Pull Request**:
   - Submit a pull request with a clear description of changes.
   - Include tests and log outputs demonstrating functionality.

### Development Tips
- **Debugging**:
  - Check `logs/bikestores_app.log` for detailed execution traces.
  - Use `sqlite3 app-config/BikeStores/cache.db` to inspect `cache.db`.
- **Extending Features**:
  - Add new database drivers in `DatabaseConnection`.
  - Enhance `NLPPipeline` with additional NLP models (e.g., BERT).
  - Implement new CLI commands in `DatabaseAnalyzerCLI`.
- **Performance Optimization**:
  - Reuse SQLite connections in `CacheSynchronizer`.
  - Cache embeddings in `NameMatchManager`.
  - Optimize feedback deduplication in `FeedbackManager`.

### Known Issues and Improvements
- **Feedback Clearing**: The feedback table is cleared if it exceeds 360 entries, potentially losing valuable data. Consider increasing the threshold or implementing selective deduplication.
- **SQLite Connection Overhead**: Frequent connection creation in `CacheSynchronizer` may cause performance issues. Implement connection pooling.
- **Name Match Redundancy**: Repeated clearing and rewriting of `name_matches` can be optimized with incremental updates.
- **Concurrency**: SQLite write contention may occur in multi-user scenarios. Use transactions and write queues to mitigate.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or contributions, contact the project maintainer at [GitHub Issues](https://github.com/m-prasad-reddy/TIA-1.1/issues).

---

*Generated on May 12, 2025, for Table Identifier Agent-v1.1.*