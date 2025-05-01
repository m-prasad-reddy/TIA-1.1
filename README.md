# Table Identification Agent (TIA)

The **Table Identification Agent (TIA)** is a Python-based command-line interface (CLI) application designed to analyze natural language queries and identify relevant database tables or entities across various database systems. It supports relational databases (e.g., Microsoft SQL Server, MY SQL,etc.,), NoSQL databases, Data Warehouses, Data Lakes, and other database applications. TIA leverages natural language processing (NLP), semantic similarity, regex patterns, and user feedback to map queries to schema elements, enhancing database interaction for developers and analysts.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
  - [Core Modules](#core-modules)
  - [Supporting Modules](#supporting-modules)
  - [Data Flow](#data-flow)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Natural Language Query Processing**: Maps queries (e.g., "Show me product stock availability at all stores") to database tables or entities using NLP, pattern matching, and semantic analysis.
- **Schema-Agnostic Design**: Supports relational databases, NoSQL, Data Warehouses, and Data Lakes by dynamically adapting to provided schema metadata.
- **Feedback System**: Allows users to confirm or correct suggestions, storing feedback to refine future predictions.
- **Cache Management**: Persists weights, name matches, feedback, and ignored queries in a SQLite database for performance optimization.
- **Query Validation**: Ensures queries are relevant, in English, and meet minimum length requirements (>=3 characters).
- **Synonym Expansion**: Enhances query matching using a configurable synonym file (`query_synonyms.json`).
- **Logging**: Detailed logs (`tia_app.log`) for debugging and monitoring query processing.

## Architecture
TIA’s modular architecture separates concerns into **Core Modules** (handling primary query processing and user interaction) and **Supporting Modules** (providing utilities like caching, schema management, and NLP). Built with Python 3.8+, TIA integrates libraries such as `spacy`, `sentence-transformers`, `langdetect`, and `sqlite3` to support diverse database environments.

### High-Level Architecture Diagram
```
+-------------------+       +-------------------+       +-------------------+
| User (CLI)        |<----->| DatabaseAnalyzerCLI|<----->| Database (SQL,    |
| (interface.py)    |       | (interface.py)    |       | NoSQL, DW, DL)    |
+-------------------+       +-------------------+       +-------------------+
                                   |
                                   v
+-------------------+       +-------------------+       +-------------------+
| TableIdentifier   |<----->| CacheSynchronizer |<----->| SQLite Cache      |
| (table_identifier)|       | (cache_synchronizer)|     | (cache.db)        |
+-------------------+       +-------------------+       +-------------------+
                                   |
                                   v
+-------------------+       +-------------------+
| NameMatchManager  |<----->| FeedbackManager   |
| (name_match_manager)|     | (manager.py)      |
+-------------------+       +-------------------+
                                   |
                                   v
+-------------------+       +-------------------+
| PatternManager    |       | SchemaManager     |
| (patterns.py)     |       | (manager.py)      |
+-------------------+       +-------------------+
```

### Core Modules
These modules form the backbone of TIA’s query processing and user interaction:

1. **DatabaseAnalyzerCLI** (`cli/interface.py`, ~589 lines)
   - **Functionality**:
     - Provides a CLI for user interaction, including menu options for database connection, query mode, feedback management, schema viewing, and query history.
     - Validates queries for length, language (English, with schema-based fallback), and schema relevance.
     - Coordinates query expansion, table identification, and feedback collection.
     - Manages query history and cleanup of database connections.
   - **Key Methods**:
     - `run()`: Displays the main menu and handles user input.
     - `_query_mode()`: Processes natural language queries and collects feedback.
     - `_validate_query()`: Checks query validity using `langdetect` and schema terms.
     - `_connect_to_database()`: Establishes database connections via configuration.

2. **TableIdentifier** (`analysis/table_identifier.py`, ~208 lines)
   - **Functionality**:
     - Maps natural language queries to database tables or entities using multiple strategies:
       - Regex pattern matching (via `PatternManager`).
       - Semantic similarity (using `sentence-transformers` embeddings).
       - Weight-based matching (from cached weights).
       - Name matches (synonyms from `NameMatchManager`).
       - Entity recognition (via `spacy` for entities like ORG, PRODUCT).
       - Custom rules (e.g., mapping “stock” to `production.stocks` in BikeStores).
     - Updates weights and name matches based on user feedback to improve future predictions.
   - **Key Methods**:
     - `identify_tables()`: Returns a list of tables and confidence score for a query.
     - `update_weights_from_feedback()`: Adjusts table-column weights based on user input.
     - `update_name_matches()`: Adds synonyms for columns (e.g., “stock” for “quantity”).

3. **CacheSynchronizer** (`config/cache_synchronizer.py`, ~512 lines)
   - **Functionality**:
     - Manages a SQLite database (`cache.db`) for persistent storage of:
       - `weights`: Table-column weights for ranking relevance.
       - `name_matches`: Column-synonym mappings (e.g., “quantity” → “stock”).
       - `feedback`: User-confirmed query-table mappings.
       - `ignored_queries`: Queries rejected as invalid or irrelevant.
     - Converts NLP embeddings (`np.ndarray`) to/from SQLite BLOBs for similarity-based retrieval.
     - Supports cache validation, clearing, and migration from file-based caches.
   - **Key Methods**:
     - `write_ignored_query()`: Stores rejected queries with reasons.
     - `write_feedback()`: Saves user feedback with query embeddings.
     - `load_weights()`: Retrieves weights for table identification.
     - `load_name_matches()`: Loads synonym mappings.

### Supporting Modules
These modules provide utilities and enhance the core functionality:

1. **NameMatchManager** (`analysis/name_match_manager.py`, ~225 lines)
   - **Functionality**:
     - Scores query tokens against schema columns using cosine similarity of embeddings.
     - Supports synonym-based matching by leveraging `name_matches` from `CacheSynchronizer`.
     - Generates column scores (e.g., `schema.table.column: score`) to guide table identification.
   - **Key Methods**:
     - `process_query()`: Computes similarity scores for query tokens and columns.

2. **FeedbackManager** (`feedback/manager.py`)
   - **Functionality**:
     - Stores and retrieves user feedback on query-table mappings.
     - Tracks feedback counts to prioritize frequently confirmed mappings.
     - Provides top queries for example suggestions in the CLI.
   - **Key Methods**:
     - `store_feedback()`: Saves query-table feedback.
     - `get_top_queries()`: Returns frequently used queries.
     - `clear_feedback()`: Resets feedback data.

3. **PatternManager** (`config/patterns.py`)
   - **Functionality**:
     - Defines regex patterns for table matching (e.g., `\bproducts\b` → `production.products` in BikeStores).
     - Allows dynamic pattern updates based on schema metadata.
   - **Key Methods**:
     - `get_patterns()`: Returns a dictionary of patterns to tables.

4. **SchemaManager** (`schema/manager.py`)
   - **Functionality**:
     - Builds schema dictionaries from database metadata, including tables, columns, primary keys, and foreign keys.
     - Supports relational and non-relational schemas by abstracting metadata extraction.
   - **Key Methods**:
     - `build_schema_dictionary()`: Creates a schema dictionary for use by other modules.

5. **ModelSingleton** (`config/model_singleton.py`)
   - **Functionality**:
     - Manages a single instance of the `sentence-transformers` model for embedding generation.
     - Ensures efficient memory usage across modules.
   - **Key Methods**:
     - Initializes and provides access to the NLP model.

6. **DatabaseConnection** (`config/manager.py`)
   - **Functionality**:
     - Handles database connections for relational (e.g., SQL Server) and other database types.
     - Supports connection string-based authentication and disconnection.
   - **Key Methods**:
     - `connect()`: Establishes a database connection.
     - `disconnect()`: Closes the connection.

### Data Flow
The data flow through TIA’s modules for processing a query (e.g., "Show me product stock availability at all stores") is as follows:

1. **User Input (DatabaseAnalyzerCLI)**:
   - The user enters a query via the CLI in `interface.py`’s `_query_mode`.
   - Example: User selects Query Mode and inputs the query.

2. **Query Validation (DatabaseAnalyzerCLI)**:
   - `interface.py`’s `_validate_query` checks:
     - Length (>=3 characters).
     - Language (English via `langdetect`, with fallback if schema terms like “products” are present).
     - Relevance (contains table/column names from `SchemaManager`’s schema dictionary).
     - Not in `ignored_queries` (checked via `CacheSynchronizer`).
   - If invalid, the query is rejected and stored in `CacheSynchronizer`’s `ignored_queries` table with a reason (e.g., “non_english_lang_id”).

3. **Query Expansion (DatabaseAnalyzerCLI)**:
   - `interface.py`’s `_expand_query_with_synonyms` uses `query_synonyms.json` to expand terms (e.g., “stock” → “quantity”).
   - The expanded query is passed to the next step.

4. **Column Scoring (NameMatchManager)**:
   - `name_match_manager.py`’s `process_query`:
     - Tokenizes the query (e.g., “product”, “stock”, “availability”, “stores”).
     - Computes cosine similarity between tokens and schema columns using `sentence-transformers` embeddings (via `ModelSingleton`).
     - Retrieves synonyms from `CacheSynchronizer`’s `name_matches` table.
     - Returns a dictionary of column scores (e.g., `production.stocks.quantity: 0.8`).

5. **Table Identification (TableIdentifier)**:
   - `table_identifier.py`’s `identify_tables` processes the query and column scores:
     - Applies regex patterns from `PatternManager` (e.g., `\bstores\b` → `sales.stores`).
     - Computes semantic similarity between the query and table names using `sentence-transformers`.
     - Uses weights and name matches from `CacheSynchronizer`’s `weights` and `name_matches` tables.
     - Performs entity recognition with `spacy` to detect entities (e.g., “stores” as ORG).
     - Applies custom rules (e.g., “stock” → `production.stocks` for BikeStores).
   - Outputs a list of tables (e.g., `['production.stocks', 'production.products', 'sales.stores']`) and a confidence score.

6. **User Feedback (DatabaseAnalyzerCLI, FeedbackManager)**:
   - `interface.py` displays the identified tables and prompts for confirmation (‘y’/’n’).
   - If confirmed (‘y’):
     - `FeedbackManager.store_feedback` saves the query-tables mapping to `CacheSynchronizer`’s `feedback` table.
     - `TableIdentifier.update_weights_from_feedback` increments weights in `CacheSynchronizer`’s `weights` table.
     - `TableIdentifier.save_name_matches` updates synonyms in `CacheSynchronizer`’s `name_matches` table.
   - If rejected (‘n’):
     - User provides correct tables (e.g., `production.stocks,production.products,sales.stores`) or skips.
     - Correct tables are validated (`TableIdentifier.validate_tables`) and saved via `FeedbackManager`.
     - The original query may be added to `CacheSynchronizer`’s `ignored_queries` with a reason (e.g., “user_rejected”).

7. **Cache Persistence (CacheSynchronizer)**:
   - `CacheSynchronizer` updates `cache.db` with:
     - Updated weights (`weights` table).
     - New or modified synonyms (`name_matches` table).
     - Feedback entries (`feedback` table).
     - Ignored queries (`ignored_queries` table).
   - Embeddings are stored as BLOBs for similarity-based retrieval in future queries.

8. **Output and Logging (DatabaseAnalyzerCLI)**:
   - The user sees the identified tables, confidence score, and feedback prompt.
   - Logs are written to `logs/tia_app.log` with details like “Schema match: table 'products' in query” or “Identified tables: [...]”.

## Installation
### Prerequisites
- Python 3.8+
- Database access (e.g., Microsoft SQL Server for BikeStores, MongoDB for NoSQL, or Data Warehouse/Data Lake systems).
- SQLite (included with Python).

### Dependencies
Install required Python packages:
```bash
pip install spacy sentence-transformers torch pyodbc numpy langdetect
python -m spacy download en_core_web_sm
```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/m-prasad-reddy/TIA-1.1.git
   cd TableIdentifier-v2
   ```

2. Configure the database:
   - Update `app-config/database_configurations.json` with your database connection details. Example for BikeStores (SQL Server):
     ```json
     [
         {
             "name": "BIKES_DB",
             "connection_string": "DRIVER={SQL Server};SERVER=your_server;DATABASE=BikeStores;Trusted_Connection=yes;"
         }
     ]
     ```
   - For NoSQL, Data Warehouses, or Data Lakes, provide appropriate connection strings or API endpoints.

3. (Optional) Add query synonyms:
   - Create `app-config/<db_name>/query_synonyms.json` (e.g., `app-config/BikeStores/query_synonyms.json`):
     ```json
     {
         "synonyms": {
             "stock": ["quantity", "inventory", "availability"],
             "availability": ["quantity", "stock"],
             "store": ["stores", "shop"],
             "product": ["products", "item"]
         }
     }
     ```

4. Ensure directories exist:
   ```bash
   mkdir -p app-config/<db_name> logs
   ```

## Usage
1. Run the application:
   ```bash
   python main.py
   ```

2. Select options from the CLI menu:
   ```
   === Table Identifier Agent ===
   Main Menu:
   1. Connect to Database
   2. Query Mode
   3. Reload Configurations
   4. Manage Feedback
   5. Manage Ignored Queries
   6. View Schema
   7. View Query History
   8. Exit
   ```

3. Example Workflow (using BikeStores):
   - **Connect to Database**: Select option 1, choose `BIKES_DB`.
   - **Query Mode**: Select option 2, enter a query:
     ```
     Enter query (or 'back'): Show me product stock availability at all stores
     Identified Tables: production.stocks, production.products, sales.stores
     Confidence: High
     Are these tables correct? (y/n): y
     ```
   - **View Schema**: Select option 6 to inspect schema elements (tables, columns, etc.).
   - **Manage Feedback**: Select option 4 to list or add feedback.
   - **Manage Ignored Queries**: Select option 5 to review rejected queries.

## Configuration
- **Database Configuration**: `app-config/database_configurations.json` defines database connections (e.g., SQL Server, MongoDB).
- **Logging**: Configured via `app-config/logging_config.ini` or defaults to `logs/tia_app.log`.
- **Cache**: Stored in `app-config/<db_name>/cache.db` (SQLite).
- **Synonyms**: Optional `app-config/<db_name>/query_synonyms.json` for query expansion.

## Logging
- Logs are written to `logs/tia_app.log`.
- Includes debugging info, errors, and query processing details (e.g., “Identified tables”, “Wrote ignored query”).
- Example log entry:
  ```
  2025-05-01 22:00:08,874 - table_identifier - DEBUG - Identified tables: ['production.stocks', 'production.products', 'sales.stores'], confidence=0.85
  ```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please include:
- Detailed description of changes.
- Tests or examples demonstrating the feature (e.g., new database type support).
- Updates to this README for new modules or configurations.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.