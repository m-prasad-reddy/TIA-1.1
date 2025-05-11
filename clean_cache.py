import sqlite3
import logging
import os
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "cache_cleanup.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cache_cleanup")

def normalize_query(query: str) -> str:
    """Normalize query for consistent comparison."""
    import re
    query = re.sub(r'\s+', ' ', query.strip().lower())
    query = re.sub(r'[^\w\s]', '', query)
    return query

def deduplicate_feedback(db_path: str) -> int:
    """Remove duplicate feedback entries, retaining the most recent."""
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()
        
        # Find duplicate queries
        cursor.execute("""
            SELECT query, COUNT(*) as entry_count
            FROM feedback
            GROUP BY query
            HAVING entry_count > 1
        """)
        duplicate_queries = cursor.fetchall()
        removed_count = 0
        
        for query, count in duplicate_queries:
            logger.info(f"Found {count} duplicate entries for query: {query}")
            # Get all entries for this query
            cursor.execute("""
                SELECT timestamp, tables, embedding, count
                FROM feedback
                WHERE query = ?
                ORDER BY timestamp DESC
            """, (query,))
            entries = cursor.fetchall()
            
            # Keep the most recent entry
            most_recent = entries[0]
            recent_timestamp, recent_tables, recent_embedding, recent_count = most_recent
            total_count = sum(entry[3] for entry in entries)
            
            # Update the most recent entry
            cursor.execute("""
                UPDATE feedback
                SET count = ?
                WHERE query = ? AND timestamp = ?
            """, (total_count, query, recent_timestamp))
            
            # Delete older duplicates
            for entry in entries[1:]:
                old_timestamp = entry[0]
                cursor.execute("""
                    DELETE FROM feedback
                    WHERE query = ? AND timestamp = ?
                """, (query, old_timestamp))
                removed_count += 1
            
            logger.info(f"Retained entry for query '{query}' with timestamp {recent_timestamp}, total count={total_count}")
        
        conn.commit()
        logger.info(f"Removed {removed_count} duplicate feedback entries")
        return removed_count
    except Exception as e:
        logger.error(f"Error deduplicating feedback: {e}")
        raise
    finally:
        conn.close()

def ensure_index(db_path: str) -> bool:
    """Ensure the feedback.query index exists."""
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()
        
        # Check if index exists
        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='index' AND name='idx_feedback_query'
        """)
        if cursor.fetchone():
            logger.info("Index idx_feedback_query already exists")
        else:
            cursor.execute("""
                CREATE INDEX idx_feedback_query ON feedback(query)
            """)
            conn.commit()
            logger.info("Created index idx_feedback_query")
        return True
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        return False
    finally:
        conn.close()

def main():
    db_path = "app-config/BikeStores/cache.db"
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return
    
    logger.info("Starting cache cleanup")
    removed = deduplicate_feedback(db_path)
    logger.info(f"Completed deduplication, removed {removed} entries")
    
    logger.info("Checking/creating index")
    if ensure_index(db_path):
        logger.info("Index setup complete")
    else:
        logger.error("Failed to set up index")

if __name__ == "__main__":
    main()