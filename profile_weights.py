# profile_weights.py: Profile write_weights and write_name_matches operations

import cProfile
import pstats
import logging
import os
import threading
from config.cache_synchronizer import CacheSynchronizer

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "profile_weights.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("profile_weights")

def profile_operations():
    logger.info("Starting profile_operations")
    
    def run_with_timeout():
        try:
            db_name = "BikeStores"
            logger.debug("Initializing CacheSynchronizer")
            cache_synchronizer = CacheSynchronizer(db_name)
            logger.debug("CacheSynchronizer initialized")
            
            # Sample weights
            weights = {
                "hr.candidates": {
                    "id": 0.1,
                    "fullname": 0.05
                }
            }
            
            # Sample name matches
            name_matches = {
                "category_id": ["category id", "categoryid", "category_identifier"],
                "category_name": ["category name", "categoryname"]
            }
            
            logger.debug("Profiling write_weights")
            cProfile.runctx("cache_synchronizer.write_weights(weights, batch_size=10)", globals(), locals(), "write_weights_profile.out")
            logger.debug("Completed profiling write_weights")
            
            logger.debug("Profiling write_name_matches")
            cProfile.runctx("cache_synchronizer.write_name_matches(name_matches, batch_size=50)", globals(), locals(), "write_name_matches_profile.out")
            logger.debug("Completed profiling write_name_matches")
            
            logger.debug("Printing profiling results")
            p = pstats.Stats("write_weights_profile.out")
            logger.info("write_weights profile:")
            p.sort_stats("cumulative").print_stats(10)
            
            p = pstats.Stats("write_name_matches_profile.out")
            logger.info("write_name_matches profile:")
            p.sort_stats("cumulative").print_stats(10)
            logger.info("Completed profile_operations")
        except Exception as e:
            logger.error(f"Error during profiling: {e}")
    
    timeout = 120  # 2-minute timeout
    thread = threading.Thread(target=run_with_timeout)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        logger.error("Operation timed out")
        raise TimeoutError("Operation timed out")

if __name__ == "__main__":
    profile_operations()