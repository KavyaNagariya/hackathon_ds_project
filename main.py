import os
import pathway as pw
from src.ingestion import load_novels

# Configuration
DATA_DIR = "data/"

def run_pipeline():
    print(f"🚀 Starting Pipeline... Reading from {DATA_DIR}")
    
    # 1. Ingestion Phase
    # This creates the computational graph but doesn't run it yet.
    raw_data_table = load_novels(DATA_DIR)
    
    # 2. Preview (Debugging)
    # Since Pathway is lazy (like Spark), we force it to compute specific rows to verify.
    print("👀 Previewing Ingested Data:")
    pw.debug.compute_and_print(raw_data_table, limit=3)
    
    print("✅ Pipeline graph built successfully.")

if __name__ == "__main__":
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"⚠️ Created {DATA_DIR}. Please put your .txt files there!")
    else:
        run_pipeline()