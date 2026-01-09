import os
import time
import threading
import pandas as pd
import numpy as np
import pathway as pw
from dotenv import load_dotenv
import google.generativeai as genai

# --- CONFIG ---
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    print("CRITICAL: GEMINI_API_KEY is missing in .env")
    exit()

genai.configure(api_key=GEMINI_KEY)

# --- EMBEDDING FUNCTION ---
def get_embeddings_batched(texts):
    """Generates embeddings for a list of text chunks."""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Batch Error: {e}. Retrying in 10s...")
        time.sleep(10)
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=texts,
                task_type="retrieval_document"
            )
            return result['embedding']
        except:
            return [[0.0] * 768 for _ in texts]

def build_knowledge_base():
    print("STARTING DATABASE CREATION...")
    
    # 1. SETUP PATHS
    if not os.path.exists("data"): os.makedirs("data")
    matrix_file = "data/novel_matrix.npy"
    chunks_file = "data/novel_chunks.csv"
    dump_file = "data/temp_dump.csv"

    # 2. READ NOVELS WITH PATHWAY
    print("Reading Novels via Pathway...")
    if os.path.exists(dump_file): os.remove(dump_file)

    # Pathway FS Read
    files = pw.io.fs.read("data/", format="plaintext", mode="static")
    pw.io.csv.write(files, dump_file)
    
    # Run Engine
    def engine():
        try: pw.run()
        except: pass
    t = threading.Thread(target=engine, daemon=True)
    t.start()
    
    # Wait for file
    print("Extracting text...", end="")
    for i in range(20):
        if os.path.exists(dump_file) and os.path.getsize(dump_file) > 1000:
            break
        time.sleep(1)
        print(".", end="")
    print(" Done.")

    # 3. PROCESS TEXT
    try:
        df_data = pd.read_csv(dump_file)
        # GLUE: Combine all lines into one big string
        full_text = " ".join(df_data['data'].astype(str).tolist())
        
        # CHUNK: Split into 500-word blocks
        all_words = full_text.split()
        proper_chunks = []
        chunk_size = 500
        
        for i in range(0, len(all_words), chunk_size):
            chunk_text = " ".join(all_words[i:i+chunk_size])
            proper_chunks.append(chunk_text)
            
        print(f"✅ Created {len(proper_chunks)} clean chunks.")
        
    except Exception as e:
        print(f"Error processing text: {e}")
        return

    # 4. GENERATE EMBEDDINGS
    print("Generating Embeddings (Batch Mode)...")
    novel_vectors = []
    batch_size = 20
    
    total_batches = (len(proper_chunks) // batch_size) + 1
    
    for i in range(0, len(proper_chunks), batch_size):
        batch = proper_chunks[i : i + batch_size]
        print(f"   Batch {i//batch_size + 1}/{total_batches}...", end="\r")
        
        vecs = get_embeddings_batched(batch)
        novel_vectors.extend(vecs)
        time.sleep(2) # Rate limit safety

    # 5. SAVE TO DISK
    novel_matrix = np.array(novel_vectors)
    np.save(matrix_file, novel_matrix)
    pd.DataFrame({'chunk': proper_chunks}).to_csv(chunks_file, index=False)
    
    print(f"\n SUCCESS! Database created.")
    print(f"   - Embeddings: {matrix_file}")
    print(f"   - Text Chunks: {chunks_file}")

if __name__ == "__main__":
    build_knowledge_base()