import os
import time
import threading
import pandas as pd
import numpy as np
import pathway as pw
from dotenv import load_dotenv
import google.generativeai as genai

# --- CONFIGURATION ---
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

import warnings
warnings.filterwarnings("ignore")

if not GEMINI_KEY:
    print("❌ CRITICAL ERROR: GEMINI_API_KEY is missing.")
else:
    genai.configure(api_key=GEMINI_KEY)

# --- OPTIMIZED BATCH EMBEDDING ---
def get_batch_embeddings(texts, retries=3):
    """
    Embeds a list of texts in one API call (Much faster).
    """
    for i in range(retries):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=texts,
                task_type="retrieval_document"
            )
            # The API returns a dictionary, usually 'embedding' which is a list of lists
            return result['embedding']
        except Exception as e:
            # If batch fails (e.g. too large), wait and retry
            print(f"   ⚠️ Batch API Hiccup: {e}. Retrying...")
            time.sleep(2)
    # Fallback: Return zeros if complete fail
    return [[0.0] * 768 for _ in texts]

def verify_safe(backstory, evidence):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Determine if the Backstory contradicts the Novel Evidence.
        EVIDENCE: {evidence}
        BACKSTORY: {backstory}
        OUTPUT: Prediction: [1=Consistent, 0=Contradiction]. Rationale: [1 sentence].
        """
        res = model.generate_content(prompt)
        return res.text
    except:
        return "Prediction: 0\nRationale: API Error"

# --- MAIN PIPELINE ---
def run_hackathon_pipeline():
    print("🚀 Starting TURBO PIPELINE...")
    
    # 1. VERIFY DATA
    if not os.path.exists("data/") or not any(f.endswith(".txt") for f in os.listdir("data/")):
        print("❌ Error: No .txt files in data/ folder.")
        return

    # 2. PATHWAY INGESTION
    print("📚 Reading Novels via Pathway...")
    files_table = pw.io.fs.read("data/", format="plaintext", mode="static", with_metadata=True)
    
    temp_csv = "data/temp_novels_dump.csv"
    if os.path.exists(temp_csv): os.remove(temp_csv)
    
    pw.io.csv.write(files_table, temp_csv)
    
    def start_engine():
        try: pw.run()
        except: pass
    t = threading.Thread(target=start_engine, daemon=True)
    t.start()
    
    # 3. SMART WAIT
    print("⏳ Processing files...", end="")
    for i in range(20):
        if os.path.exists(temp_csv) and os.path.getsize(temp_csv) > 100:
            print(" Done!")
            break # Exit immediately when file is ready
        time.sleep(1)
        print(".", end="")
    
    # 4. LOAD & CHUNK
    try:
        df_novels = pd.read_csv(temp_csv)
    except:
        print("\n❌ Failed to read Pathway output.")
        return

    chunks = []
    chunk_size = 500
    for _, row in df_novels.iterrows():
        words = str(row['data']).split()
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i+chunk_size]))
            
    print(f"✂️  Generated {len(chunks)} chunks.")

    # 5. BATCH EMBEDDING (The Speed Fix)
    print("⚡ Computing Embeddings (Batch Mode)...")
    novel_vectors = []
    batch_size = 20 # Safe batch size for Gemini
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"   Batch {i // batch_size + 1}/{(len(chunks)//batch_size)+1}...", end="\r")
        
        # Get 20 embeddings at once
        batch_vecs = get_batch_embeddings(batch)
        novel_vectors.extend(batch_vecs)
        time.sleep(0.5) # Slight cool-down for rate limits

    novel_matrix = np.array(novel_vectors)
    print(f"\n✅ Index Ready. Matrix Shape: {novel_matrix.shape}")

    # 6. PROCESS QUERIES
    print("\n❓ Processing Queries...")
    df_queries = pd.read_csv("data/test.csv")
    results = []
    
    for idx, row in df_queries.iterrows():
        try:
            q_vec = np.array(get_batch_embeddings([row['content']])[0])
            scores = np.dot(novel_matrix, q_vec)
            top_5 = np.argsort(scores)[-5:][::-1]
            evidence = "\n".join([chunks[i] for i in top_5])
            
            verdict = verify_safe(row['content'], evidence)
            
            pred = 1 if "1" in verdict[:20] else 0
            rationale = verdict.replace("\n", " ").split("Rationale:")[-1].strip()
            
            print(f"[{idx+1}/{len(df_queries)}] {row['char']}: {pred}")
            results.append({"id": row['id'], "Prediction": pred, "Rationale": rationale})
            
        except Exception as e:
            print(f"⚠️ Error: {e}")

    pd.DataFrame(results).to_csv("results.csv", index=False)
    print("\n🎉 SUCCESS! 'results.csv' created.")

if __name__ == "__main__":
    run_hackathon_pipeline()