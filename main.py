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

# Suppress the "FutureWarning"
import warnings
warnings.filterwarnings("ignore")

if not GEMINI_KEY:
    print("❌ CRITICAL ERROR: GEMINI_API_KEY is missing.")
else:
    genai.configure(api_key=GEMINI_KEY)

# --- HELPER FUNCTIONS ---
def get_embedding_safe(text, retries=3):
    """Robust embedding with retries."""
    for i in range(retries):
        try:
            # Using the specialized embedding model
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            time.sleep(1)
    # Return zero vector if all retries fail to prevent crash
    return [0.0] * 768

def verify_safe(backstory, evidence):
    """Gemini reasoning with error handling."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Determine if the Backstory contradicts the Novel Evidence.
        
        EVIDENCE:
        {evidence}
        
        BACKSTORY:
        {backstory}
        
        OUTPUT FORMAT:
        Prediction: [1 for Consistent, 0 for Contradiction]
        Rationale: [1 sentence reason]
        """
        res = model.generate_content(prompt)
        return res.text
    except:
        return "Prediction: 0\nRationale: API Error"

# --- MAIN PIPELINE ---
def run_hackathon_pipeline():
    print("🚀 Starting Time-Bounded Pipeline...")
    
    # 1. VERIFY DATA EXISTS
    if not os.path.exists("data/"):
        print("❌ Error: 'data/' folder not found.")
        return
    
    txt_files = [f for f in os.listdir("data/") if f.endswith(".txt")]
    if not txt_files:
        print("❌ Error: No .txt files found in data/. Please upload the novels.")
        return
    print(f"   Found novels: {txt_files}")

    # 2. PATHWAY INGESTION (The Trick: Dump to CSV)
    print("📚 Ingesting novels via Pathway...")
    
    # Define the graph
    # We read from the directory. mode="static" means read once.
    files_table = pw.io.fs.read(
        "data/", 
        format="plaintext", 
        mode="static", 
        with_metadata=True
    )
    
    # We write to a temp file immediately
    temp_csv = "data/temp_novels_dump.csv"
    if os.path.exists(temp_csv):
        os.remove(temp_csv) # Clean up previous run
        
    pw.io.csv.write(files_table, temp_csv)
    
    # Run Pathway in a Daemon Thread
    def start_engine():
        try:
            pw.run()
        except:
            pass
            
    t = threading.Thread(target=start_engine, daemon=True)
    t.start()
    
    # 3. THE COUNTDOWN (Force proceed after 15s)
    print("⏳ Waiting 15 seconds for Pathway to process files...")
    for i in range(15, 0, -1):
        if os.path.exists(temp_csv) and os.path.getsize(temp_csv) > 100:
            # If file exists and has content, we might be done early, 
            # but let's wait a bit to ensure full write.
            pass 
        print(f"   {i}...", end="\r")
        time.sleep(1)
        
    print("\n✅ Time's up. Reading extracted data...")
    
    # 4. READ DUMPED DATA (Python Mode)
    try:
        if not os.path.exists(temp_csv):
            print("❌ Pathway failed to create the CSV file. Check file permissions.")
            return
            
        df_novels = pd.read_csv(temp_csv)
        print(f"   Successfully loaded {len(df_novels)} rows/files from Pathway output.")
    except Exception as e:
        print(f"❌ Error reading Pathway output: {e}")
        return

    # 5. CHUNK & EMBED (Python Loop - Visible Progress)
    print("\n✂️ Chunking and Embedding...")
    
    chunks = []
    chunk_size = 500 # words
    
    # Iterate through the novels we loaded
    for _, row in df_novels.iterrows():
        text = str(row['data']) # 'data' column from Pathway
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])
            chunks.append(chunk_text)
            
    print(f"   Generated {len(chunks)} text chunks.")
    
    # Create Matrix
    novel_vectors = []
    print("   Computing Embeddings (this is the slow part)...")
    for idx, chunk in enumerate(chunks):
        if idx % 5 == 0:
            print(f"   Encoded {idx}/{len(chunks)} chunks...", end="\r")
        vec = get_embedding_safe(chunk)
        novel_vectors.append(vec)
        
    novel_matrix = np.array(novel_vectors)
    print(f"\n✅ Index Ready. Matrix Shape: {novel_matrix.shape}")

    # 6. PROCESS QUERIES
    print("\n❓ Processing Test Cases...")
    df_queries = pd.read_csv("data/test.csv")
    results = []
    
    for idx, row in df_queries.iterrows():
        try:
            q_id = row['id']
            char = row['char']
            backstory = row['content']
            
            print(f"[{idx+1}/{len(df_queries)}] Checking {char} (ID: {q_id})...")
            
            # Vector Search
            q_vec = np.array(get_embedding_safe(backstory))
            scores = np.dot(novel_matrix, q_vec)
            top_5_idx = np.argsort(scores)[-5:][::-1]
            
            evidence = "\n\n".join([chunks[i] for i in top_5_idx])
            
            # Reasoning
            verdict = verify_safe(backstory, evidence)
            
            # Logic parsing
            pred = 1 if "1" in verdict[:20] else 0
            rationale = verdict.replace("\n", " ").strip()
            if "Rationale:" in rationale:
                rationale = rationale.split("Rationale:")[-1].strip()
                
            results.append({
                "id": q_id,
                "Prediction": pred,
                "Rationale": rationale
            })
            
            time.sleep(2) # Be nice to the API
            
        except Exception as e:
            print(f"⚠️ Skipped row {idx}: {e}")

    # 7. SAVE RESULTS
    pd.DataFrame(results).to_csv("results.csv", index=False)
    print("\n🎉 SUCCESS! 'results.csv' created.")

if __name__ == "__main__":
    run_hackathon_pipeline()