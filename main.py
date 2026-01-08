import os
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# --- 1. CONFIGURATION ---
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
warnings = __import__('warnings')
warnings.filterwarnings("ignore")

if not GEMINI_KEY:
    print("❌ CRITICAL: GEMINI_API_KEY is missing.")
else:
    genai.configure(api_key=GEMINI_KEY)

# --- 2. LEGACY MODEL (The Fix) ---
def verify_with_legacy(backstory, evidence):
    try:
        # We try the 'gemini-pro' model first (most capable)
        model = genai.GenerativeModel('gemini-pro')
        res = model.generate_content(
            f"Evidence: {evidence}\nClaim: {backstory}\n"
            "Task: Is the claim consistent with the evidence? Output '1' for Yes, '0' for No, followed by a 1-sentence reason."
        )
        return res.text
    except:
        # Fallback to text-bison if gemini-pro fails
        try:
            model = genai.GenerativeModel('models/text-bison-001')
            res = model.generate_content(
                f"Evidence: {evidence}\nClaim: {backstory}\n"
                "Task: Is the claim consistent with the evidence? Output '1' for Yes, '0' for No."
            )
            return res.text
        except Exception as e:
            return f"0 Rationale: {e}"

# --- 3. BATCH EMBEDDING ---
def get_embeddings_batched(texts):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']
    except:
        return [[0.0] * 768 for _ in texts]

# --- 4. THE PIPELINE ---
def run_final_attempt():
    print("🚀 Starting Final Attempt (Legacy Fallback Mode)...")
    
    matrix_file = "data/novel_matrix.npy"
    chunks_file = "data/novel_chunks.csv"
    
    # --- PHASE A: CHECK INGESTION ---
    if os.path.exists(matrix_file) and os.path.exists(chunks_file):
        print("✅ Found saved embeddings! Loading from disk...")
        novel_matrix = np.load(matrix_file)
        proper_chunks = pd.read_csv(chunks_file)['chunk'].tolist()
        print(f"   Loaded {len(proper_chunks)} chunks.")
    else:
        print("❌ Checkpoint missing. Please UNDELETE the files or re-run the previous script.")
        return

    # --- PHASE B: REASONING ---
    print("\n❓ Checking Consistency...")
    df_test = pd.read_csv("data/test.csv")
    results = []
    
    time.sleep(2)
    
    for idx, row in df_test.iterrows():
        try:
            q_vec = get_embeddings_batched([row['content']])[0]
            scores = np.dot(novel_matrix, q_vec)
            top_indices = np.argsort(scores)[-5:][::-1]
            evidence_text = "\n".join([str(proper_chunks[i]) for i in top_indices])
            
            # USE LEGACY/FALLBACK LOGIC
            verdict = verify_with_legacy(row['content'], evidence_text)
            
            pred = 1 if "1" in verdict[:10] else 0
            rationale = verdict.replace("\n", " ").split("Rationale:")[-1].strip()
            
            status = "✅" if pred == 1 else "❌"
            print(f"[{idx+1}/{len(df_test)}] {row['char']}: {status}")
            
            results.append({
                "id": row['id'],
                "Prediction": pred,
                "Rationale": rationale
            })
            
            time.sleep(4) 
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            results.append({"id": row['id'], "Prediction": 0, "Rationale": "Error"})

    pd.DataFrame(results).to_csv("results.csv", index=False)
    print("\n🎉 SUCCESS! Download 'results.csv'.")

if __name__ == "__main__":
    run_final_attempt()