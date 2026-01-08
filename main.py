import os
import time
import pandas as pd
import numpy as np
import requests
import json
from dotenv import load_dotenv
import google.generativeai as genai

# --- CONFIG ---
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_KEY)

# --- AUTO-DISCOVERY (Finds the working URL) ---
def get_working_model_url():
    print("🔍 Discovering available models...")
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        response = requests.get(f"{base_url}?key={GEMINI_KEY}")
        models = response.json().get('models', [])
        for m in models:
            if "generateContent" in m.get("supportedGenerationMethods", []):
                # Prefer Flash or Pro if available
                if "flash" in m['name'] or "pro" in m['name']:
                    print(f"✅ Selected Model: {m['name']}")
                    return f"https://generativelanguage.googleapis.com/v1beta/{m['name']}:generateContent?key={GEMINI_KEY}"
        
        # Fallback to whatever is first
        if models:
            m = models[0]
            print(f"⚠️ Fallback Model: {m['name']}")
            return f"https://generativelanguage.googleapis.com/v1beta/{m['name']}:generateContent?key={GEMINI_KEY}"
            
    except Exception as e:
        print(f"❌ Discovery Failed: {e}")
    return None

VALID_MODEL_URL = get_working_model_url()

# --- REASONING WITH RETRY (Fixes 503 Errors) ---
def verify_claim(backstory, evidence):
    if not VALID_MODEL_URL: return "0 Rationale: No model found."
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Evidence: {evidence}\n\nClaim: {backstory}\n\nTask: Is the claim consistent? Output '1' if Yes, '0' if No. Start with the number."
            }]
        }]
    }
    
    # RETRY LOOP (Crucial for 503 errors)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(VALID_MODEL_URL, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 200:
                try:
                    return response.json()['candidates'][0]['content']['parts'][0]['text']
                except:
                    return "0 Rationale: Blocked by Safety Filter"
            
            elif response.status_code == 503:
                print(f"   ⚠️ Server Busy (503). Retrying {attempt+1}/{max_retries}...", end="\r")
                time.sleep(5) # Wait longer for 503
            
            elif response.status_code == 429:
                print(f"   ⚠️ Rate Limit (429). Cooling down...", end="\r")
                time.sleep(10)
            else:
                return f"0 Rationale: HTTP Error {response.status_code}"
                
        except Exception as e:
            time.sleep(2)
            
    return "0 Rationale: API Failed after retries"

# --- EMBEDDING HELPER ---
def get_query_embedding(text):
    try:
        res = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")
        return res['embedding']
    except:
        return [0.0] * 768

# --- MAIN EXECUTION ---
def run_main():
    print("🚀 STARTING REASONING PIPELINE...")
    
    # 1. LOAD DATABASE
    if not os.path.exists("data/novel_matrix.npy"):
        print("❌ Database missing! Please run 'create_database.py' first.")
        return
        
    print("📂 Loading Knowledge Base...", end="")
    novel_matrix = np.load("data/novel_matrix.npy")
    chunks_df = pd.read_csv("data/novel_chunks.csv")
    proper_chunks = chunks_df['chunk'].tolist()
    print(f" Loaded {len(proper_chunks)} chunks.")

    # 2. PROCESS TEST SET
    df_test = pd.read_csv("data/test.csv")
    results = []
    
    print(f"❓ Processing {len(df_test)} queries...")
    
    for idx, row in df_test.iterrows():
        try:
            # A. Search
            q_vec = np.array(get_query_embedding(row['content']))
            scores = np.dot(novel_matrix, q_vec)
            top_indices = np.argsort(scores)[-5:][::-1]
            evidence = "\n".join([str(proper_chunks[i]) for i in top_indices])
            
            # B. Reason
            verdict = verify_claim(row['content'], evidence)
            
            # C. Parse Cleanly
            pred = 1 if verdict.strip().startswith("1") else 0
            # Remove the number from rationale
            rationale = verdict.replace("1", "").replace("0", "").strip()
            # Clean up leading punctuation like ", " or ": "
            if rationale.startswith(",") or rationale.startswith(":"):
                rationale = rationale[1:].strip()
                
            # Fallback if rationale is empty
            if not rationale: rationale = "inconsistent with narrative context"

            print(f"[{idx+1}/{len(df_test)}] ID {row['id']}: {pred}")
            
            results.append({
                "id": row['id'],
                "Prediction": pred,
                "Rationale": rationale
            })
            
            time.sleep(3) # Safe buffer
            
        except Exception as e:
            print(f"⚠️ Error Row {idx}: {e}")
            results.append({"id": row['id'], "Prediction": 0, "Rationale": "Processing Error"})

    # 3. SAVE
    pd.DataFrame(results).to_csv("results.csv", index=False)
    print("\n🎉 DONE! 'results.csv' is ready for submission.")

if __name__ == "__main__":
    run_main()