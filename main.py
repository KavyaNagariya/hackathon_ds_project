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

# --- 1. TARGET THE ONE WORKING MODEL (GEMMA) ---
# Your logs confirmed 'gemma-3-1b-it' is the only one responding.
VALID_MODEL_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-1b-it:generateContent?key={GEMINI_KEY}"

# --- 2. EMBEDDING HELPER ---
def get_query_embedding(text):
    try:
        res = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")
        return res['embedding']
    except:
        return [0.0] * 768

# --- 3. REASONING (Gemma Mode) ---
def verify_claim(backstory, evidence):
    headers = {'Content-Type': 'application/json'}
    
    # Simple prompt for Gemma
    payload = {
        "contents": [{
            "parts": [{
                "text": f"""
                You are a logic checker.
                
                EVIDENCE: {evidence}
                CLAIM: {backstory}
                
                TASK:
                Does the Evidence support the Claim?
                Reply exactly like this:
                1 | [One short sentence explaining why]
                OR
                0 | [One short sentence explaining why]
                """
            }]
        }]
    }
    
    # RETRY LOOP
    for attempt in range(5):
        try:
            response = requests.post(VALID_MODEL_URL, headers=headers, data=json.dumps(payload))
            
            # SUCCESS
            if response.status_code == 200:
                try:
                    return response.json()['candidates'][0]['content']['parts'][0]['text']
                except:
                    return "0 | Safety Filter"
            
            # RATE LIMIT (429) -> WAIT LONG TIME
            elif response.status_code == 429:
                wait_time = 60  # Wait 1 minute
                print(f"   ⚠️ Rate Limit (429). Cooling down for {wait_time}s...", end="\r")
                time.sleep(wait_time)
                continue
            
            # 404 or 503
            elif response.status_code in [404, 503]:
                # If Gemma fails, we have no backup, so just return error 0
                return f"0 | API Error {response.status_code}"
                
            else:
                print(f"   ⚠️ Error {response.status_code}...")
                return f"0 | Error {response.status_code}"

        except:
            time.sleep(5)
            
    return "0 | Connection Failed"

# --- 4. MAIN PIPELINE ---
def run_gemma_pipeline():
    print("🚀 STARTING GEMMA PIPELINE (The Chosen One)...")
    print("   (Using 'gemma-3-1b-it' with safety delays)")
    
    if not os.path.exists("data/novel_matrix.npy"):
        print("❌ Database missing! Run 'create_database.py' first.")
        return
        
    novel_matrix = np.load("data/novel_matrix.npy")
    chunks_df = pd.read_csv("data/novel_chunks.csv")
    proper_chunks = chunks_df['chunk'].tolist()
    
    df_test = pd.read_csv("data/test.csv")
    results = []
    
    for idx, row in df_test.iterrows():
        try:
            # 1. Search
            q_vec = np.array(get_query_embedding(row['content']))
            scores = np.dot(novel_matrix, q_vec)
            top_indices = np.argsort(scores)[-5:][::-1]
            evidence = "\n".join([str(proper_chunks[i]) for i in top_indices])
            
            # 2. Reason
            raw_verdict = verify_claim(row['content'], evidence)
            
            # 3. Clean Parsing
            clean = raw_verdict.strip()
            
            # Default Prediction
            pred = 0
            rationale = "Inconsistent or API Error"
            
            if len(clean) > 0:
                # Check first character
                if clean[0] == '1': pred = 1
                elif clean[0] == '0': pred = 0
                
                # Extract rationale
                if "|" in clean:
                    rationale = clean.split("|", 1)[1].strip()
                else:
                    rationale = clean[1:].strip()
            
            # Cleanup
            if len(rationale) < 5: rationale = "Inconsistent with provided narrative."

            status_icon = "✅" if pred == 1 else "❌"
            print(f"[{idx+1}/{len(df_test)}] {status_icon} | {rationale[:40]}...")
            
            results.append({
                "id": row['id'],
                "Prediction": pred,
                "Rationale": rationale
            })
            
            # CRITICAL: SLEEP 10 SECONDS
            time.sleep(10) 
            
        except Exception as e:
            print(f"⚠️ Error Row {idx}: {e}")
            results.append({"id": row['id'], "Prediction": 0, "Rationale": "Error"})

    pd.DataFrame(results).to_csv("results.csv", index=False)
    print("\n🎉 DONE! Download 'results.csv'.")

if __name__ == "__main__":
    run_gemma_pipeline()