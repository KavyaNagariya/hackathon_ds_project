import os
import time
import re
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

# --- 1. TARGET GEMMA (The only one that works for you) ---
VALID_MODEL_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-1b-it:generateContent?key={GEMINI_KEY}"

# --- 2. EMBEDDING HELPER ---
def get_query_embedding(text):
    try:
        res = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")
        return res['embedding']
    except:
        return [0.0] * 768

# --- 3. REASONING ---
def verify_claim(backstory, evidence):
    headers = {'Content-Type': 'application/json'}
    
    # Precise Prompt to stop numbering
    payload = {
        "contents": [{
            "parts": [{
                "text": f"""
                You are a consistency checker.
                
                EVIDENCE: {evidence}
                CLAIM: {backstory}
                
                TASK:
                Does the Evidence support the Claim?
                
                OUTPUT INSTRUCTIONS:
                - Start with 1 (Consistent) or 0 (Inconsistent).
                - Use a pipe symbol '|'.
                - Write a single sentence explanation.
                - DO NOT use bullet points or numbering like '1.'.
                
                Example:
                1 | The evidence confirms he was in Paris.
                0 | The evidence states he was in Rome, not Paris.
                """
            }]
        }]
    }
    
    for attempt in range(5):
        try:
            response = requests.post(VALID_MODEL_URL, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                try:
                    return response.json()['candidates'][0]['content']['parts'][0]['text']
                except:
                    return "0 | Safety Filter"
            elif response.status_code == 429:
                time.sleep(60) # Wait for rate limit
                continue
            elif response.status_code in [404, 503]:
                return f"0 | API Error {response.status_code}"
        except:
            time.sleep(5)
            
    return "0 | Connection Failed"

# --- 4. THE CLEANER FUNCTION ---
def clean_parsing(raw_text):
    """
    Cleans up the messy output from the model.
    """
    if not raw_text: return 0, "Error processing."
    
    clean = raw_text.strip()
    
    # 1. Get Prediction Number
    pred = 0
    # Check if it starts with 1 or 0
    if clean.startswith("1"): pred = 1
    elif clean.startswith("0"): pred = 0
    
    # 2. Extract Rationale
    # Remove the "1 |" or "1." or "1," prefix
    rationale = clean
    for prefix in ["1 |", "0 |", "1.", "0.", "1,", "0,"]:
        if rationale.startswith(prefix):
            rationale = rationale[len(prefix):].strip()
            break
            
    # Also strip raw numbers if they remain
    if len(rationale) > 1 and rationale[0].isdigit() and rationale[1] in [".", " "]:
        rationale = rationale[2:].strip()

    # 3. SAFETY NET: Fix Logical Contradictions
    # If the text says "contradicts", force pred to 0
    lower_rat = rationale.lower()
    if "contradicts" in lower_rat or "does not support" in lower_rat or "inconsistent" in lower_rat:
        pred = 0
    
    return pred, rationale

# --- 5. MAIN PIPELINE ---
def run_final_pipeline():
    print("(Using 'gemma-3-1b-it' with safety delays)")
    
    if not os.path.exists("data/novel_matrix.npy"):
        print("Database missing! Run 'create_database.py' first.")
        return
        
    novel_matrix = np.load("data/novel_matrix.npy")
    chunks_df = pd.read_csv("data/novel_chunks.csv")
    proper_chunks = chunks_df['chunk'].tolist()
    
    df_test = pd.read_csv("data/test.csv")
    results = []
    
    for idx, row in df_test.iterrows():
        try:
            # Search
            q_vec = np.array(get_query_embedding(row['content']))
            scores = np.dot(novel_matrix, q_vec)
            top_indices = np.argsort(scores)[-5:][::-1]
            evidence = "\n".join([str(proper_chunks[i]) for i in top_indices])
            
            # Reason
            raw_verdict = verify_claim(row['content'], evidence)
            
            # Clean
            pred, rationale = clean_parsing(raw_verdict)
            
            status_icon = "✅" if pred == 1 else "❌"
            print(f"[{idx+1}/{len(df_test)}] {status_icon} | {rationale[:50]}...")
            
            results.append({
                "id": row['id'],
                "Prediction": pred,
                "Rationale": rationale
            })
            
            time.sleep(10) # Keep safe delay
            
        except Exception as e:
            print(f"Error Row {idx}: {e}")
            results.append({"id": row['id'], "Prediction": 0, "Rationale": "Error"})

    # Save
    pd.DataFrame(results).to_csv("results.csv", index=False)
    print("\n DONE! 'results.csv' is ready.")

if __name__ == "__main__":
    run_final_pipeline()