import os
import time
import pandas as pd
import numpy as np
import requests
import json
import csv
from dotenv import load_dotenv
import google.generativeai as genai

# --- CONFIG ---
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_KEY)
VALID_MODEL_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-1b-it:generateContent?key={GEMINI_KEY}"

def get_query_embedding(text):
    try:
        res = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")
        return res['embedding']
    except:
        return [0.0] * 768

# --- CHAIN OF THOUGHT (CoT) PROMPT ---
def verify_claim(backstory, evidence):
    headers = {'Content-Type': 'application/json'}
    
    # We provide EXAMPLES so the model copies the logic.
    prompt_text = f"""
    You are a Literary Investigator. Your task is to verify if a CLAIM is supported by the EVIDENCE text.

    [EXAMPLE 1: INCONSISTENT CASE]
    EVIDENCE: "He lived in Rome during 1840."
    CLAIM: "He was a baker in Paris in 1840."
    THOUGHTS: The Claim mentions Paris. The Evidence places him in Rome. Locations contradict.
    VERDICT: 0 | The evidence places him in Rome; not Paris.

    [EXAMPLE 2: CONSISTENT CASE]
    EVIDENCE: "Dantes was imprisoned in the Chateau d'If for 14 years."
    CLAIM: "He spent over a decade in the Chateau d'If prison."
    THOUGHTS: "14 years" matches "over a decade". "Chateau d'If" matches.
    VERDICT: 1 | The evidence confirms his 14-year imprisonment in Chateau d'If.

    [YOUR TURN]
    EVIDENCE:
    {evidence}
    
    CLAIM:
    {backstory}
    
    INSTRUCTIONS:
    1. Compare proper nouns (Names, Places, Ships). If the Claim has a name NOT in the Evidence, be skeptical.
    2. Check the Action. Does the Evidence support the specific act described?
    3. Output the Final Verdict exactly like the examples.

    VERDICT:
    """

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}]
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
                time.sleep(60)
                continue
            elif response.status_code in [404, 503]:
                return f"0 | API Error {response.status_code}"
        except:
            time.sleep(5)
            
    return "0 | Connection Failed"

# --- CLEANER FUNCTION (No Quotes, No Hallucinated Numbers) ---
def clean_parsing(raw_text):
    if not raw_text: return 0, "Error processing"
    clean = raw_text.strip()
    
    # 1. Parsing: Look for "VERDICT:" if the model typed it
    if "VERDICT:" in clean:
        clean = clean.split("VERDICT:")[-1].strip()
        
    pred = 0
    if clean.startswith("1"): pred = 1
    elif clean.startswith("0"): pred = 0
    
    # 2. Extract Rationale
    rationale = clean
    for prefix in ["1 |", "0 |", "1.", "0.", "1 ", "0 "]:
        if rationale.startswith(prefix):
            rationale = rationale[len(prefix):].strip()
            
    # 3. CSV Sanitation (Commas -> Semicolons)
    # This prevents the "double quote" issue forever.
    rationale = rationale.replace(",", ";").replace('"', '').replace("'", "").replace("\n", " ")
    
    # 4. Logic Safety Check
    lower = rationale.lower()
    # If rationale clearly says "not support" but pred is 1, fix it.
    if pred == 1 and ("not support" in lower or "contradicts" in lower or "no mention" in lower):
        pred = 0
        
    return pred, rationale

def run_cot_pipeline():
    print("Using Chain Of Thought ")
    
    if not os.path.exists("data/novel_matrix.npy"):
        print(" Database missing!")
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
            
            # Reason (CoT)
            raw_verdict = verify_claim(row['content'], evidence)
            
            # Clean
            pred, rationale = clean_parsing(raw_verdict)
            
            status_icon = "✅" if pred == 1 else "❌"
            print(f"[{idx+1}/{len(df_test)}] {status_icon} | {rationale[:60]}...")
            
            results.append({
                "id": row['id'],
                "Prediction": pred,
                "Rationale": rationale
            })
            
            time.sleep(10) # 10s delay
            
        except Exception as e:
            print(f"Error Row {idx}: {e}")
            results.append({"id": row['id'], "Prediction": 0, "Rationale": "Error"})

    # Save with NO QUOTING
    pd.DataFrame(results).to_csv("results.csv", index=False, quoting=csv.QUOTE_NONE, escapechar=' ')
    print("\n  DONE! 'results.csv' ready.")

if __name__ == "__main__":
    run_cot_pipeline()