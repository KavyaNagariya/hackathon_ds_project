import os
import time
import pandas as pd
import numpy as np
import requests
import json
import re
import csv
from dotenv import load_dotenv
import google.generativeai as genai

# --- CONFIG ---
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_KEY)
VALID_MODEL_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-1b-it:generateContent?key={GEMINI_KEY}"

# --- DATABASE LOAD ---
if not os.path.exists("data/novel_matrix.npy"):
    print("❌ Database missing! Run 'create_database.py' first.")
    exit()

novel_matrix = np.load("data/novel_matrix.npy")
chunks_df = pd.read_csv("data/novel_chunks.csv")
proper_chunks = chunks_df['chunk'].tolist()

def get_query_embedding(text):
    try:
        res = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")
        return res['embedding']
    except:
        return [0.0] * 768

# --- THE "SCORE" PROMPT (0-10) ---
def get_similarity_score(backstory, evidence):
    headers = {'Content-Type': 'application/json'}
    
    # We ask for a simple number. This is easier for the model than complex logic.
    prompt_text = f"""
    Task: Compare the Claim against the Evidence.
    
    EVIDENCE FROM NOVEL:
    {evidence}
    
    CLAIM ABOUT CHARACTER:
    {backstory}
    
    INSTRUCTIONS:
    Rate the accuracy on a scale of 0 to 10.
    - 0 to 4: The Evidence contradicts the Claim, or the Claim mentions specific details (ships, crimes, dates) NOT found in the Evidence.
    - 5 to 6: Vague match. Plausible but unsupported.
    - 7 to 10: Strong match. The Evidence explicitly confirms the key events in the Claim.
    
    OUTPUT FORMAT:
    SCORE: [Number]
    REASON: [One short sentence]
    """

    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    
    for attempt in range(3):
        try:
            response = requests.post(VALID_MODEL_URL, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                text = response.json()['candidates'][0]['content']['parts'][0]['text']
                return text
            elif response.status_code == 429:
                time.sleep(10)
        except:
            time.sleep(2)
    return "SCORE: 0 REASON: Error"

# --- CLEANER ---
def parse_output(raw_text):
    if not raw_text: return 0, "Error"
    clean = raw_text.strip().replace("\n", " ")
    
    # 1. Extract Score
    score = 0
    match = re.search(r"SCORE:\s*(\d+)", clean)
    if match:
        score = int(match.group(1))
    
    # 2. Extract Reason
    reason = "Inconsistent based on evidence."
    if "REASON:" in clean:
        reason = clean.split("REASON:", 1)[1].strip()
    
    # 3. Apply Threshold (Magic Number = 7)
    # Only scores of 7, 8, 9, 10 count as "Consistent" (1)
    pred = 1 if score >= 7 else 0
    
    # 4. Sanitation (No quotes/commas)
    reason = reason.replace(",", ";").replace('"', '').replace("'", "")
    
    return pred, reason, score

# --- MAIN RUN ---
def run_score_pipeline():
    print("Converting queries embeddings and providing predictions...")
    
    df_test = pd.read_csv("data/test.csv")
    results = []
    
    for idx, row in df_test.iterrows():
        try:
            # 1. Search
            q_vec = np.array(get_query_embedding(row['content']))
            scores = np.dot(novel_matrix, q_vec)
            top_indices = np.argsort(scores)[-5:][::-1]
            evidence = "\n".join([str(proper_chunks[i]) for i in top_indices])
            
            # 2. Score (0-10)
            raw_output = get_similarity_score(row['content'], evidence)
            pred, rationale, score = parse_output(raw_output)
            
            # 3. Console Feedback
            icon = "✅" if pred == 1 else "❌"
            print(f"[{idx+1}/{len(df_test)}] {icon} (Score: {score}) | {rationale[:50]}...")
            
            results.append({
                "id": row['id'],
                "Prediction": pred,
                "Rationale": rationale
            })
            
            time.sleep(5) # Rate limit safety
            
        except Exception as e:
            print(f" Error Row {idx}: {e}")
            results.append({"id": row['id'], "Prediction": 0, "Rationale": "Error"})

    # Save cleanly
    pd.DataFrame(results).to_csv("results.csv", index=False, quoting=csv.QUOTE_NONE, escapechar=' ')
    print("\n DONE! Download 'results.csv' now.")

if __name__ == "__main__":
    run_score_pipeline()