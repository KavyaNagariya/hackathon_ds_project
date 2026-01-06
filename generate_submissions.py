import pandas as pd
import requests
import time

# The URL of your running Pathway server
SERVER_URL = "http://127.0.0.1:8765/v1/retrieve"

def check_consistency(backstory):
    """Sends a single backstory to the server to be checked."""
    prompt = f"""
    TASK: Determine if the following backstory is consistent with the novel.
    
    BACKSTORY: "{backstory}"
    
    INSTRUCTIONS:
    1. Search the novel for evidence.
    2. If the backstory contradicts the novel, output 'Inconsistent'.
    3. If it fits or there is no contradicting evidence, output 'Consistent'.
    4. Provide a 1-sentence reason after the final answer.
    
    OUTPUT FORMAT: [Consistent/Inconsistent]. [Reasoning].
    """
    
    try:
        response = requests.post(SERVER_URL, json={"query": prompt, "k": 5})
        if response.status_code == 200:
            result = response.json()['response']
            # Simple logic to get the 1/0 prediction
            prediction = 0 if "Inconsistent" in result else 1
            return prediction, result
        else:
            return 0, f"Error: Status {response.status_code}"
            
    except Exception as e:
        return 0, f"Connection Error: {e}"

def main():
    print("--- Starting Submission Generation ---")
    
    # 1. Read the test file
    try:
        df = pd.read_csv("test.csv")
        print(f"Loaded {len(df)} rows from test.csv.")
    except FileNotFoundError:
        print("Error: 'test.csv' not found. Please put it in this folder.")
        return

    predictions = []
    rationales = []

    # 2. Loop through every row
    print("Processing backstories. This may take a while...")
    for index, row in df.iterrows():
        # Show progress every 10 rows
        if index % 10 == 0:
            print(f"Processing row {index}/{len(df)}...")
            
        pred, reason = check_consistency(row['content'])
        predictions.append(pred)
        rationales.append(reason)
        # Wait a bit to be nice to the server
        time.sleep(0.2)

    # 3. Save the final results
    submission_df = pd.DataFrame({
        "id": df['id'],
        "prediction": predictions,
        "rationale": rationales
    })
    
    submission_df.to_csv("results.csv", index=False)
    print("\n✅ Done! 'results.csv' has been created successfully.")

if __name__ == "__main__":
    main()