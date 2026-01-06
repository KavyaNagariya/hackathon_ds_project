import pandas as pd
import requests
import time

# --- CONFIGURATION ---
# The URL where your create_database.py server is listening
SERVER_URL = "http://127.0.0.1:8765/v1/retrieve"

def check_consistency(backstory, book_name):
    """
    Sends the backstory to the Pathway Server and asks the LLM to judge it.
    """
    # 1. We construct the query
    # We explicitly ask for a binary consistency check
    prompt = f"""
    You are a consistency checker for the novel '{book_name}'.
    
    BACKSTORY TO CHECK:
    "{backstory}"
    
    TASK:
    1. Search the novel for evidence related to this backstory.
    2. Determine if the backstory is CONSISTENT (Possible) or INCONSISTENT (Contradicts the book).
    3. Output ONLY the word 'Consistent' or 'Inconsistent', followed by a short explanation.
    """
    
    try:
        # Send to Pathway (This assumes you are running the server!)
        # Note: We use a simple HTTP request to the Pathway server
        response = requests.post(
            SERVER_URL,
            json={"query": prompt, "k": 5} # k=5 means "get 5 pieces of evidence"
        )
        
        if response.status_code == 200:
            result_text = response.json()['response']
            
            # Simple parsing logic (You can improve this!)
            if "Inconsistent" in result_text:
                return 0, result_text
            else:
                return 1, result_text
        else:
            print(f"Server Error: {response.status_code}")
            return 0, "Error connecting to server"
            
    except Exception as e:
        print(f"Connection Failed: {e}")
        return 0, "Server not running?"

def main():
    print("--- 🚀 STARTING SUBMISSION GENERATOR ---")
    
    # 1. Load the Test Data
    # Note: Based on your files, the column is named 'content', not 'backstory'
    df = pd.read_csv("test.csv")
    print(f"Loaded {len(df)} stories to check.")
    
    predictions = []
    rationales = []
    
    # 2. Loop through every row
    for index, row in df.iterrows():
        story_id = row['id']
        backstory_text = row['content'] # The column name in your CSV
        book = row['book_name']
        
        print(f"Processing ID {story_id} ({book})...")
        
        # Ask the AI
        pred, reason = check_consistency(backstory_text, book)
        
        predictions.append(pred)
        rationales.append(reason)
        
        # Sleep briefly to be nice to the API
        time.sleep(0.5)

    # 3. Save the Results
    submission_df = pd.DataFrame({
        "id": df['id'],
        "prediction": predictions,
        "rationale": rationales
    })
    
    submission_df.to_csv("results.csv", index=False)
    print("\n✅ DONE! Results saved to 'results.csv'. Good luck!")

if __name__ == "__main__":
    main()
