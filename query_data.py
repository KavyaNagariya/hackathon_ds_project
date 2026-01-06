import requests
import json

# The URL of your running Pathway server
SERVER_URL = "http://127.0.0.1:8765/v1/retrieve"

def test_query():
    query_text = "Was the Count of Monte Cristo born rich?"
    print(f"--- Testing Query: '{query_text}' ---")

    # The prompt for the AI
    prompt = f"""
    Based on the novel, answer this question: {query_text}
    If the information is not in the book, say "I don't know".
    """

    try:
        # Send the question to the Pathway server
        response = requests.post(
            SERVER_URL,
            json={"query": prompt, "k": 3} # Get top 3 pieces of evidence
        )

        if response.status_code == 200:
            # Print the AI's answer
            print("\n--- AI Response ---")
            print(response.json()['response'])
        else:
            print(f"Error: Server returned status code {response.status_code}")
            print("Did you forget to run create_database.py first?")

    except Exception as e:
        print(f"Connection failed: {e}")
        print("Is the Pathway server running in another terminal?")

if __name__ == "__main__":
    test_query()