from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load API keys
load_dotenv()

def main():
    # --- CHANGE: Use Google Gemini instead of OpenAI ---
    # Make sure GOOGLE_API_KEY is in your .env file
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 1. Get embedding for a word
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector[:5]}... (Length: {len(vector)})")

    # 2. Compare two words (Calculates distance)
    # Note: We manually calculate distance because the evaluator might default to OpenAI
    vector_a = embedding_function.embed_query("apple")
    vector_b = embedding_function.embed_query("iphone")
    
    # Simple Cosine Similarity check manually to avoid extra dependencies
    import numpy as np
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    score = cosine_similarity(vector_a, vector_b)
    print(f"Similarity between 'apple' and 'iphone': {score}")

if __name__ == "__main__":
    main()