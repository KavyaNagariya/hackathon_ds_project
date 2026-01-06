import pathway as pw
import os
from dotenv import load_dotenv

# Load your API keys (OpenAI / Gemini)
load_dotenv()

def main():
    # --- 1. THE LIBRARIAN (Data Input) ---
    # Monitors the 'books' folder for any .txt files
    # Note: Make sure your books are inside a folder named 'books'
    data_sources = pw.io.fs.read(
        "./books",  
        format="binary",
        mode="static",
        with_metadata=True,
    )

    # --- 2. THE BRAIN (Embeddings) ---
    # We use OpenAI Embeddings to match your query_data.py
    # Ensure OPENAI_API_KEY is in your .env file
    embedder_model = pw.xpacks.llm.embedders.OpenAIEmbedder(
        model="text-embedding-3-small",
        cache_strategy=pw.xpacks.llm.embedders.CacheStrategy.DISK, # Caches results so it's faster next time
    )

    # --- 3. THE SERVER (Vector Store) ---
    # This replaces "Chroma.from_documents"
    vector_server = pw.xpacks.llm.vector_store.VectorStoreServer(
        data_sources,
        embedder=embedder_model,
        parser=pw.xpacks.llm.parsers.OpenParse(),
    )

    # --- 4. RUN IT ---
    print("🚀 Pathway Server is starting...")
    print("Host: 127.0.0.1 | Port: 8765")
    print("Keep this script RUNNING. Do not close it!")
    
    # This runs the server forever until you stop it (Ctrl+C)
    vector_server.run_server(
        host="127.0.0.1",
        port=8765,
        threaded=False, # We want this to be the main process
        with_cache=True
    )

if __name__ == "__main__":
    main()
