import pathway as pw
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load API keys from .env
load_dotenv()

def main():
    # --- 1. The Input (Books) ---
    # Reads all .txt files from the 'books' folder
    data_sources = pw.io.fs.read(
        "./books",
        format="binary",
        mode="static",
        with_metadata=True,
    )

    # --- 2. The Brain (Embeddings) ---
    # We use Google's FREE embeddings model
    embedder_model = pw.xpacks.llm.embedders.LangChainEmbedder(
        GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )

    # --- 3. The Server (Vector Store) ---
    # This creates the searchable index
    vector_server = pw.xpacks.llm.vector_store.VectorStoreServer(
        data_sources,
        embedder=embedder_model,
        parser=pw.xpacks.llm.parsers.OpenParse(),
    )

    # --- 4. Run It ---
    print("🚀 Pathway Server is starting on port 8765...")
    print("Keep this terminal OPEN so the other script can talk to it.")
    
    # Runs forever until you press Ctrl+C
    vector_server.run_server(
        host="0.0.0.0",
        port=8765,
        threaded=False,
        with_cache=True
    )

if __name__ == "__main__":
    main()