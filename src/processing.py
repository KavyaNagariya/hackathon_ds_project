import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm.embedders import LiteLLMEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter

def build_vector_store(novel_table, api_key):
    """
    Builds a Vector Store Server.
    """
    embedder = LiteLLMEmbedder(
        api_key=api_key,
        model="gemini/text-embedding-004", 
        retry_strategy=pw.udfs.FixedDelayRetryStrategy(),
    )

    splitter = TokenCountSplitter(max_tokens=500, min_tokens=50)

    # We build the vector store server object
    # This automatically handles chunking, embedding, and indexing
    vector_store = VectorStoreServer(
        novel_table,
        embedder=embedder,
        splitter=splitter,
        # We ensure it parses the text column correctly
        parser=pw.xpacks.llm.parsers.ParseUnstructured() 
    )
    
    return vector_store