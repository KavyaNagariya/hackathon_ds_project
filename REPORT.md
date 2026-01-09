# Technical Approach Report

## 1. Overall Approach

We implemented a **Retrieval-Augmented Generation (RAG)** pipeline to verify character backstories against full-text novels.

1.  **Ingestion:** We read raw text files and split them into 500-word chunks.
2.  **Indexing:** We generated vector embeddings (using `models/text-embedding-004`) for all chunks and stored them in a local numpy matrix for low-latency access.
3.  **Search:** For each user query, we retrieve the Top-5 most semantically similar chunks.
4.  **Reasoning:** We employ a "Judge" LLM (`gemma-3-1b-it`) to compare the retrieved evidence against the claim and output a strictly formatted binary verdict (1=Consistent, 0=Contradictory).

## 2. Handling Long Context

Novels are too large to fit into a standard LLM context window (often 1M+ tokens for a corpus). We solved this via **Chunking & Retrieval**:

- Instead of reading the whole book at once, we broke it into **500-word segments**.
- We only feed the model the **Top 5 chunks** (~2,500 words) that are most relevant to the specific claim.
- This turns a "needle in a haystack" problem into a focused reading comprehension task.

## 3. Distinguishing Signal from Noise

We filter noise in two stages:

1.  **Semantic Ranking (Signal Extraction):**
    - Vector Search filters out 99% of the book that is irrelevant.
    - If a backstory mentions "mutiny," the embeddings naturally surface chunks containing "mutiny," "revolt," or "betrayal."
2.  **LLM Reasoning (Noise Rejection):**
    - The retrieval step often brings back tangential mentions (e.g., a "mutiny" in a different timeline).
    - The Logic Checker (Gemma) explicitly checks if the _specific_ details in the evidence support the _specific_ claim, returning "0" if the retrieved text is unrelated, effectively ignoring the noise.

## 4. Key Limitations & Failure Cases

1.  **Implicit Knowledge:** If a fact is never explicitly stated but only implied across 50 pages (e.g., a slow character arc), the chunk-based retriever might miss the connection.
2.  **Retrieval Failure:** If the embedding model fails to match the query vocabulary to the book's vocabulary (e.g., "The villain" vs "He"), we fetch the wrong chunks, and the Judge correctly returns "0" (False Negative).
3.  **API Rate Limits:** The Free Tier of Gemini API is strict (15 RPM). The system is heavily throttled (`time.sleep(10)`), making it slow for large datasets.
