# Hackathon Data Science Project (RAG Pipeline)

A robust Retrieval-Augmented Generation (RAG) system to verify character backstories against literary novels using Google's Gemini/Gemma models.

## 📂 Project Structure

- **`create_database.py`**: Ingests novels (`data/*.txt`), processes text into chunks, generates embeddings, and saves the vector store (`novel_matrix.npy`).
- **`main.py`**: The execution engine. Loads the database, processes queries from `data/test.csv`, and uses `gemma-3-1b-it` to verify claims.
- **`data/`**: Stores raw texts, the vector database, and input/output CSVs.

## 🚀 Setup & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Build the Database (Run Once)

This reads the novels and creates the vector index.

```bash
python create_database.py
```

_Output: `data/novel_matrix.npy` and `data/novel_chunks.csv`_

### 4. Run the Analysis

This verifies the backstories in `test.csv`.

```bash
python main.py
```

_Output: `results.csv` containing Predictions (0/1) and Rationale._

## 🛠 Features

- **Resilient API Calls**: Includes automatic retries and exponential backoff for Rate Limits (429).
- **Batch Processing**: Efficiently batches embedding requests to avoid timeouts.
- **Strict Formatting**: Forces the LLM to output clean `1 | Reason` format for programmatic parsing.
