import os
from langchain_community.vectorstores import PathwayVectorClient
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI # <--- NEW IMPORT
from langchain.prompts import ChatPromptTemplate

# 1. SETUP THE SEARCH (Keep OpenAI here)
# This matches the embeddings you used in create_database.py
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Connect to your running Pathway Server (from Step 1)
# Note: Ensure create_database.py is running in a separate terminal/notebook!
db = PathwayVectorClient(
    url="http://127.0.0.1:8765",
    embedding=embeddings
)

# 2. SETUP THE BRAIN (Swap to Gemini here)
# Make sure you have your GOOGLE_API_KEY set in your .env file
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0
)

# 3. THE PROMPT (Standard RAG)
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # Search the book
    results = db.similarity_search_with_score(query_text, k=5)
    
    # Prepare the context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Ask Gemini
    response = model.invoke(prompt)
    
    print(f"Response: {response.content}")
    return response.content
