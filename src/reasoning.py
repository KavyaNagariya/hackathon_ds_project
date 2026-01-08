from google import genai

def verify_consistency(backstory, retrieved_context, gemini_key):
    """
    Uses the modern Google GenAI SDK for reasoning.
    cite: 3.1
    """
    client = genai.Client(api_key=gemini_key)
    
    prompt = f"""
    You are a literary analyst. Compare the evidence to the backstory.
    
    NOVEL EVIDENCE:
    {retrieved_context}
    
    HYPOTHETICAL BACKSTORY:
    {backstory}
    
    RULES:
    - Output '1' if consistent, '0' if it contradicts facts/dates/traits.
    - Provide a short 1-sentence Rationale.
    
    FORMAT:
    Prediction: [0/1]
    Rationale: [Reason]
    """
    
    # Using the latest 2.0 Flash for high speed and reasoning
    response = client.models.generate_content(
        model='gemini-2.0-flash', 
        contents=prompt
    )
    
    return response.text