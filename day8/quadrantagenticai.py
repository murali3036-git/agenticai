import ollama
from qdrant_client import QdrantClient, models

client = QdrantClient("http://localhost:6333")
COLLECTION = "enterprise_v2"

def get_embed(text):
    return ollama.embed(model="nomic-embed-text", input=text)['embeddings'][0]

def setup_real_data():
    if client.collection_exists(COLLECTION): client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
    
    # THE ACTUAL DATA (The only truth)
    real_facts = [
        "Project Aegis has an officially allocated budget of $10,000.",
        "The project manager is Sarah Chen."
    ]
    points = [models.PointStruct(id=i, vector=get_embed(f), payload={"text": f}) for i, f in enumerate(real_facts)]
    client.upsert(COLLECTION, points=points)
    print("✅ Database updated with REAL budget: $10,000")

def search_tool(query):
    res = client.query_points(COLLECTION, query=get_embed(query), limit=1).points
    # Safety Check: If nothing is found, return a very clear 'empty' signal
    if not res: return "DATA_NOT_FOUND"
    return res[0].payload['text']

def run_grounded_agent(question):
    # The 'System Prompt' is now much stricter
    system_rules = """
    You are a strict data assistant. 
    Rule 1: Only answer using provided search results.
    Rule 2: If the search returns 'DATA_NOT_FOUND', say 'I do not have that information'.
    Rule 3: NEVER make up numbers.
    """
    
    # Step 1: Search
    context = search_tool(question)
    
    # Step 2: Final Answer Generation
    prompt = f"{system_rules}\n\nContext: {context}\nUser Question: {question}\nAnswer:"
    response = ollama.generate(model="llama3", prompt=prompt)['response']
    print(f"🏁 REAL ANSWER: {response}")

if __name__ == "__main__":
    setup_real_data()
    run_grounded_agent("What is the budget for Aegis?")