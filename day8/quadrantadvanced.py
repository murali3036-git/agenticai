import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue

# Configuration
EMBED_MODEL = "nomic-embed-text" # Make sure to: ollama pull nomic-embed-text
CHAT_MODEL = "llama3"
COLLECTION_NAME = "connected_data"

client = QdrantClient("http://localhost:6333")

def get_embedding(text):
    """Bridge: Converts text into a vector that Qdrant can understand."""
    return ollama.embed(model=EMBED_MODEL, input=text)['embeddings'][0]

def setup_database():
    """Ingestion: Putting real data into the brain."""
    data = [
        {"id": 1, "text": "Our office is located at 123 AI Lane, Tech City.", "dept": "hr"},
        {"id": 2, "text": "The password for the guest WiFi is 'OpenSesame2024'.", "dept": "it"},
        {"id": 3, "text": "Quarterly bonuses are processed on the 15th of next month.", "dept": "hr"}
    ]
    
    # Create collection with the correct vector size for nomic-embed-text
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

    points = [
        PointStruct(id=item['id'], vector=get_embedding(item['text']), payload=item)
        for item in data
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def connected_agent(question):
    print(f"👤 User: {question}")

    # 1. PLANNER: Determine which department to search in
    planner_prompt = f"Categorize this question into 'hr' or 'it': '{question}'. Return only the word."
    dept = ollama.generate(model=CHAT_MODEL, prompt=planner_prompt)['response'].strip().lower()

    # 2. TOOLS: Search using the vector of the ACTUAL question
    query_vector = get_embedding(question) # This creates the RELATION
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(must=[FieldCondition(key="dept", match=MatchValue(value=dept))]),
        limit=1
    ).points

    if not search_results:
        print("❌ No relevant data found.")
        return

    context = search_results[0].payload['text']
    print(f"📖 Found Context: {context}")

    # 3. OUTPUT: Combine the data with the question
    final_answer = ollama.generate(
        model=CHAT_MODEL,
        prompt=f"Answer the question based ONLY on this context: {context}\n\nQuestion: {question}"
    )['response']
    
    print(f"🏁 Final Answer: {final_answer}")

if __name__ == "__main__":
    setup_database()
    connected_agent("What is the WiFi password?")