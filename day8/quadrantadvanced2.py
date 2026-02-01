import ollama
from qdrant_client import QdrantClient, models

# --- 1. INITIALIZATION ---
client = QdrantClient("http://localhost:6333")
COLLECTION = "agent_knowledge"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

def get_embedding(text):
    return ollama.embed(model=EMBED_MODEL, input=text)['embeddings'][0]

# --- 2. THE AGENT'S TOOL (Search) ---
def search_tool(query):
    """The Agent calls this to look up facts it doesn't know."""
    query_vec = get_embedding(query)
    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        limit=2
    ).points
    return [r.payload['document'] for r in results]

# --- 3. THE AGENTIC ENGINE ---
class Agent:
    def __init__(self):
        self.memory = []

    def run(self, user_goal):
        print(f"👤 USER: {user_goal}")

        # STAGE 1: PLANNER (The Agent thinks before acting)
        planner_prompt = f"""
        You are a planning agent. You have a search tool.
        Goal: {user_goal}
        Should you search the database for more info? (Yes/No)
        If Yes, what specific keyword should you search for?
        Format: Answer | Keyword
        """
        plan = ollama.generate(model=LLM_MODEL, prompt=planner_prompt)['response']
        print(f"🧠 THOUGHT: {plan}")

        # STAGE 2: EXECUTION (The Agent uses the tool)
        context = ""
        if "yes" in plan.lower():
            keyword = plan.split("|")[-1].strip()
            context = search_tool(keyword)
            self.memory.append(f"Search results for {keyword}: {context}")

        # STAGE 3: OUTPUT (Synthesis)
        final_prompt = f"""
        Goal: {user_goal}
        Context found: {context}
        Provide a final, helpful answer.
        """
        response = ollama.generate(model=LLM_MODEL, prompt=final_prompt)['response']
        print(f"🏁 FINAL OUTPUT: {response}\n")

# --- 4. DATA SETUP & RUN ---
def setup_data():
    if client.collection_exists(COLLECTION): client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
    
    docs = ["The company's server room password is 'Blue-Sky-99'.", "Manager: Sarah Chen."]
    points = [models.PointStruct(id=i, vector=get_embedding(t), payload={"document": t}) for i, t in enumerate(docs)]
    client.upsert(COLLECTION, points=points)

if __name__ == "__main__":
    setup_data()
    my_agent = Agent()
    my_agent.run("I need to get into the server room. What is the password?")