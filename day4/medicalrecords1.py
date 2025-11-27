import os
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- A. Synthetic Medical Records ---
# In a real application, you would load these from files (PDF, JSON, EHR export).
# We use Python strings here to simulate de-identified, chunked patient notes.
medical_records = [
    """
    Patient ID: P1001, Name: John Doe
    Date of Visit: 2025-10-20
    Diagnosis: Type 2 Diabetes Mellitus (ICD-10: E11.9)
    Medication: Metformin 500mg, twice daily. 
    Notes: Patient's A1C level is 7.5%. Advised diet modification and increased physical activity.
    """,
    """
    Patient ID: P1001, Name: John Doe
    Date of Visit: 2025-11-15
    Chief Complaint: Persistent joint pain in the knees (Osteoarthritis).
    Test Results: X-ray confirmed moderate cartilage wear.
    Treatment: Started on Celecoxib 200mg daily. Follow-up scheduled in 4 weeks.
    """,
    """
    Patient ID: P1002, Name: Alice Smith
    Date of Visit: 2025-09-01
    Diagnosis: Seasonal Allergic Rhinitis.
    Medication: Cetirizine 10mg PRN (as needed). 
    Notes: Allergies primarily triggered by pollen. No asthma history.
    """,
    """
    Patient ID: P1002, Name: Alice Smith
    Date of Visit: 2025-11-20
    Chief Complaint: Routine follow-up. Blood pressure recorded at 120/80 mmHg.
    Vitals: Healthy heart rate (72 bpm). Patient maintains a healthy lifestyle.
    No changes to current medication recommended.
    """
]

# Convert strings into LangChain Document objects
documents = [Document(page_content=record) for record in medical_records]

# --- B. Chunking and Embedding ---
# 1. Split documents into smaller, semantically coherent chunks (Crucial for RAG)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 2. Initialize Ollama Embeddings (Uses nomic-embed-text or the model you pulled)
print("Initializing Ollama Embeddings...")
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Create FAISS Vector Store
# FAISS is an efficient, in-memory index for fast similarity search.
print("Creating FAISS index (Embedding documents)...")
vectorstore = FAISS.from_documents(docs, ollama_embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Retrieve top 2 relevant documents

# --- C. RAG Chain Definition ---
# 1. Initialize Ollama LLM
ollama_llm = ChatOllama(model="llama3", temperature=0)

# 2. Define the RAG Prompt Template
# The template instructs the LLM to use the provided context and remain factual.
RAG_PROMPT_TEMPLATE = """
You are a highly specialized medical assistant. Your task is to accurately and concisely answer the question
based ONLY on the medical records provided in the context below. Do not use external knowledge.
If the information is not in the context, state that explicitly.

CONTEXT:
{context}

QUESTION: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# 3. Construct the RAG Chain using LCEL
rag_chain = (
    # Pass the question to the retriever, and the result (context) to the prompt template
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | ollama_llm
    | StrOutputParser()
)

# --- D. Query the RAG System ---
user_query = "What medications is patient P1001 currently taking and for what conditions?"

print(f"\n--- Querying Patient Records ---")
print(f"User Query: {user_query}")
print("-" * 30)

# Execute the RAG chain:
# 1. Question is embedded.
# 2. FAISS finds the most similar documents (records P1001's diabetes and joint pain).
# 3. Those documents are inserted into the RAG_PROMPT_TEMPLATE as CONTEXT.
# 4. Ollama (llama3) reads the context and the question to generate the final answer.
final_answer = rag_chain.invoke(user_query)

print(f"\n✅ LLM (Ollama) Answer:")
print(final_answer)

# Example of a query where the answer is NOT in the documents
query_outside_context = "What is the recommended dosage for Penicillin for children?"
print(f"\n--- Querying Outside Context ---")
print(f"User Query: {query_outside_context}")
print("-" * 30)

final_answer_out = rag_chain.invoke(query_outside_context)
print(f"\n✅ LLM (Ollama) Answer:")
print(final_answer_out)