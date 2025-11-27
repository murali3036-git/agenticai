"""
Docstring for agentloop.hybrid

Hybrid RAG (Basic Multiple Retrievers)
In advanced RAG, you might combine different types of retrieval. This example shows a simple hybrid approach where the LLM is given context from two separate sources (two different vector stores) before generating the answer.
"""

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- A. Two Separate Data Sources ---
# Source 1: Company Policy Documents
policy_docs = [
    Document(page_content="Policy P101: All vacation requests must be submitted 14 days in advance.", metadata={"source": "HR Policy"}),
    Document(page_content="Policy P102: Remote work is permitted for up to 3 days per week with manager approval.", metadata={"source": "HR Policy"}),
]
# Source 2: Technical Procedure Manuals
procedure_docs = [
    Document(page_content="Procedure T201: System reboot sequence takes 15 minutes and must be logged in Jira.", metadata={"source": "IT Manual"}),
    Document(page_content="Procedure T202: To restore a backup, contact the lead engineer on duty.", metadata={"source": "IT Manual"}),
]

# --- B. Embeddings and Two Vector Stores ---
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create two independent vector stores
policy_vectorstore = FAISS.from_documents(policy_docs, ollama_embeddings)
procedure_vectorstore = FAISS.from_documents(procedure_docs, ollama_embeddings)

# --- C. The Hybrid Context Aggregator ---
def aggregate_context(question):
    """Retrieves relevant context from both vector stores."""
    # Retrieve 1 most relevant document from HR policies
    policy_context = policy_vectorstore.as_retriever(k=1).invoke(question)
    # Retrieve 1 most relevant document from IT procedures
    procedure_context = procedure_vectorstore.as_retriever(k=1).invoke(question)
    
    # Combine the content from both sources into one large context string
    combined_content = (
        "--- HR Policy Context ---\n" + "\n".join([d.page_content for d in policy_context]) +
        "\n\n--- IT Procedure Context ---\n" + "\n.join([d.page_content for d in procedure_context])" # FIX: Use correct join
    )
    # Correction for join on the second part:
    combined_content = (
        "--- HR Policy Context ---\n" + "\n".join([d.page_content for d in policy_context]) +
        "\n\n--- IT Procedure Context ---\n" + "\n".join([d.page_content for d in procedure_context])
    )

    return combined_content

# --- D. RAG Chain and Query ---

# 🛑 FIX: Define the required rag_prompt variable 
RAG_TEMPLATE = """You are an expert assistant. Use the following retrieved context 
to answer the user's question. If the information is not present in the context, state 
that you cannot answer the specific part of the question.

Context:
{context}

Question: {question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE) # This resolves the NameError

ollama_llm = ChatOllama(model="llama3", temperature=0)

# The input runs through the custom aggregator (RunnableLambda) first
hybrid_chain = (
    # This structure maps the original question to both the 'context' and 'question' keys
    {"context": RunnableLambda(aggregate_context), "question": RunnablePassthrough()}
    | rag_prompt # This now correctly references the defined prompt
    | ollama_llm
    | StrOutputParser()
)

user_query = "If I need to reboot the server, how long will it take, and how far in advance do I need to ask for a vacation?"

print(f"\n--- Hybrid RAG System (Two Sources) ---")
print(f"User Query: {user_query}")
print("-" * 40)

# The chain retrieves two pieces of context (one policy, one procedure) and answers both parts.
final_answer = hybrid_chain.invoke(user_query)

print(f"\n✅ LLM (Ollama) Answer:")
print(final_answer)