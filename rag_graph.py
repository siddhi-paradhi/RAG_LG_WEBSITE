from langgraph.graph.state import StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from typing import TypedDict, List, Any
import os

load_dotenv()
TOGETHER_key = os.getenv("TOGETHER_API_KEY")
TOGETHER_base = os.getenv("TOGETHER_API_BASE")

os.environ["OPENAI_API_KEY"] = TOGETHER_key
os.environ["OPENAI_API_BASE"] = TOGETHER_base

qdrant = QdrantClient(host="localhost", port=6333)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = QdrantVectorStore(
    client=qdrant,
    collection_name="website_rag",
    embedding=embedding_model,
    content_payload_key="page_content"
)

llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.2,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

class RagState(TypedDict):
    question: str
    answer: str
    sources: List[Any]

def rag_node(state: RagState) -> RagState:
    query = state["question"]
    result = qa_chain.invoke(query)
    return {
        "question": query,
        "answer": result["result"],
        "sources": result["source_documents"]
    }

builder = StateGraph(RagState)
builder.add_node("RAGChain", rag_node)
builder.set_entry_point("RAGChain")
builder.set_finish_point("RAGChain")
graph = builder.compile()

def run_rag_query(query: str):
    result = graph.invoke({"question": query, "answer": "", "sources": []})
    return result["answer"], result["sources"]

if __name__ == "__main__":
    user_query = input("Ask something: ")
    answer, sources = run_rag_query(user_query)
    
    print("\nAnswer:\n", answer)
    print("\nSources:\n")
    for doc in sources:
        print("-", doc.metadata.get("text", "No text metadata"))