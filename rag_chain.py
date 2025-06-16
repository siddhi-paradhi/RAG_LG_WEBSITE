from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
import os

os.environ["OPENAI_API_KEY"] = "e9d39388365f9404a7ba3286d892ec87a3c0794e4b451ce2218298ad5ef52a5f"  
os.environ["OPENAI_API_BASE"] = "https://api.together.xyz/v1"

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

query = input("Ask something: ")
result = qa_chain.invoke(query)

print("\nAnswer:\n", result["result"])
print("\nSources:\n")
for doc in result["source_documents"]:
    print("—", doc.metadata.get("text", "No text metadata"))