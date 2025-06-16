import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

DATA_DIR = "./trusted_sites"
CHUNK_SIZE = 500

documents = []

if os.path.exists(DATA_DIR):
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    content = entry.get("content", "").strip()
                    url = entry.get("url", "unknown")
                    if content:
                        chunks = [content[i:i + CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
                        for i, chunk in enumerate(chunks):
                            documents.append(
                                Document(
                                    page_content=chunk,
                                    metadata={"source": url, "chunk_id": f"{filename}_chunk_{i}"}
                                )
                            )

else:
    print(f"Warning: JSON folder {DATA_DIR} does not exist. Skipping JSON data.")

if os.path.exists("website_data.txt"):
    with open("website_data.txt", "r", encoding="utf-8") as f:
        text = f.read()
        chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": f"website_data.txt_chunk_{i}"}
                )
            )
else:
    print("Warning: website_data.txt not found — skipping.")

print(f"Total documents to upload: {len(documents)}")

if len(documents) == 0:
    print("No documents to upload. Please check your data sources.")
    exit()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="website_rag",
    content_payload_key="page_content"
)

print(f"Uploaded {len(documents)} chunks to Qdrant collection 'website_rag'")
