from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)  # or however you're connecting

client.delete_collection("my_collection")
