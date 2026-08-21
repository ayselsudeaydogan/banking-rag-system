import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to existing Chroma database
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("banking_knowledge")

print("Collection:", collection.name)
print("Total documents:", collection.count())

# Test questions
questions = [
    "Emeklilik Gözetim Merkezi ne zaman kurulmuştur?",
    "Bir finansal sistemde en fazla fon sunan birim hangisidir?",
    "Basel-II'nin amacı nedir?"
]

for question in questions:

    print("\n===================================")
    print("QUESTION")
    print("===================================")
    print(question)

    # Convert question to embedding
    query_embedding = model.encode(question).tolist()

    # Search Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    print("\nTOP 5 RESULTS")
    print("===================================")

    for i in range(5):
        print(f"\nResult {i + 1}")
        print("-----------------------------------")
        print("Distance:", results["distances"][0][i])
        print("Source:", results["metadatas"][0][i]["source"])
        print("Type:", results["metadatas"][0][i]["type"])
        print("Document:")
        print(results["documents"][0][i])
