import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Load test questions
chapter_df = pd.read_excel("BÖLÜM_SONU_SORULARI.xlsx")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Chroma
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("banking_knowledge")

print("\n===================================")
print("TASK 7 RETRIEVAL ERROR ANALYSIS")
print("===================================")

wrong_count = 0

for _, row in chapter_df.iterrows():

    question = row["Soru"]
    correct_answer = str(row["CEVAP METNİ"]).strip()

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"source": "chapter_questions"}
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    matches = [
        correct_answer.lower() in document.lower()
        for document in documents
    ]

    # We only want cases where the correct answer
    # is not found in the top 5 results.
    if not any(matches):

        wrong_count += 1

        print("\n===================================")
        print(f"ERROR #{wrong_count}")
        print("===================================")

        print("\nQuestion:")
        print(question)

        print("\nCorrect answer:")
        print(correct_answer)

        print("\nRetrieved results:")

        for i in range(5):

            print(f"\nResult {i + 1}")
            print("-----------------------------------")
            print("Distance:", round(distances[i], 4))
            print("Document:")
            print(documents[i])

print("\n===================================")
print("ERROR ANALYSIS COMPLETE")
print("===================================")

print("Total Top-5 errors:", wrong_count)

