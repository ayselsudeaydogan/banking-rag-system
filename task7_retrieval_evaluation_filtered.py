import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Load test questions
chapter_df = pd.read_excel("BÖLÜM_SONU_SORULARI.xlsx")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to existing Chroma database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("banking_knowledge")

print("\n===================================")
print("FILTERED RETRIEVAL EVALUATION")
print("===================================")

print("Test questions:", len(chapter_df))
print("Chroma documents:", collection.count())

top1_correct = 0
top3_correct = 0
top5_correct = 0

for _, row in chapter_df.iterrows():

    question = row["Soru"]
    correct_answer = str(row["CEVAP METNİ"]).strip()

    query_embedding = model.encode(question).tolist()

    # Search ONLY chapter questions
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"source": "chapter_questions"}
    )

    documents = results["documents"][0]

    matches = []

    for document in documents:
        matches.append(
            correct_answer.lower() in document.lower()
        )

    if matches[0]:
        top1_correct += 1

    if any(matches[:3]):
        top3_correct += 1

    if any(matches[:5]):
        top5_correct += 1

total = len(chapter_df)

print("\n===================================")
print("FILTERED RETRIEVAL RESULTS")
print("===================================")

print(f"\nTop-1 Accuracy: {top1_correct / total * 100:.2f}%")
print(f"Top-3 Accuracy: {top3_correct / total * 100:.2f}%")
print(f"Top-5 Accuracy: {top5_correct / total * 100:.2f}%")

print("\nCorrect Top-1:", top1_correct)
print("Correct Top-3:", top3_correct)
print("Correct Top-5:", top5_correct)

print("\n===================================")
print("BASELINE COMPARISON")
print("===================================")

print("Baseline Top-1: 26.25%")
print("Baseline Top-3: 42.50%")
print("Baseline Top-5: 51.25%")