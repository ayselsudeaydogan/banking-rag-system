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
print("TASK 7 RETRIEVAL EVALUATION")
print("===================================")

print("Test questions:", len(chapter_df))
print("Chroma documents:", collection.count())

top1_correct = 0
top3_correct = 0
top5_correct = 0

wrong_predictions = []

for index, row in chapter_df.iterrows():

    question = row["Soru"]
    correct_answer = str(row["CEVAP METNİ"]).strip()

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    documents = results["documents"][0]

    # Check retrieved documents
    matches = []

    for document in documents:
        if correct_answer.lower() in document.lower():
            matches.append(True)
        else:
            matches.append(False)

    if matches[0]:
        top1_correct += 1

    if any(matches[:3]):
        top3_correct += 1

    if any(matches[:5]):
        top5_correct += 1

    if not matches[0]:
        wrong_predictions.append({
            "question": question,
            "correct_answer": correct_answer,
            "top_result": documents[0]
        })

# Calculate accuracy
total = len(chapter_df)

top1_accuracy = top1_correct / total * 100
top3_accuracy = top3_correct / total * 100
top5_accuracy = top5_correct / total * 100

print("\n===================================")
print("RETRIEVAL RESULTS")
print("===================================")

print(f"\nTop-1 Accuracy: {top1_accuracy:.2f}%")
print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

print("\nCorrect Top-1:", top1_correct)
print("Correct Top-3:", top3_correct)
print("Correct Top-5:", top5_correct)

print("\nWrong Top-1 predictions:", len(wrong_predictions))

print("\n===================================")
print("FIRST 10 TOP-1 ERRORS")
print("===================================")

for error in wrong_predictions[:10]:

    print("\nQuestion:")
    print(error["question"])

    print("\nCorrect answer:")
    print(error["correct_answer"])

    print("\nRetrieved document:")
    print(error["top_result"])

    print("-----------------------------------")