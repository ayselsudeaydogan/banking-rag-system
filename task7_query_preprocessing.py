import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


def preprocess_question(question):
    """
    Removes generic question patterns that do not
    contribute much to semantic retrieval.
    """

    generic_patterns = [
        "Aşağıdakilerden hangisi",
        "Aşağıdaki ifadelerden hangisi",
        "Aşağıdaki seçeneklerden hangisi",
        "Aşağıdakilerin hangisinde",
        "Aşağıdaki seçeneklerden",
        "hangisi"
    ]

    processed = question

    for pattern in generic_patterns:
        processed = processed.replace(pattern, "")

    return processed.strip()


# Load test questions
chapter_df = pd.read_excel("BÖLÜM_SONU_SORULARI.xlsx")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Chroma
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("banking_knowledge")

print("\n===================================")
print("TASK 7 QUERY PREPROCESSING TEST")
print("===================================")

print("Test questions:", len(chapter_df))
print("Chroma documents:", collection.count())

top1_correct = 0
top3_correct = 0
top5_correct = 0

for _, row in chapter_df.iterrows():

    original_question = row["Soru"]
    correct_answer = str(row["CEVAP METNİ"]).strip()

    # Preprocess question
    processed_question = preprocess_question(original_question)

    # Create embedding from processed query
    query_embedding = model.encode(processed_question).tolist()

    # Search only chapter questions
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"source": "chapter_questions"}
    )

    documents = results["documents"][0]

    matches = [
        correct_answer.lower() in document.lower()
        for document in documents
    ]

    if matches[0]:
        top1_correct += 1

    if any(matches[:3]):
        top3_correct += 1

    if any(matches[:5]):
        top5_correct += 1


total = len(chapter_df)

top1 = top1_correct / total * 100
top3 = top3_correct / total * 100
top5 = top5_correct / total * 100


print("\n===================================")
print("QUERY PREPROCESSING RESULTS")
print("===================================")

print(f"\nTop-1 Accuracy: {top1:.2f}%")
print(f"Top-3 Accuracy: {top3:.2f}%")
print(f"Top-5 Accuracy: {top5:.2f}%")

print("\nCorrect Top-1:", top1_correct)
print("Correct Top-3:", top3_correct)
print("Correct Top-5:", top5_correct)


print("\n===================================")
print("COMPARISON")
print("===================================")

print("\nBaseline:")
print("Top-1: 26.25%")
print("Top-3: 42.50%")
print("Top-5: 51.25%")

print("\nMetadata Filtering:")
print("Top-1: 60.00%")
print("Top-3: 76.25%")
print("Top-5: 80.00%")

print("\nQuery Preprocessing:")
print(f"Top-1: {top1:.2f}%")
print(f"Top-3: {top3:.2f}%")
print(f"Top-5: {top5:.2f}%")