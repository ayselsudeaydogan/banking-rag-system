import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# SETUP
# ==========================================

chapter_df = pd.read_excel("BÖLÜM_SONU_SORULARI.xlsx")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "banking_knowledge"
)


# ==========================================
# RETRIEVAL DISTANCE ANALYSIS
# ==========================================

distances = []
correct_flags = []


for _, row in chapter_df.iterrows():

    question = row["Soru"]
    correct_answer = str(row["CEVAP METNİ"]).strip()

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        where={"source": "chapter_questions"}
    )

    document = results["documents"][0][0]
    distance = results["distances"][0][0]

    is_correct = (
        correct_answer.lower() in document.lower()
    )

    distances.append(distance)
    correct_flags.append(is_correct)


# ==========================================
# ANALYSIS
# ==========================================

data = pd.DataFrame({
    "distance": distances,
    "correct": correct_flags
})

correct_distances = data[
    data["correct"] == True
]["distance"]

wrong_distances = data[
    data["correct"] == False
]["distance"]


print("\n===================================")
print("TASK 7 THRESHOLD ANALYSIS")
print("===================================")

print("\nTotal questions:", len(data))

print("\nCorrect Top-1 retrievals:")
print("Count:", len(correct_distances))
print("Min:", round(correct_distances.min(), 4))
print("Mean:", round(correct_distances.mean(), 4))
print("Max:", round(correct_distances.max(), 4))

print("\nWrong Top-1 retrievals:")
print("Count:", len(wrong_distances))
print("Min:", round(wrong_distances.min(), 4))
print("Mean:", round(wrong_distances.mean(), 4))
print("Max:", round(wrong_distances.max(), 4))


# ==========================================
# THRESHOLD EXPERIMENT
# ==========================================

print("\n===================================")
print("THRESHOLD EXPERIMENT")
print("===================================")

thresholds = [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

for threshold in thresholds:

    accepted = data[
        data["distance"] <= threshold
    ]

    if len(accepted) == 0:
        print(
            f"\nThreshold {threshold:.2f}: "
            "No results accepted"
        )
        continue

    accepted_correct = accepted[
        accepted["correct"] == True
    ]

    precision = (
        len(accepted_correct)
        / len(accepted)
        * 100
    )

    coverage = (
        len(accepted)
        / len(data)
        * 100
    )

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Accepted: {len(accepted)}/{len(data)} "
        f"({coverage:.2f}%)"
    )

    print(
        f"Correct among accepted: "
        f"{len(accepted_correct)}"
    )

    print(
        f"Precision: {precision:.2f}%"
    )