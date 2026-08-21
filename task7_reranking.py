import pandas as pd
import chromadb
import numpy as np
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
# COSINE SIMILARITY
# ==========================================

def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ==========================================
# EVALUATION
# ==========================================

top1_correct = 0
top3_correct = 0
top5_correct = 0


for _, row in chapter_df.iterrows():

    question = row["Soru"]
    correct_answer = str(
        row["CEVAP METNİ"]
    ).strip()

    # Query preprocessing
    processed_question = question

    generic_patterns = [
        "Aşağıdakilerden hangisi",
        "Aşağıdaki ifadelerden hangisi",
        "Aşağıdaki seçeneklerden hangisi",
        "Aşağıdakilerin hangisinde",
        "Aşağıdaki seçeneklerden",
        "hangisi"
    ]

    for pattern in generic_patterns:
        processed_question = processed_question.replace(
            pattern,
            ""
        )

    processed_question = processed_question.strip()

    # Query embedding
    query_embedding = model.encode(
        processed_question
    )

    # Get candidates from Chroma
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=5,
        where={
            "source": "chapter_questions"
        }
    )

    documents = results["documents"][0]

    # ======================================
    # RERANK
    # ======================================

    candidate_embeddings = model.encode(
        documents
    )

    reranked = []

    for document, embedding in zip(
        documents,
        candidate_embeddings
    ):

        similarity = cosine_similarity(
            query_embedding,
            embedding
        )

        reranked.append(
            (document, similarity)
        )

    # Highest similarity first
    reranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    reranked_documents = [
        item[0]
        for item in reranked
    ]

    # ======================================
    # EVALUATION
    # ======================================

    matches = [
        correct_answer.lower()
        in document.lower()
        for document in reranked_documents
    ]

    if matches[0]:
        top1_correct += 1

    if any(matches[:3]):
        top3_correct += 1

    if any(matches[:5]):
        top5_correct += 1


# ==========================================
# RESULTS
# ==========================================

total = len(chapter_df)

top1 = top1_correct / total * 100
top3 = top3_correct / total * 100
top5 = top5_correct / total * 100


print("\n===================================")
print("TASK 7 RERANKING RESULTS")
print("===================================")

print(f"\nTop-1 Accuracy: {top1:.2f}%")
print(f"Top-3 Accuracy: {top3:.2f}%")
print(f"Top-5 Accuracy: {top5:.2f}%")

print("\nCorrect Top-1:", top1_correct)
print("Correct Top-3:", top3_correct)
print("Correct Top-5:", top5_correct)


print("\n===================================")
print("PREVIOUS BEST")
print("===================================")

print("Query preprocessing:")
print("Top-1: 65.00%")
print("Top-3: 77.50%")
print("Top-5: 78.75%")