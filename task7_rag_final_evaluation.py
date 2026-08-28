import pandas as pd
import chromadb
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================================
# SETUP
# ==========================================

print("\n===================================")
print("TASK 7 FINAL RAG EVALUATION")
print("===================================")

test_df = pd.read_excel(
    "BÖLÜM_SONU_SORULARI.xlsx"
)

print("\nTest questions:", len(test_df))


# ==========================================
# QUERY PREPROCESSING
# ==========================================

def preprocess_question(question):

    generic_patterns = [
        "Aşağıdakilerden hangisi",
        "Aşağıdaki ifadelerden hangisi",
        "Aşağıdaki seçeneklerden hangisi",
        "Aşağıdakilerin hangisinde",
        "Aşağıdaki seçeneklerden",
        "hangisi"
    ]

    processed = str(question)

    for pattern in generic_patterns:
        processed = processed.replace(
            pattern,
            ""
        )

    return processed.strip()


# ==========================================
# EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# CHROMA
# ==========================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "banking_knowledge"
)

print(
    "Collection:",
    collection.name
)

print(
    "Documents:",
    collection.count()
)


# ==========================================
# BGE CROSS-ENCODER
# ==========================================

print("\nLoading BGE reranker...")

reranker_name = "BAAI/bge-reranker-v2-m3"

tokenizer = AutoTokenizer.from_pretrained(
    reranker_name
)

reranker_model = AutoModelForSequenceClassification.from_pretrained(
    reranker_name
)

reranker_model.eval()


def rerank_documents(
    query,
    documents
):

    pairs = [
        [query, document]
        for document in documents
    ]

    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = reranker_model(
            **inputs
        )

    scores = (
        outputs.logits
        .view(-1)
        .tolist()
    )

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked


# ==========================================
# MULTI-QUERY
# ==========================================

def generate_queries(question):

    processed = preprocess_question(
        question
    )

    queries = [
        processed,
        question,
        f"{processed} temel bilgi",
        f"{processed} açıklaması",
        f"{processed} cevabı"
    ]

    unique_queries = []

    for query in queries:

        query = query.strip()

        if (
            query
            and query not in unique_queries
        ):
            unique_queries.append(query)

    return unique_queries


# ==========================================
# MULTI-QUERY + UNION + CROSS-ENCODER
# ==========================================

def retrieve(
    question,
    candidate_k=30
):

    queries = generate_queries(
        question
    )

    candidate_documents = []

    # --------------------------------------
    # Multi-Query Retrieval
    # --------------------------------------

    for query in queries:

        query_embedding = (
            embedding_model
            .encode(query)
            .tolist()
        )

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=candidate_k,
            where={
                "source":
                "chapter_questions"
            }
        )

        # ----------------------------------
        # Union
        # ----------------------------------

        for document in (
            results["documents"][0]
        ):

            if (
                document
                not in candidate_documents
            ):
                candidate_documents.append(
                    document
                )

    # --------------------------------------
    # Cross-Encoder Reranking
    # --------------------------------------

    ranked = rerank_documents(
        question,
        candidate_documents
    )

    return ranked


# ==========================================
# EVALUATION
# ==========================================

candidate_recall = 0
top1_correct = 0
top3_correct = 0
top5_correct = 0

total = len(test_df)

print("\n===================================")
print("RUNNING FINAL EVALUATION")
print("===================================")


for index, row in test_df.iterrows():

    question = str(
        row["Soru"]
    )

    correct_answer = str(
        row["CEVAP METNİ"]
    ).strip().lower()

    ranked_results = retrieve(
        question
    )

    # --------------------------------------
    # Candidate Recall@30
    # --------------------------------------

    candidate_documents = [
        document
        for score, document
        in ranked_results[:30]
    ]

    candidate_found = any(
        correct_answer
        in document.lower()
        for document
        in candidate_documents
    )

    if candidate_found:
        candidate_recall += 1

    # --------------------------------------
    # Top-K Evaluation
    # --------------------------------------

    ranked_documents = [
        document
        for score, document
        in ranked_results
    ]

    matches = [
        correct_answer
        in document.lower()
        for document
        in ranked_documents
    ]

    if matches:
        if matches[0]:
            top1_correct += 1

        if any(matches[:3]):
            top3_correct += 1

        if any(matches[:5]):
            top5_correct += 1

    print(
        f"\rProgress: "
        f"{index + 1}/{total}",
        end=""
    )


# ==========================================
# RESULTS
# ==========================================

print("\n")

print("===================================")
print("FINAL RAG RESULTS")
print("===================================")

print(
    f"Candidate Recall@30: "
    f"{candidate_recall / total * 100:.2f}%"
)

print(
    f"Top-1 Accuracy: "
    f"{top1_correct / total * 100:.2f}%"
)

print(
    f"Top-3 Accuracy: "
    f"{top3_correct / total * 100:.2f}%"
)

print(
    f"Top-5 Accuracy: "
    f"{top5_correct / total * 100:.2f}%"
)

print("\nCorrect Candidate Recall:",
      candidate_recall)

print("Correct Top-1:",
      top1_correct)

print("Correct Top-3:",
      top3_correct)

print("Correct Top-5:",
      top5_correct)


# ==========================================
# BASELINE COMPARISON
# ==========================================

print("\n===================================")
print("REFERENCE RESULT")
print("===================================")

print(
    "Previous best:"
)

print(
    "Candidate Recall@30: 96.25%"
)

print(
    "Top-1: 95.00%"
)

print(
    "Top-3: 96.25%"
)

print(
    "Top-5: 96.25%"
)

print("\n===================================")
print("DONE")
print("===================================")
