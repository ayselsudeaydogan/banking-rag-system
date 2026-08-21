import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


load_dotenv()

# OpenAI
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Chroma
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    "banking_knowledge"
)


def preprocess_question(question):

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


def retrieve(question, top_k=3):

    processed_question = preprocess_question(question)

    query_embedding = embedding_model.encode(
        processed_question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"source": "question_answers"}
    )

    return results


def generate_answer(question):

    results = retrieve(question)

    documents = results["documents"][0]
    distances = results["distances"][0]

    print("\nRetrieved documents:")

    for i, (document, distance) in enumerate(
        zip(documents, distances), 1
    ):
        print(f"\nSource {i}")
        print("Distance:", round(distance, 4))
        print(document)

    # Use only sufficiently relevant results
    relevant_documents = []

    DISTANCE_THRESHOLD = 0.80

    for document, distance in zip(
        documents,
        distances
    ):
        if distance <= DISTANCE_THRESHOLD:
            relevant_documents.append(document)

    if not relevant_documents:

        return (
            "Bu soruyla ilgili yeterli bilgi "
            "bilgi tabanında bulunamadı."
        )

    context = "\n\n".join(
        relevant_documents
    )

    prompt = f"""
Sen Türkçe çalışan bir finans ve bankacılık bilgi asistanısın.

Aşağıdaki KAYNAK BİLGİ dışında hiçbir bilgi kullanma.

Eğer sorunun cevabı kaynaklarda bulunmuyorsa:

"Bu bilgi bilgi tabanında bulunamadı."

cevabını ver.

Bilgi uydurma.

KAYNAK BİLGİ:
{context}

KULLANICI SORUSU:
{question}

Kısa, açık ve doğru bir Türkçe cevap ver.
"""

    response = openai_client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text


print("\n===================================")
print("TASK 7 RAG GROUNDING TEST")
print("===================================")

while True:

    question = input(
        "\nSorunuzu yazın (çıkmak için q): "
    )

    if question.lower() == "q":
        break

    answer = generate_answer(question)

    print("\n===================================")
    print("ANSWER")
    print("===================================")

    print(answer)