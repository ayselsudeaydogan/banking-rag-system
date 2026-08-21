import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ==========================================
# SETUP
# ==========================================

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_collection(
    "banking_knowledge"
)


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

    processed = question

    for pattern in generic_patterns:
        processed = processed.replace(pattern, "")

    return processed.strip()


# ==========================================
# RETRIEVAL
# ==========================================

def retrieve_context(question, top_k=3):

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


# ==========================================
# RAG GENERATION
# ==========================================

def generate_answer(question):

    results = retrieve_context(question)

    documents = results["documents"][0]

    context = "\n\n".join(documents)

    prompt = f"""
Sen bir finans ve bankacılık bilgi asistanısın.

Sadece aşağıda verilen bilgiye dayanarak cevap ver.
Bilgi yeterli değilse bunu açıkça belirt.
Bilgi uydurma.

KAYNAK BİLGİ:
{context}

KULLANICI SORUSU:
{question}

Kısa, açık ve doğru bir Türkçe cevap ver.
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text, documents


# ==========================================
# TEST
# ==========================================

print("\n===================================")
print("TASK 7 RAG GENERATION")
print("===================================")

question = input("\nSorunuzu yazın: ")

answer, sources = generate_answer(question)

print("\n===================================")
print("ANSWER")
print("===================================")

print(answer)

print("\n===================================")
print("RETRIEVED SOURCES")
print("===================================")

for i, source in enumerate(sources, 1):
    print(f"\nSource {i}")
    print("-----------------------------------")
    print(source)