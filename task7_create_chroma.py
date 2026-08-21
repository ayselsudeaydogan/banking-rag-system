import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

print("Loading datasets...")

# Load cleaned Q&A dataset
qa_df = pd.read_excel("question_answers_clean.xlsx")

# Load chapter-end questions
chapter_df = pd.read_excel("BÖLÜM_SONU_SORULARI.xlsx")

print("Q&A records:", len(qa_df))
print("Chapter questions:", len(chapter_df))

# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

# Create / reset collection
try:
    client.delete_collection("banking_knowledge")
except Exception:
    pass

collection = client.create_collection(
    name="banking_knowledge",
    metadata={"description": "Banking RAG knowledge base"}
)

print("\nCreating Q&A documents...")

qa_documents = []
qa_ids = []
qa_metadatas = []

for i, row in qa_df.iterrows():
    document = f"Question: {row['question']}\nAnswer: {row['answer']}"

    qa_documents.append(document)
    qa_ids.append(f"qa_{i}")
    qa_metadatas.append({
        "source": "question_answers",
        "type": "qa"
    })

print("Creating chapter-question documents...")

for i, row in chapter_df.iterrows():
    document = (
        f"Question: {row['Soru']}\n"
        f"A: {row['A']}\n"
        f"B: {row['B']}\n"
        f"C: {row['C']}\n"
        f"D: {row['D']}\n"
        f"E: {row['E']}\n"
        f"Correct answer: {row['CEVAP ŞIKKI']} - {row['CEVAP METNİ']}"
    )

    qa_documents.append(document)
    qa_ids.append(f"chapter_{i}")
    qa_metadatas.append({
        "source": "chapter_questions",
        "type": "multiple_choice"
    })

print("\nTotal documents:", len(qa_documents))

print("\nCreating embeddings...")
embeddings = model.encode(
    qa_documents,
    show_progress_bar=True
).tolist()

print("\nAdding documents to Chroma...")

collection.add(
    ids=qa_ids,
    documents=qa_documents,
    embeddings=embeddings,
    metadatas=qa_metadatas
)

print("\n===================================")
print("CHROMA DATABASE CREATED")
print("===================================")

print("Collection:", collection.name)
print("Total documents:", collection.count())
print("Database path: ./chroma_db")

