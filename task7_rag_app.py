import os
import html
from pathlib import Path
from uuid import uuid4

import streamlit as st
import chromadb
import torch

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langfuse import get_client, propagate_attributes

load_dotenv()
langfuse = get_client()


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Bank AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

OLD_ROSE = "#D58198"
ROSEWOOD = "#AE3569"
SAND = "#A89C70"
EBONY = "#515335"
DARK_COFFEE = "#503429"

CREAM = "#FAF7F1"
WHITE = "#FFFDF9"
SOFT = "#F4EFE7"
BORDER = "#E3D8C9"


# ============================================================
# GLOBAL CSS

st.markdown("""
<style>

/* =========================
   ZIRAAT-INSPIRED BANK UI
   ========================= */

.stApp {
    background: #f5f5f5;
    color: #333333;
}

.main .block-container {
    max-width: 1280px;
    padding-top: 0;
    padding-bottom: 100px;
}

/* Hide Streamlit chrome */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* =========================
   TOP BANK HEADER
   ========================= */

.bank-header {
    width: 100%;
    background: #ffffff;
    border-bottom: 1px solid #dddddd;
    margin-bottom: 32px;
}

.bank-header-inner {
    max-width: 1280px;
    margin: 0 auto;
    padding: 18px 34px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.bank-brand {
    display: flex;
    align-items: center;
    gap: 13px;
}

.bank-logo {
    width: 42px;
    height: 42px;
    background: #e30613;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 11px;
    font-weight: 800;
}

.bank-brand-text {
    font-size: 22px;
    font-weight: 700;
    color: #333333;
    letter-spacing: -0.5px;
}

.bank-brand-sub {
    font-size: 10px;
    color: #777777;
    margin-top: 2px;
}

.bank-nav {
    display: flex;
    align-items: center;
    gap: 30px;
    font-size: 13px;
    color: #555555;
}

.bank-nav span {
    cursor: default;
}

.bank-nav .active {
    color: #e30613;
    font-weight: 700;
}


/* =========================
   HERO
   ========================= */

.hero {
    background: #ffffff;
    border-top: 5px solid #e30613;
    padding: 42px 50px 38px;
    margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.hero-small {
    color: #e30613;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
}

.hero-title {
    color: #222222;
    font-size: 38px;
    font-weight: 700;
    margin: 0;
}

.hero-description {
    color: #666666;
    font-size: 15px;
    margin-top: 9px;
}


/* =========================
   MAIN CONTENT
   ========================= */

.content-card {
    background: #ffffff;
    border: 1px solid #dddddd;
    box-shadow: 0 2px 8px rgba(0,0,0,0.035);
    padding: 28px;
}

.section-label {
    color: #e30613;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 20px;
}


/* =========================
   EMPTY CHAT
   ========================= */

.empty-state {
    min-height: 410px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}

.empty-icon {
    width: 62px;
    height: 62px;
    border: 2px solid #e30613;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #e30613;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 20px;
}

.empty-title {
    font-size: 25px;
    font-weight: 700;
    color: #333333;
}

.empty-description {
    max-width: 500px;
    margin-top: 9px;
    color: #777777;
    font-size: 14px;
    line-height: 1.7;
}


/* =========================
   USER MESSAGE
   ========================= */

.user-message {
    display: flex;
    justify-content: flex-end;
    margin: 22px 0;
}

.user-wrap {
    max-width: 76%;
}

.user-label {
    text-align: right;
    font-size: 10px;
    color: #888888;
    margin-bottom: 5px;
}

.user-bubble {
    background: #e30613;
    color: #ffffff;
    padding: 13px 19px;
    border-radius: 4px;
    font-size: 14px;
    line-height: 1.55;
}


/* =========================
   AI MESSAGE
   ========================= */

.ai-message {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin: 24px 0 12px;
}

.ai-badge {
    width: 38px;
    height: 38px;
    min-width: 38px;
    border-radius: 50%;
    background: #e30613;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    font-weight: 800;
}

.ai-bubble {
    max-width: 82%;
    background: #f8f8f8;
    border-left: 3px solid #e30613;
    color: #333333;
    padding: 15px 19px;
    font-size: 14px;
    line-height: 1.7;
}


/* =========================
   SOURCES
   ========================= */

div[data-testid="stExpander"] {
    border: 1px solid #dddddd !important;
    border-radius: 3px !important;
    background: #ffffff !important;
    margin: 10px 0 24px 50px;
}

.source-card {
    background: #ffffff;
    border: 1px solid #dddddd;
    border-top: 3px solid #e30613;
    padding: 16px;
    min-height: 175px;
}

.source-number {
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 1px;
    color: #e30613;
    margin-bottom: 9px;
}

.source-question {
    font-size: 12px;
    font-weight: 700;
    line-height: 1.5;
    color: #333333;
}

.source-answer {
    margin-top: 11px;
    font-size: 11px;
    line-height: 1.55;
    color: #666666;
}


/* =========================
   QUICK QUESTIONS
   ========================= */

.quick-title {
    font-size: 12px;
    font-weight: 800;
    color: #333333;
    margin-bottom: 16px;
}

.quick-description {
    font-size: 12px;
    line-height: 1.6;
    color: #777777;
    margin-bottom: 18px;
}

div.stButton > button {
    width: 100%;
    min-height: 45px;
    background: #ffffff;
    border: 1px solid #d5d5d5;
    border-radius: 3px;
    color: #444444;
    font-size: 11px;
    text-align: left;
    padding-left: 14px;
}

div.stButton > button:hover {
    border-color: #e30613;
    color: #e30613;
    background: #fffafa;
}


/* =========================
   CHAT INPUT
   ========================= */

div[data-testid="stChatInput"] {
    width: min(820px, 72vw);
    left: 50%;
    transform: translateX(-50%);
}

div[data-testid="stChatInput"] > div {
    background: #ffffff !important;
    border: 2px solid #e30613 !important;
    border-radius: 4px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
}

div[data-testid="stChatInput"] textarea {
    color: #333333 !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #999999 !important;
}

div[data-testid="stChatInput"] button {
    background: #e30613 !important;
}


/* =========================
   MOBILE
   ========================= */

@media (max-width: 900px) {

    .main .block-container {
        padding: 0 12px 100px;
    }

    .bank-nav {
        display: none;
    }

    .hero {
        padding: 30px 24px;
    }

    .hero-title {
        font-size: 30px;
    }

    .content-card {
        padding: 20px;
    }

    div[data-testid="stChatInput"] {
        width: 90vw;
    }
}


/* Feedback buttons */
div[data-testid="stFeedback"] {
    margin-top: 6px !important;
    margin-bottom: 12px !important;
}

div[data-testid="stFeedback"] button {
    color: #555555 !important;
    opacity: 1 !important;
}

div[data-testid="stFeedback"] button svg {
    color: #555555 !important;
    stroke: currentColor !important;
}

div[data-testid="stFeedback"] button:hover {
    color: #e30613 !important;
    background-color: #fff5f5 !important;
}

div[data-testid="stFeedback"] button:hover svg {
    color: #e30613 !important;
    stroke: currentColor !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():


    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY bulunamadı. .env dosyanı kontrol et."
        )

    client = OpenAI(api_key=api_key)

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    base_dir = Path(__file__).resolve().parent

    possible_paths = [
        base_dir / "chroma_db",
        base_dir,
        base_dir.parent / "chroma_db",
    ]

    collection = None

    for db_path in possible_paths:

        try:

            chroma_client = chromadb.PersistentClient(
                path=str(db_path)
            )

            names = chroma_client.list_collections()

            collection_names = [
                x.name if hasattr(x, "name") else str(x)
                for x in names
            ]

            if "banking_knowledge" in collection_names:

                collection = chroma_client.get_collection(
                    "banking_knowledge"
                )

                break

        except Exception:
            continue

    if collection is None:
        raise RuntimeError(
            "banking_knowledge collection bulunamadı."
        )

    reranker_name = "BAAI/bge-reranker-v2-m3"

    tokenizer = AutoTokenizer.from_pretrained(
        reranker_name
    )

    reranker = AutoModelForSequenceClassification.from_pretrained(
        reranker_name
    )

    reranker.eval()

    return (
        client,
        embedding_model,
        collection,
        tokenizer,
        reranker
    )


try:

    (
        openai_client,
        embedding_model,
        collection,
        reranker_tokenizer,
        reranker_model
    ) = load_models()

except Exception as error:

    st.error(str(error))
    st.stop()


# ============================================================
# QUERY FUNCTIONS
# ============================================================

def preprocess_question(question):

    patterns = [
        "Aşağıdakilerden hangisi",
        "Aşağıdaki ifadelerden hangisi",
        "Aşağıdaki seçeneklerden hangisi",
        "Aşağıdakilerin hangisinde",
        "Aşağıdaki seçeneklerden",
        "hangisi"
    ]

    processed = str(question)

    for pattern in patterns:
        processed = processed.replace(
            pattern,
            ""
        )

    return processed.strip()


def generate_queries(question):

    with langfuse.start_as_current_observation(
        as_type="span",
        name="query-generation",
        input={"question": question}
    ) as observation:

        processed = preprocess_question(question)

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

            if query and query not in unique_queries:
                unique_queries.append(query)

        observation.update(
            output={"queries": unique_queries}
        )

        return unique_queries


# ============================================================
# RERANKER
# ============================================================

def rerank_documents(question, documents):

    if not documents:
        return []

    with langfuse.start_as_current_observation(
        as_type="span",
        name="reranking",
        input={
            "question": question,
            "candidate_count": len(documents)
        }
    ) as observation:

        pairs = [
            [question, document]
            for document in documents
        ]

        inputs = reranker_tokenizer(
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

        scores = outputs.logits.view(-1).tolist()

        ranked = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True
        )

        observation.update(
            output={
                "candidate_count": len(documents),
                "top_scores": [
                    float(score)
                    for score, _ in ranked[:3]
                ]
            }
        )

        return ranked


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(question, top_k=3):

    with langfuse.start_as_current_observation(
        as_type="retriever",
        name="retrieval",
        input={
            "question": question,
            "top_k": top_k
        }
    ) as observation:

        queries = generate_queries(question)

        candidate_documents = []

        for query in queries:

            query_embedding = embedding_model.encode(
                query
            ).tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=30,
                where={
                    "source": "question_answers"
                }
            )

            for document in results["documents"][0]:

                if document not in candidate_documents:
                    candidate_documents.append(document)

        candidate_documents = candidate_documents[:30]

        ranked = rerank_documents(
            question,
            candidate_documents
        )

        final_results = ranked[:top_k]

        observation.update(
            output={
                "query_count": len(queries),
                "candidate_count": len(candidate_documents),
                "returned_count": len(final_results)
            }
        )

        return final_results


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question):

    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-generation",
        input={"question": question}
    ) as trace:

        results = retrieve(question)

        documents = [
            document
            for score, document in results
        ]

        source_items = [
            {
                "text": document,
                "score": float(score)
            }
            for score, document in results
        ]

        if not documents:

            answer = "Bu bilgi bilgi tabanında bulunamadı."

            trace.update(
                output={
                    "answer": answer,
                    "sources": []
                }
            )

            return answer, []

        context = "\\n\\n".join(documents)

        prompt = f"""
Sen Türkçe çalışan bir finans ve bankacılık bilgi asistanısın.

SADECE aşağıdaki KAYNAK BİLGİYİ kullan.

Sorunun cevabı kaynaklarda bulunmuyorsa:

Bu bilgi bilgi tabanında bulunamadı.

cevabını ver.

Bilgi uydurma.

KAYNAK BİLGİ:

{context}

KULLANICI SORUSU:

{question}

Kısa, açık ve doğru bir Türkçe cevap ver.
"""

        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm-generation",
            input={
                "question": question,
                "context": context,
                "model": "gpt-5-mini"
            }
        ) as generation:

            response = openai_client.responses.create(
                model="gpt-5-mini",
                input=prompt
            )

            answer = response.output_text.strip()

            generation.update(
                output={
                    "answer": answer
                },
                model="gpt-5-mini"
            )

        trace.update(
            output={
                "answer": answer,
                "sources": source_items
            }
        )

        return (
            answer,
            source_items
        )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

if "langfuse_session_id" not in st.session_state:
    st.session_state.langfuse_session_id = str(uuid4())


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<style>

/* ============================================================
   ZIRAAT-INSPIRED BANK AI UI
   ============================================================ */

.stApp {
    background: #f4f4f4;
    color: #2f2f2f;
}

.main .block-container {
    max-width: 1180px;
    padding-top: 0.8rem;
    padding-bottom: 7rem;
}

#MainMenu,
footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* ============================================================
   HEADER
   ============================================================ */

.bank-header {
    background: #ffffff;
    border-bottom: 4px solid #E30613;
    padding: 17px 28px 15px;
    margin: -1rem -1rem 0.8rem;
}

.bank-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.bank-logo {
    width: 40px;
    height: 40px;
    min-width: 40px;
    border-radius: 50%;
    background: #E30613;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .3px;
}

.bank-name {
    color: #252525;
    font-size: 20px;
    font-weight: 750;
    line-height: 1.1;
}

.bank-subtitle {
    color: #777777;
    font-size: 10px;
    margin-top: 4px;
}


/* ============================================================
   NAVIGATION
   ============================================================ */

.nav-wrap {
    margin: 0 0 1.5rem;
}

div[data-testid="stHorizontalBlock"] .nav-button button {
    border: none !important;
    background: transparent !important;
    color: #555555 !important;
    border-radius: 0 !important;
    min-height: 38px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-align: center !important;
    box-shadow: none !important;
}

div[data-testid="stHorizontalBlock"] .nav-button button:hover {
    color: #E30613 !important;
    background: #ffffff !important;
}

.nav-active {
    color: #E30613;
    font-size: 12px;
    font-weight: 800;
    text-align: center;
    padding: 10px 4px;
    border-bottom: 2px solid #E30613;
}


/* ============================================================
   HERO
   ============================================================ */

.hero-card {
    background: #ffffff;
    border: 1px solid #e2e2e2;
    border-top: 5px solid #E30613;
    padding: 34px 42px 31px;
    margin-bottom: 24px;
    box-shadow: 0 5px 18px rgba(0,0,0,.045);
}

.hero-label {
    color: #E30613;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.7px;
    margin-bottom: 9px;
}

.hero-title {
    color: #242424;
    font-size: 35px;
    font-weight: 750;
    letter-spacing: -.7px;
    line-height: 1.15;
}

.hero-text {
    color: #6c6c6c;
    font-size: 13px;
    line-height: 1.7;
    margin-top: 9px;
    max-width: 760px;
}


/* ============================================================
   CONTENT CARDS
   ============================================================ */

.panel {
    background: #ffffff;
    border: 1px solid #dfdfdf;
    box-shadow: 0 4px 15px rgba(0,0,0,.035);
    padding: 25px;
}

.panel-label {
    color: #E30613;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.3px;
    margin-bottom: 7px;
}

.panel-title {
    color: #303030;
    font-size: 21px;
    font-weight: 750;
    margin-bottom: 5px;
}

.panel-text {
    color: #777777;
    font-size: 12px;
    line-height: 1.65;
}


/* ============================================================
   EMPTY ASSISTANT STATE
   ============================================================ */

.empty-state {
    min-height: 330px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}

.empty-badge {
    width: 56px;
    height: 56px;
    border: 2px solid #E30613;
    border-radius: 50%;
    color: #E30613;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 800;
    margin-bottom: 17px;
}

.empty-title {
    color: #303030;
    font-size: 24px;
    font-weight: 750;
}

.empty-text {
    color: #777777;
    font-size: 12px;
    line-height: 1.7;
    max-width: 470px;
    margin-top: 7px;
}


/* ============================================================
   CHAT MESSAGES
   ============================================================ */

.user-message {
    display: flex;
    justify-content: flex-end;
    margin: 18px 0;
}

.user-bubble {
    max-width: 78%;
    background: #E30613;
    color: #ffffff;
    padding: 12px 17px;
    border-radius: 4px;
    font-size: 13px;
    line-height: 1.6;
}

.ai-message {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 18px 0 10px;
}

.ai-badge {
    width: 34px;
    height: 34px;
    min-width: 34px;
    border-radius: 50%;
    background: #E30613;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 8px;
    font-weight: 800;
}

.ai-bubble {
    max-width: 82%;
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-left: 3px solid #E30613;
    color: #333333;
    padding: 13px 17px;
    font-size: 13px;
    line-height: 1.7;
}


/* ============================================================
   SOURCES
   ============================================================ */

div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #dedede !important;
    border-left: 3px solid #E30613 !important;
    border-radius: 4px !important;
    margin: 8px 0 18px 44px !important;
}

.source-heading {
    color: #E30613;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.source-question {
    color: #333333;
    font-size: 12px;
    font-weight: 700;
    line-height: 1.55;
}

.source-answer {
    color: #707070;
    font-size: 11px;
    line-height: 1.6;
}

.source-score {
    display: inline-block;
    margin-top: 10px;
    padding: 4px 8px;
    border: 1px solid #e30613;
    border-radius: 3px;
    color: #e30613;
    background: #fffafa;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.2px;
}


/* ============================================================
   QUICK QUESTIONS
   ============================================================ */

.quick-card {
    background: #ffffff;
    border: 1px solid #dedede;
    border-top: 4px solid #E30613;
    padding: 22px;
    box-shadow: 0 4px 15px rgba(0,0,0,.035);
}

.quick-title {
    color: #303030;
    font-size: 19px;
    font-weight: 750;
}

.quick-text {
    color: #777777;
    font-size: 11px;
    line-height: 1.65;
    margin: 6px 0 16px;
}

div.stButton > button {
    background: #ffffff;
    color: #444444;
    border: 1px solid #d5d5d5;
    border-radius: 4px;
    min-height: 43px;
    font-size: 11px;
    text-align: left;
    padding: 8px 13px;
    box-shadow: none;
}

div.stButton > button:hover {
    color: #E30613;
    border-color: #E30613;
    background: #fffafa;
}


/* ============================================================
   CATEGORY CARDS
   ============================================================ */

.category-card {
    background: #ffffff;
    border: 1px solid #dddddd;
    border-top: 3px solid #E30613;
    padding: 22px;
    min-height: 175px;
    box-shadow: 0 3px 12px rgba(0,0,0,.035);
}

.category-number {
    color: #E30613;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.category-title {
    color: #303030;
    font-size: 18px;
    font-weight: 750;
    margin-top: 10px;
}

.category-description {
    color: #777777;
    font-size: 12px;
    line-height: 1.65;
    margin-top: 7px;
}


/* ============================================================
   CHAT INPUT
   ============================================================ */

div[data-testid="stChatInput"] {
    max-width: 820px;
}

div[data-testid="stChatInput"] > div {
    background: #ffffff !important;
    border: 2px solid #E30613 !important;
    border-radius: 5px !important;
    box-shadow: 0 4px 18px rgba(0,0,0,.10) !important;
}

div[data-testid="stChatInput"] textarea {
    color: #333333 !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #999999 !important;
}

div[data-testid="stChatInput"] button {
    background: #E30613 !important;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 900px) {

    .main .block-container {
        padding-left: 14px;
        padding-right: 14px;
    }

    .hero-card {
        padding: 28px 24px;
    }

    .hero-title {
        font-size: 28px;
    }

    .bank-header {
        margin-left: -0.5rem;
        margin-right: -0.5rem;
    }

    div[data-testid="stExpander"] {
        margin-left: 0 !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

header_col, new_chat_col = st.columns([5, 1])

with header_col:
    st.markdown("""
    <div class="bank-header">
        <div class="bank-brand">
            <div class="bank-logo">ZB</div>
            <div>
                <div class="bank-name">Ziraat Bankası</div>
                <div class="bank-subtitle">Yapay Zeka Bilgi Asistanı</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with new_chat_col:
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    if st.button(
        "Yeni Sohbet",
        key="new_chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.selected_question = None
        st.session_state.active_page = "assistant"
        st.rerun()


# ============================================================
# NAVIGATION
# ============================================================

if "active_page" not in st.session_state:
    st.session_state.active_page = "assistant"

nav1, nav2, nav3, nav4, nav5 = st.columns([1.5, 1.5, 1.2, 1.2, 1.5])

with nav1:
    if st.session_state.active_page == "assistant":
        st.markdown('<div class="nav-active">Bilgi Asistanı</div>', unsafe_allow_html=True)
    elif st.button("Bilgi Asistanı", key="nav_assistant", use_container_width=True):
        st.session_state.active_page = "assistant"
        st.rerun()

with nav2:
    if st.session_state.active_page == "banking":
        st.markdown('<div class="nav-active">Bankacılık</div>', unsafe_allow_html=True)
    elif st.button("Bankacılık", key="nav_banking", use_container_width=True):
        st.session_state.active_page = "banking"
        st.rerun()

with nav3:
    if st.session_state.active_page == "finance":
        st.markdown('<div class="nav-active">Finans</div>', unsafe_allow_html=True)
    elif st.button("Finans", key="nav_finance", use_container_width=True):
        st.session_state.active_page = "finance"
        st.rerun()

with nav4:
    if st.session_state.active_page == "about":
        st.markdown('<div class="nav-active">Hakkında</div>', unsafe_allow_html=True)
    elif st.button("Hakkında", key="nav_about", use_container_width=True):
        st.session_state.active_page = "about"
        st.rerun()

with nav5:
    st.markdown(
        '<div style="text-align:right;color:#999;font-size:10px;padding-top:12px;">BANK AI · RAG</div>',
        unsafe_allow_html=True
    )


# ============================================================
# ASSISTANT PAGE
# ============================================================

if st.session_state.active_page == "assistant":

    st.markdown("""
    <div class="hero-card">
        <div class="hero-label">ZİRAAT BANKASI · YAPAY ZEKA</div>
        <div class="hero-title">Size nasıl yardımcı olabiliriz?</div>
        <div class="hero-text">
            Bankacılık ve finans hakkında merak ettiğiniz soruları
            bilgi tabanımıza dayalı olarak yanıtlayın.
        </div>
    </div>
    """, unsafe_allow_html=True)

    chat_column, quick_column = st.columns([2.65, 1.15], gap="large")

    with chat_column:

        st.markdown(
            '<div class="panel-label">BİLGİ ASİSTANI</div>',
            unsafe_allow_html=True
        )

        if not st.session_state.messages:

            st.markdown("""
            <div class="panel">
                <div class="empty-state">
                    <div class="empty-badge">AI</div>
                    <div class="empty-title">Sorunuzu yazın</div>
                    <div class="empty-text">
                        Bankacılık ve finans hakkında bir soru sorun.
                        Yanıtlar bilgi tabanındaki kaynaklara dayanır.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown('<div class="panel">', unsafe_allow_html=True)

            for message_index, message in enumerate(st.session_state.messages):

                role = message["role"]
                content = html.escape(str(message["content"]))

                if role == "user":

                    st.markdown(
                        f"""
                        <div class="user-message">
                            <div class="user-bubble">{content}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="ai-message">
                            <div class="ai-badge">AI</div>
                            <div class="ai-bubble">{content}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # ====================================================
                    # FEEDBACK
                    # ====================================================

                    feedback = st.feedback(
                        "thumbs",
                        key=f"feedback_{message_index}"
                    )

                    if feedback is not None:
                        message["feedback"] = (
                            "positive"
                            if feedback == 1
                            else "negative"
                        )

                    sources = message.get("sources", [])

                    if sources:

                        with st.expander(
                            f"Yanıt kaynakları · {len(sources)}",
                            expanded=False
                        ):

                            st.markdown(
                                '<div class="sources-intro">'
                                'Yanıt, aşağıdaki bilgi tabanı kaynakları kullanılarak oluşturuldu.'
                                '</div>',
                                unsafe_allow_html=True
                            )

                            for i, source in enumerate(sources, start=1):

                                if isinstance(source, dict):
                                    source_text = str(source.get("text", ""))
                                    reranker_score = source.get("score")
                                else:
                                    source_text = str(source)
                                    reranker_score = None

                                import re

                                # Kaynak metninde kalmış HTML etiketlerini temizle
                                source_text = re.sub(
                                    r"<[^>]*>",
                                    " ",
                                    source_text
                                )

                                # HTML entity'lerini çöz
                                source_text = html.unescape(
                                    source_text
                                )

                                # Fazla boşlukları temizle
                                source_text = re.sub(
                                    r"\\s+",
                                    " ",
                                    source_text
                                ).strip()

                                # Question / Answer ayrımı
                                parts = source_text.split(
                                    "Answer:",
                                    1
                                )

                                question_text = (
                                    parts[0]
                                    .replace("Question:", "")
                                    .strip()
                                )

                                answer_text_source = (
                                    parts[1].strip()
                                    if len(parts) > 1
                                    else ""
                                )

                                question_html = html.escape(
                                    question_text
                                )

                                answer_html = html.escape(
                                    answer_text_source
                                )

                                # ÖNEMLİ:
                                # HTML stringi satır başından itibaren
                                # girintisiz oluşturuluyor.
                                # Böylece Streamlit bunu code block
                                # olarak algılamıyor.

                                if reranker_score is not None:
                                    score_html = (
                                        f'<div class="source-score">'
                                        f'Reranker skoru: {float(reranker_score):.4f}'
                                        f'</div>'
                                    )
                                else:
                                    score_html = ""

                                source_html = (
                                    '<div class="source-card">'
                                    f'<div class="source-number">KAYNAK {i:02d}</div>'
                                    f'<div class="source-question">{question_html}</div>'
                                    + (
                                        f'<div class="source-answer">{answer_html}</div>'
                                        if answer_html
                                        else ""
                                    )
                                    + score_html
                                    + '</div>'
                                )

                                st.markdown(
                                    source_html,
                                    unsafe_allow_html=True
                                )


    with quick_column:

        st.markdown("""
        <div class="quick-card">
            <div class="quick-title">Sık Sorulan Sorular</div>
            <div class="quick-text">
                Hazır sorulardan birini seçerek bilgi asistanıyla başlayabilirsiniz.
            </div>
        </div>
        """, unsafe_allow_html=True)

        suggestions = [
            "Basel-II'nin amacı nedir?",
            "Bankaların sermaye yeterliliği nedir?",
            "Merkez bankasının görevleri nelerdir?",
            "Bankacılıkta risk nedir?"
        ]

        for i, suggestion in enumerate(suggestions):

            if st.button(
                suggestion,
                key=f"suggestion_{i}",
                use_container_width=True
            ):
                st.session_state.selected_question = suggestion
                st.session_state.active_page = "assistant"
                st.rerun()


# ============================================================
# BANKING PAGE
# ============================================================

elif st.session_state.active_page == "banking":

    st.markdown("""
    <div class="hero-card">
        <div class="hero-label">ZİRAAT BANKASI · BANKACILIK</div>
        <div class="hero-title">Bankacılık hakkında bilgi edinin.</div>
        <div class="hero-text">
            Bankacılık sisteminin temel kavramlarını ve kurumların
            işleyişini bilgi tabanındaki içeriklere dayanarak keşfedin.
        </div>
    </div>
    """, unsafe_allow_html=True)

    banking_cards = [
        (
            "01",
            "Bankacılık Sistemi",
            "Bankaların finansal sistem içindeki rolü, temel bankacılık faaliyetleri ve bankacılık yapısı.",
            "Bankaların temel görevleri nelerdir?"
        ),
        (
            "02",
            "Merkez Bankacılığı",
            "Merkez bankalarının görevleri, para politikası ve finansal sistemdeki temel işlevleri.",
            "Merkez bankasının görevleri nelerdir?"
        ),
        (
            "03",
            "Bankacılık Faaliyetleri",
            "Mevduat, kredi, kiralık kasa ve diğer bankacılık hizmetleri hakkında temel bilgiler.",
            "Bankaların temel faaliyetleri nelerdir?"
        ),
        (
            "04",
            "Bankacılık Riskleri",
            "Bankacılık sektöründe karşılaşılan riskler ve risklerin yönetilmesine ilişkin temel kavramlar.",
            "Bankacılıkta risk nedir?"
        ),
    ]

    cols = st.columns(2, gap="large")

    for i, card in enumerate(banking_cards):

        with cols[i % 2]:

            number, title, description, question_text = card

            st.markdown(
                f"""
                <div class="category-card">
                    <div class="category-number">{number}</div>
                    <div class="category-title">{title}</div>
                    <div class="category-description">{description}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Asistana sor",
                key=f"banking_card_{i}",
                use_container_width=True
            ):
                st.session_state.selected_question = question_text
                st.session_state.active_page = "assistant"
                st.rerun()


# ============================================================
# FINANCE PAGE
# ============================================================

elif st.session_state.active_page == "finance":

    st.markdown("""
    <div class="hero-card">
        <div class="hero-label">ZİRAAT BANKASI · FİNANS</div>
        <div class="hero-title">Finansal konuları keşfedin.</div>
        <div class="hero-text">
            Finansal sistemler, risk yönetimi, finansal piyasalar ve
            bankacılık düzenlemeleri hakkında bilgi tabanına dayalı
            içerikleri inceleyin.
        </div>
    </div>
    """, unsafe_allow_html=True)

    finance_cards = [
        (
            "01",
            "Finansal Sistemler",
            "Finansal sistemin yapısı, fon arzı ve talebi ile finansal kurumların temel rolleri.",
            "Finansal sistem nedir?"
        ),
        (
            "02",
            "Risk Yönetimi",
            "Bankacılık riskleri, sermaye yeterliliği ve risklerin ölçülmesine ilişkin temel kavramlar.",
            "Bankaların sermaye yeterliliği neden önemlidir?"
        ),
        (
            "03",
            "Finansal Piyasalar",
            "Finansal piyasaların yapısı ve bankacılık ile finans arasındaki temel ilişkiler.",
            "Finansal piyasalar nedir?"
        ),
        (
            "04",
            "Düzenlemeler ve Standartlar",
            "Basel I, Basel II, sermaye yeterliliği ve bankacılık sektöründeki düzenleyici standartlar.",
            "Basel-II'nin amacı nedir?"
        ),
    ]

    cols = st.columns(2, gap="large")

    for i, card in enumerate(finance_cards):

        with cols[i % 2]:

            number, title, description, question_text = card

            st.markdown(
                f"""
                <div class="category-card">
                    <div class="category-number">{number}</div>
                    <div class="category-title">{title}</div>
                    <div class="category-description">{description}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Bu konuyu asistana sor",
                key=f"finance_card_{i}",
                use_container_width=True
            ):
                st.session_state.selected_question = question_text
                st.session_state.active_page = "assistant"
                st.rerun()


# ============================================================
# ABOUT PAGE
# ============================================================

elif st.session_state.active_page == "about":

    st.markdown("""
    <div class="hero-card">
        <div class="hero-label">BANK AI · RAG SİSTEMİ</div>
        <div class="hero-title">Bilgi tabanına dayalı yapay zeka asistanı.</div>
        <div class="hero-text">
            Bankacılık ve finans sorularını bilgi tabanından ilgili
            içerikleri getirerek yanıtlayan bir Retrieval-Augmented
            Generation (RAG) uygulamasıdır.
        </div>
    </div>
    """, unsafe_allow_html=True)

    about_cols = st.columns(3, gap="large")

    about_items = [
        (
            "RETRIEVAL",
            "Bilgi tabanından soruyla ilişkili içerikler bulunur."
        ),
        (
            "RERANKING",
            "Bulunan adaylar BGE Cross-Encoder ile yeniden sıralanır."
        ),
        (
            "GENERATION",
            "Yanıt yalnızca getirilen kaynak bilgisi kullanılarak oluşturulur."
        ),
    ]

    for col, (title, description) in zip(about_cols, about_items):

        with col:

            st.markdown(
                f"""
                <div class="category-card">
                    <div class="category-number">{title}</div>
                    <div class="category-description" style="margin-top:14px;">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# INPUT
# ============================================================

question = None

if st.session_state.active_page == "assistant":

    typed_question = st.chat_input(
        "Bankacılık veya finans hakkında sorun..."
    )

    question = typed_question

    if question is None:

        question = st.session_state.selected_question

        st.session_state.selected_question = None


# ============================================================
# PROCESS
# ============================================================

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.spinner(
        "Bilgi tabanında aranıyor..."
    ):
        with propagate_attributes(
            session_id=st.session_state.langfuse_session_id
        ):
            answer, sources = generate_answer(
                question
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )

    st.rerun()
