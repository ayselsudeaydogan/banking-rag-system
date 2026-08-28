# Banking RAG System

A Retrieval-Augmented Generation (RAG) system developed during my AI internship to provide grounded answers to banking and insurance-related questions.

## Overview

This project combines semantic embeddings, ChromaDB, and an LLM to retrieve relevant information from a structured banking knowledge base and generate context-aware answers.

## RAG Pipeline

User Question
→ Query Preprocessing
→ Embedding
→ ChromaDB Retrieval
→ Context Filtering
→ LLM Generation
→ Grounded Answer

## Technologies

- Python
- all-MiniLM-L6-v2
- ChromaDB
- OpenAI API
- GPT-5-mini
- Pandas
- Sentence Transformers

## Dataset

The knowledge base consists of structured banking and insurance question-answer data and chapter-end questions.

A total of 1,322 documents were indexed in ChromaDB.

## Retrieval Evaluation

The retrieval pipeline was evaluated progressively:

| Method | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Baseline | 26.25% | 42.50% | 51.25% |
| Metadata Filtering | 60.00% | 76.25% | 80.00% |
| Query Preprocessing | **65.00%** | **77.50%** | 78.75% |
| Same-Model Reranking | 65.00% | 77.50% | 78.75% |

Query preprocessing currently provides the best Top-1 retrieval performance.

## Grounding

The system includes a grounding mechanism to prevent unsupported answers.

When relevant information cannot be found in the knowledge base, the system responds:

> "Bu bilgi bilgi tabanında bulunamadı."

This helps reduce hallucinated answers.

## Current Status

Completed:

- Data analysis and quality checks
- Data preparation
- Embedding generation
- ChromaDB vector database
- Semantic retrieval
- Metadata filtering
- Query preprocessing
- Retrieval evaluation
- Error analysis
- Threshold analysis
- RAG answer generation
- Grounding tests
- Initial reranking experiment

Planned:

- Multilingual cross-encoder reranking
- Retrieval performance optimization
- Improved prompt and context selection
- Gradio-based user interface
- Final system evaluation

## Project Structure

```text
banking-rag-system/
├── README.md
├── .gitignore
├── task7_data_analysis.py
├── task7_prepare_data.py
├── task7_create_chroma.py
├── task7_retrieval_test.py
├── task7_retrieval_evaluation.py
├── task7_retrieval_evaluation_filtered.py
├── task7_query_preprocessing.py
├── task7_rag_generation.py
├── task7_rag_grounding_test.py
├── task7_threshold_analysis.py
├── task7_reranking.py
└── task7_retrieval_error_analysis.py

Status
🚧 Work in progress.
