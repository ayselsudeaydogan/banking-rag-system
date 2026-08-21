import pandas as pd

# ==========================================
# 1. QUESTION-ANSWER DATASET
# ==========================================

qa_df = pd.read_excel("question_answers.xlsx")

print("\n===================================")
print("QUESTION-ANSWER DATA CLEANING")
print("===================================")

print("\nOriginal records:", len(qa_df))

# Remove exact duplicate rows
qa_clean = qa_df.drop_duplicates().copy()

print("Duplicate records removed:", len(qa_df) - len(qa_clean))
print("Clean records:", len(qa_clean))

# Remove unnecessary spaces
qa_clean["question"] = qa_clean["question"].astype(str).str.strip()
qa_clean["answer"] = qa_clean["answer"].astype(str).str.strip()

# Save clean dataset
qa_clean.to_excel(
    "question_answers_clean.xlsx",
    index=False
)

print("Clean dataset saved: question_answers_clean.xlsx")


# ==========================================
# 2. CHAPTER-END QUESTIONS
# ==========================================

chapter_df = pd.read_excel("BÖLÜM_SONU_SORULARI.xlsx")

print("\n===================================")
print("CHAPTER QUESTIONS ANALYSIS")
print("===================================")

print("\nDataset shape:")
print(chapter_df.shape)

print("\nColumns:")
print(chapter_df.columns.tolist())

print("\nFirst 5 rows:")
print(chapter_df.head())

print("\nMissing values:")
print(chapter_df.isnull().sum())

print("\nDuplicate rows:")
print(chapter_df.duplicated().sum())

print("\nTotal records:")
print(len(chapter_df))
