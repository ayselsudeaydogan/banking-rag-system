import pandas as pd

# Load question-answer dataset
qa_df = pd.read_excel("question_answers.xlsx")

print("\n===================================")
print("TASK 7 DATA ANALYSIS")
print("===================================")

print("\nDataset shape:")
print(qa_df.shape)

print("\nColumns:")
print(qa_df.columns.tolist())

print("\nFirst 5 rows:")
print(qa_df.head())

print("\nMissing values:")
print(qa_df.isnull().sum())

print("\nDuplicate rows:")
print(qa_df.duplicated().sum())

empty_questions = (qa_df["question"].astype(str).str.strip() == "").sum()
empty_answers = (qa_df["answer"].astype(str).str.strip() == "").sum()

print("\nEmpty questions:", empty_questions)
print("Empty answers:", empty_answers)

qa_df["question_length"] = qa_df["question"].astype(str).str.len()
qa_df["answer_length"] = qa_df["answer"].astype(str).str.len()

print("\nQuestion length statistics:")
print(qa_df["question_length"].describe())

print("\nAnswer length statistics:")
print(qa_df["answer_length"].describe())

print("\nVery short questions (< 10 characters):")
print((qa_df["question_length"] < 10).sum())

print("\nVery short answers (< 20 characters):")
print((qa_df["answer_length"] < 20).sum())

print("\n===================================")
print("QUALITY CHECK DETAILS")
print("===================================")

print("\nVery short answers (< 20 characters):")
short_answers = qa_df[qa_df["answer_length"] < 20]
print(short_answers[["question", "answer"]].to_string(index=False))

print("\n===================================")
print("DUPLICATE RECORDS")
print("===================================")

duplicates = qa_df[qa_df.duplicated(keep=False)]
print(duplicates[["question", "answer"]].to_string(index=False))
