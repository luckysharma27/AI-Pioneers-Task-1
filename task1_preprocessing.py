import pandas as pd
import numpy as np

# Load the sample dataset
df = pd.read_csv("student_performance_raw.csv")

print("Original shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 1. Remove duplicate records
df = df.drop_duplicates().reset_index(drop=True)

# 2. Handle missing numerical values using the median
numeric_cols = [
    "Age", "Study_Hours_Per_Day", "Attendance_Percent",
    "Assignments_Score", "Final_Score"
]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 3. Encode categorical variables
df["Gender_Encoded"] = df["Gender"].map({"Male": 0, "Female": 1})
df["Passed_Encoded"] = df["Passed"].map({"No": 0, "Yes": 1})

# 4. Select useful features for analysis
selected_features = [
    "Study_Hours_Per_Day", "Attendance_Percent",
    "Assignments_Score", "Final_Score"
]

# 5. Min-Max normalization
for col in selected_features:
    min_value = df[col].min()
    max_value = df[col].max()
    df[col + "_Normalized"] = (
        (df[col] - min_value) / (max_value - min_value)
    )

# 6. Basic exploratory analysis
print("\nDataset information:")
print(df.info())

print("\nDescriptive statistics:")
print(df[selected_features].describe())

print("\nCorrelation matrix:")
print(df[selected_features].corr())

# Save the processed dataset
df.to_csv("student_performance_processed.csv", index=False)

print("\nProcessed dataset saved as student_performance_processed.csv")
