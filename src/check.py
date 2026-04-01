import pandas as pd

# Load dataset
df = pd.read_csv("data/processed/spam_cleaned.csv")  # adjust path if needed

# Check labels
print("Label distribution:")
print(df['label'].value_counts())
print("\nLabel proportion:")
print(df['label'].value_counts(normalize=True))

# Check column names
print("\nColumns:", df.columns)

# Show some sample messages
print("\nSample spam messages:")
print(df[df['label']==1].sample(10)['message'].values)

print("\nSample ham messages:")
print(df[df['label']==0].sample(10)['message'].values)

# Inspect spam keywords in ham messages
spam_keywords = ["free", "win", "winner", "cash", "prize", "offer", "urgent", "click", "buy now", "iphone"]

print("\nSpam keywords in HAM messages:")
for word in spam_keywords:
    count = df[(df['label']==0) & df['message'].str.contains(word, case=False, na=False)].shape[0]
    print(f"{word}: {count} ham messages")

# Optional: check message length
df['length'] = df['message'].str.len()
print("\nMessage length stats:")
print(df['length'].describe())