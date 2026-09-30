import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load dataset
BASE_DIR = Path(__file__).resolve().parents[1]

file_path = BASE_DIR / "data" / "raw" / "diabetes.csv"
df = pd.read_csv(file_path)

# Separate input features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Data preprocessing completed!")
print("Total rows:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Number of features:", X.shape[1])

# Create processed data folder
# Create processed data folder
processed_dir = BASE_DIR / "data" / "processed"

processed_dir.mkdir(parents=True, exist_ok=True)

processed_dir.mkdir(parents=True, exist_ok=True)

X_train.to_csv(processed_dir / "X_train.csv", index=False)
X_test.to_csv(processed_dir / "X_test.csv", index=False)
y_train.to_csv(processed_dir / "y_train.csv", index=False)
y_test.to_csv(processed_dir / "y_test.csv", index=False)
print("\nProcessed datasets saved successfully!")