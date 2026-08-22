import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri(
    "sqlite:///C:/Users/jashi/OneDrive/Desktop/Diabetes_Prediction_MLOps/mlflow.db"
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load processed training and testing data
X_train = pd.read_csv("../data/processed/X_train.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")
y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("../data/processed/y_test.csv").squeeze()

# Set MLflow experiment
mlflow.set_experiment("Diabetes_Prediction")

# Start MLflow run
with mlflow.start_run():

    # Model parameters
    max_iter = 1000

    # Create model
    model = LogisticRegression(max_iter=max_iter)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log parameters
    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("max_iter", max_iter)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model to MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        name="diabetes_model"
    )

    print("MLflow experiment completed!")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

# Save model locally
joblib.dump(model, "../models/diabetes_model.pkl")

print("\nModel saved successfully!")