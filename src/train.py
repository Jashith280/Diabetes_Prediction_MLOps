import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# Project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# MLflow configuration
# ==========================================

mlflow.set_tracking_uri(
    f"sqlite:///{BASE_DIR / 'mlflow.db'}"
)

mlflow.set_experiment("Diabetes_Prediction")


# ==========================================
# Load processed data
# ==========================================

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).squeeze()

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).squeeze()


print("Training data shape:", X_train.shape)
print("Testing data shape :", X_test.shape)


# ==========================================
# Start MLflow run
# ==========================================

with mlflow.start_run():

    # --------------------------------------
    # Model parameters
    # --------------------------------------

    max_iter = 1000

    model = LogisticRegression(
        max_iter=max_iter
    )


    # --------------------------------------
    # Train model
    # --------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------
    # Predictions
    # --------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------
    # Evaluation metrics
    # --------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # --------------------------------------
    # Log parameters to MLflow
    # --------------------------------------

    mlflow.log_param(
        "model",
        "Logistic Regression"
    )

    mlflow.log_param(
        "max_iter",
        max_iter
    )

    mlflow.log_param(
        "training_rows",
        len(X_train)
    )

    mlflow.log_param(
        "testing_rows",
        len(X_test)
    )

    mlflow.log_param(
        "features",
        X_train.shape[1]
    )


    # --------------------------------------
    # Log metrics to MLflow
    # --------------------------------------

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )


    # --------------------------------------
    # Log model to MLflow
    # --------------------------------------

    mlflow.sklearn.log_model(
        sk_model=model,
        name="diabetes_model"
    )


    # --------------------------------------
    # Save model locally
    # --------------------------------------

    model_path = MODEL_DIR / "diabetes_model.pkl"

    joblib.dump(
        model,
        model_path
    )


    # --------------------------------------
    # Output
    # --------------------------------------

    print("\nMLflow experiment completed!")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nModel saved successfully!")
    print("Model path:", model_path)