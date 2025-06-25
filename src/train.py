import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model():
    print("Starting training...")

    # Load data
    data = load_iris()
    print("Data loaded")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")

    # Model init and training
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    print("Model trained")

    # Prediction
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc}")

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_param("max_iter", 200)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
    print("MLflow logging done")

if __name__ == "__main__":
    train_model()
