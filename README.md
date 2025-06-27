# MLOps Pipeline Project

## 📌 Project Overview

This project is a full-stack MLOps pipeline that takes a machine learning model from development to production with automated deployment, monitoring, and performance tracking. The pipeline includes model training, API serving, containerization, CI/CD, scalability, monitoring, and alerting.

It was designed as part of a DevOps-focused project with permission to use MLOps techniques.

---

## 🛠️ Technologies Used

* **FastAPI** – For serving the machine learning model via REST API.
* **Scikit-learn** – For model training.
* **MLflow** – For model tracking and artifact management.
* **Docker** – For containerization.
* **Azure Container Apps** – For cloud deployment.
* **GitHub Actions** – For CI/CD automation.
* **Prometheus** – For metrics scraping.
* **Grafana** – For monitoring and visualizing metrics.
* **Locust** – For load testing.

---

## 🔄 Pipeline Components

### 1. **Model Development and Tracking**

* A logistic regression model is trained using `scikit-learn`.
* MLflow is used to track parameters, metrics, and save the model artifact.

### 2. **API Endpoint**

* The model is exposed through a FastAPI application with endpoints:

  * `/predict`: Accepts input features and returns predictions.
  * `/metrics`: Exposes Prometheus-compatible metrics (e.g., request count, latency, exceptions).

### 3. **Containerization**

* The application is containerized using a `Dockerfile`.
* Docker image is pushed to Azure Container Registry.

### 4. **Deployment (Azure Container Apps)**

* The container is deployed to Azure Container Apps.
* Autoscaling is enabled based on HTTP request load.

### 5. **CI/CD with GitHub Actions**

* A GitHub Actions workflow automatically:

  * Lints the code with pylint
  * Runs unit tests with pytest
  * Measures code coverage using `coverage.py` (integrated into pytest)
  * Builds and pushes the Docker image
  * Deploys to Azure on each push to `main`

### 6. **Monitoring with Prometheus and Grafana**

* Prometheus scrapes metrics from the `/metrics` endpoint.
* Grafana visualizes:

  * Request count
  * Latency distribution
  * Exceptions count
  * Alert notifications via Gmail is configured in Grafana.

### 7. **Load Testing**

* Locust simulates concurrent users to test API resilience.
* Stress testing results are visualized in Grafana.

---

## 📂 Project Structure

```
.
├── .github/workflows/            # GitHub Actions CI/CD config
├── Dockerfile                    # Container build file
├── requirements.txt              # Python dependencies
├── src/
│   └── train.py 
    ├── main.py                      # FastAPI app exposing predict and metrics endpoints
                 # Model training script
├── tests/
│   ├── main_test.py              # Tests for FastAPI app
│   └── train_test.py             # Tests for training script
├── model_artifact/              # Saved ML model (MLflow format)
├── prometheus.yml               # Prometheus config file
├── .env       # SMTP config for alerting
└── locustfile.py                # Load testing script
```

---

## 🧪 How to Run Locally

1. **Train the model**

```bash
python src/train.py
```

2. **Run the FastAPI server**

```bash
uvicorn main:app --reload
```

3. **Run Docker**

```bash
docker build -t mlops-api .
docker run -p 8000:8000 mlops-api
```

4. **Run Prometheus and Grafana**

```bash
docker-compose up -d
```

5. **Run Locust for Load Testing**

```bash
locust -f locustfile.py --host https://<your-api-endpoint>
```

---

## 📈 Sample Grafana Visualizations

* Request Count → Stat/Bar
* Latency → Histogram/Time Series (use: `rate(request_latency_seconds_sum[5m]) / rate(request_latency_seconds_count[5m])`)
* Exception Count → Single Stat

---

## 🔔 Alerting

* Configure SMTP in `.env` for email alerts.

---

## 🔐 Secrets and Configs

* GitHub Secrets are used to store sensitive values:

  * `MLOPSAPI_AZURE_CLIENT_ID`
  * `MLOPSAPI_AZURE_TENANT_ID`
  * `MLOPSAPI_AZURE_SUBSCRIPTION_ID`
  * `MLOPSAPI_REGISTRY_USERNAME`
  * `MLOPSAPI_REGISTRY_PASSWORD`

---
* A `.env` file can be used locally for Grafana SMTP credentials. Be sure to add `.env` to `.gitignore`.


## ✅ Status

✅ Model training, testing, tracking complete
✅ CI/CD and deployment pipeline live
✅ Monitoring and alerting in place
✅ Load testing validated

---

## 🙌 Credits

Built by Arij Arfaoui for Mission Entreprise @ ESPRIT

---

## 📧 Contact

**Email:** [arij.arfaoui@esprit.tn](mailto:arij.arfaoui@esprit.tn)
