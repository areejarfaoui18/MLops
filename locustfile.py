from locust import HttpUser, task, between
import json

class MLOpsUser(HttpUser):
    wait_time = between(1, 2)  # wait time between tasks in seconds

    @task
    def predict(self):
        payload = {
            "features": [1.0, 2.0, 3.0, 4.0]  # Replace with your model's expected input
        }
        headers = {"Content-Type": "application/json"}
        self.client.post("/predict", data=json.dumps(payload), headers=headers)
