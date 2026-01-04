from locust import HttpUser, between, task


class CheckoutUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task(5)
    def checkout_flow(self):
        payload = {
            "user_id": "user-123",
            "cart_total": 42.5,
            "correlation_id": "locust-demo",
        }
        self.client.post("/api/checkout", json=payload)

    @task(1)
    def health(self):
        self.client.get("/healthz")
