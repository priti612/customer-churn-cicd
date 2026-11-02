from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    payload = {
        "Gender": "Female",
        "Senior_Citizen": "No",
        "Tenure_Months": 12,
        "Contract_Type": "Month-to-month",
        "Internet_Service": "Fiber",
        "Payment_Method": "Cash",
        "Monthly_Charges_INR": 950.0,
        "Total_Charges_INR": 11400.0,
        "Tech_Support": "No",
        "Online_Security": "No",
        "Support_Tickets_6M": 3,
        "Avg_Data_Usage_GB": 100.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["Churn", "Not Churn"]
