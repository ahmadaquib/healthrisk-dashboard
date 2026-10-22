from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """1. Test health check endpoint returns 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "diabetes" in data["available_models"]
    assert "breast_cancer" in data["available_models"]


def test_diseases_endpoint():
    """2. Test /api/diseases returns both diabetes and breast cancer with schemas and examples."""
    response = client.get("/api/diseases")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    disease_ids = [d["id"] for d in data]
    assert "diabetes" in disease_ids
    assert "breast_cancer" in disease_ids

    # Check structure of fields and example patients
    for d in data:
        assert len(d["fields"]) > 0
        assert "high_risk" in d["example_patients"]
        assert "low_risk" in d["example_patients"]


def test_high_risk_scores_higher_than_low_risk():
    """3. Test high-risk example scores higher probability than low-risk example for each disease."""
    diseases_resp = client.get("/api/diseases")
    assert diseases_resp.status_code == 200
    diseases = diseases_resp.json()

    for d in diseases:
        d_id = d["id"]
        high_ex = d["example_patients"]["high_risk"]["values"]
        low_ex = d["example_patients"]["low_risk"]["values"]

        resp_high = client.post(f"/api/predict/{d_id}", json=high_ex)
        resp_low = client.post(f"/api/predict/{d_id}", json=low_ex)

        assert resp_high.status_code == 200, f"Failed high risk prediction for {d_id}: {resp_high.text}"
        assert resp_low.status_code == 200, f"Failed low risk prediction for {d_id}: {resp_low.text}"

        prob_high = resp_high.json()["probability"]
        prob_low = resp_low.json()["probability"]

        assert prob_high > prob_low, f"For {d_id}, high prob ({prob_high}) is not > low prob ({prob_low})"
        assert len(resp_high.json()["top_factors"]) > 0


def test_missing_field_returns_422():
    """4. Test missing field returns 422 with clear error message."""
    incomplete_payload = {
        "Pregnancies": 2,
        # Glucose missing
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 25.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 33,
    }
    response = client.post("/api/predict/diabetes", json=incomplete_payload)
    assert response.status_code == 422
    assert "Missing required fields" in response.json()["detail"]
    assert "Glucose" in response.json()["detail"]


def test_impossible_value_returns_422():
    """5. Test biologically impossible values (e.g. negative age, glucose 5000) return 422."""
    impossible_payload = {
        "Pregnancies": 2,
        "Glucose": 5000,  # Clearly impossible glucose
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 25.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": -5,  # Negative age
    }
    response = client.post("/api/predict/diabetes", json=impossible_payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "exceeds biologically possible limit" in detail or "below biologically possible limit" in detail


def test_out_of_range_but_plausible_value_returns_warning():
    """6. Test an out-of-training-range but biologically plausible value returns 200 with warnings."""
    # Training max for Glucose was ~199 mg/dL. 220 mg/dL is plausible for severe diabetes (<=400 sane limit)
    plausible_high_payload = {
        "Pregnancies": 3,
        "Glucose": 220.0,
        "BloodPressure": 80.0,
        "SkinThickness": 30.0,
        "Insulin": 150.0,
        "BMI": 32.0,
        "DiabetesPedigreeFunction": 0.6,
        "Age": 45.0,
    }
    response = client.post("/api/predict/diabetes", json=plausible_high_payload)
    assert response.status_code == 200
    data = response.json()
    assert "warnings" in data
    assert len(data["warnings"]) > 0
    assert any("Plasma Glucose" in w and "exceeds training set maximum" in w for w in data["warnings"])
