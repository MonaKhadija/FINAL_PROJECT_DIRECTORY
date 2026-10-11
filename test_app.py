import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_sentiment_prediction(client):
    response = client.post("/predict", json={"text": "I am extremely happy and love this tool!"})
    json_data = response.get_json()

    assert response.status_code == 200
    assert json_data["sentiment"] == "Positive"
    assert "confidence" in json_data