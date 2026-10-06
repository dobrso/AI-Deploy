import pytest
from fastapi.testclient import TestClient

from ai_deploy.api.api import app

@pytest.fixture
def api_client():
    with TestClient(app) as client:
        yield client

class TestAPI:
    def test_health_returns_ok(self, api_client):
        response = api_client.get('/health')
        assert response.status_code == 200
        assert response.json() == {'status': 'ok'}

    def test_predict_is_working(self, api_client):
        response = api_client.post('/predict', json={'text': 'Привет, мир!'})
        assert response.status_code == 200
        assert 'result' in response.json()

    def test_predict_missing_text_returns_422(self, api_client):
        response = api_client.post('/predict')
        assert response.status_code == 422
