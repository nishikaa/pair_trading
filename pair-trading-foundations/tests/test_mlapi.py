import pytest
from fastapi.testclient import TestClient
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from numpy.testing import assert_almost_equal

from mlapi.main import app

@pytest.fixture
def client():
    FastAPICache.init(InMemoryBackend())
    with TestClient(app) as c:
        yield c

def test_get_health(client):
    response = client.get("/health")
    assert response.status_code == 200  # HTTP_200_OK
    assert "status" in response.json()


def test_mlapi(client):
    response = client.post(
        "/mlapi-predict",
        headers={"Content-Type": "application/json"},
        json={
                "duration_in_days": 2,
                "dollar_amt": 100,
             },
    )
    print(response.json())
    assert response.status_code == 200

def test_mlapi_fail_days(client):
    # Invalid schema
    response = client.post(
        "/mlapi-predict",
        headers={"Content-Type": "application/json"},
        json={
                "duration_in_days": 100,
             },
    )
    print(response.json())
    assert response.status_code == 422

def test_mlapi_fail_dollar(client):
    # Invalid figures
    response = client.post(
        "/mlapi-predict",
        headers={"Content-Type": "application/json"},
        json={
                "dollar_amt": 100,
             },
    )
    print(response.json())
    assert response.status_code == 422