import pytest
from fastapi.testclient import TestClient
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from numpy.testing import assert_almost_equal

from src.main import app

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
    data = {""}
    response = client.post(
        "/mlapi-predict",
        json=data,
    )
    print(response.json())
    assert response.status_code == 200