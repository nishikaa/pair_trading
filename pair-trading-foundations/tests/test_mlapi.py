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

def test_predict_format_fail(client):
    # Wrong format
    data = {"texts": ["I hate you.", "I love you."]}
    response = client.post(
        "/project-predict",
        json=data,
    )
    print(response.json())
    assert response.status_code == 422

    # Wrong format
    data = {"txts": ["I hate you.", "I love you."]}
    response = client.post(
        "/project-predict",
        json=data,
    )
    print(response.json())
    assert response.status_code == 422

def test_predict_capital_fail(client):
    # Wrong format
    data = {"texT": ["I hate you.", "I love you."]}
    response = client.post(
        "/project-predict",
        json=data,
    )
    print(response.json())
    assert response.status_code == 422

    # Wrong format
    data = {"TEXT": ["I hate you.", "I love you."]}
    response = client.post(
        "/project-predict",
        json=data,
    )
    print(response.json())
    assert response.status_code == 422

def test_predict_capital_input_pass(client):
    # Wrong format
    data = {"text": ["I HATE YOU.", "I LOVE YOU."]}
    response = client.post(
        "/project-predict",
        json=data,
    )
    print(response.json())
    assert response.status_code == 200

def test_predict_nonetype(client):
    # Empty input
    data = {"text": 2}
    response = client.post(
        "/project-predict",
        json=data,
    )
    print(response.json())
    assert response.status_code == 422

def test_predict_text_empty_pass(client):
    # Empty input
    data = {"text": [""]}
    response = client.post(
        "/project-predict",
        json=data,
    )
    print(response.json())
    assert response.status_code == 200

    # Empty array
    data = {"text": []}
    response = client.post(
        "/project-predict",
        json=data,
    )
    print(response.json())
    assert response.status_code == 200

def test_predict(client):
    data = {"text": ["I hate you.", "I love you."]}
    response = client.post(
        "/project-predict",
        json=data,
    )
    print(response.json())
    assert response.status_code == 200
    assert isinstance(response.json()["predictions"], list)
    assert isinstance(response.json()["predictions"][0], list)
    assert isinstance(response.json()["predictions"][0][0], dict)
    assert isinstance(response.json()["predictions"][1][0], dict)
    assert set(response.json()["predictions"][0][0].keys()) == {"label", "score"}
    assert set(response.json()["predictions"][0][1].keys()) == {"label", "score"}
    assert set(response.json()["predictions"][1][0].keys()) == {"label", "score"}
    assert set(response.json()["predictions"][1][1].keys()) == {"label", "score"}
    assert response.json()["predictions"][0][0]["label"] == "NEGATIVE"
    assert response.json()["predictions"][0][1]["label"] == "POSITIVE"
    assert response.json()["predictions"][1][0]["label"] == "POSITIVE"
    assert response.json()["predictions"][1][1]["label"] == "NEGATIVE"
    assert (
        assert_almost_equal(
            response.json()["predictions"][0][0]["score"], 0.936, decimal=1
        )
        is None
    )
    assert (
        assert_almost_equal(
            response.json()["predictions"][0][1]["score"], 0.064, decimal=1
        )
        is None
    )
    assert (
        assert_almost_equal(
            response.json()["predictions"][1][0]["score"], 0.997, decimal=1
        )
        is None
    )
    assert (
        assert_almost_equal(
            response.json()["predictions"][1][1]["score"], 0.003, decimal=1
        )
        is None
    )
