import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
import chromadb

from main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_chroma_collection():
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="test_esilv_trading")
    yield collection
    chroma_client.delete_collection(name="test_esilv_trading")
