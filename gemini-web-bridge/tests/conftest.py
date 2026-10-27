import pytest
from fastapi.testclient import TestClient
from support import install_fake, make_settings

import server


@pytest.fixture
def fake(monkeypatch):
    return install_fake(monkeypatch)


@pytest.fixture
def client(fake):
    with TestClient(server.create_app(make_settings())) as test_client:
        yield test_client
