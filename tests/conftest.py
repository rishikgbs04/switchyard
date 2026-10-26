import pytest
from fastapi.testclient import TestClient

from switchyard.app import create_app
from switchyard.config import Settings
from switchyard.providers import MockProvider

KEY = "sk-test"
AUTH = {"Authorization": f"Bearer {KEY}"}


def make_settings(tmp_path, **over) -> Settings:
    base = dict(
        db_path=str(tmp_path / "test.db"),
        api_keys=[{"key": KEY, "requests_per_minute": 1000, "monthly_budget_usd": 100}],
        models=[
            {"name": "small", "provider": "mock", "tier": 1, "input_cost_per_mtok": 0.1, "output_cost_per_mtok": 0.4},
            {"name": "medium", "provider": "mock", "tier": 2, "input_cost_per_mtok": 1, "output_cost_per_mtok": 4},
            {"name": "large", "provider": "mock", "tier": 3, "input_cost_per_mtok": 5, "output_cost_per_mtok": 25},
        ],
    )
    base.update(over)
    return Settings(**base)


@pytest.fixture
def mock():
    return MockProvider()


@pytest.fixture
def client(tmp_path, mock):
    app = create_app(make_settings(tmp_path), providers={"mock": mock})
    return TestClient(app)


def chat(client, content, model="auto", **kw):
    return client.post("/v1/chat/completions", headers=AUTH,
                       json={"model": model, "messages": [{"role": "user", "content": content}], **kw})
