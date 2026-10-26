from fastapi.testclient import TestClient

from switchyard.app import create_app
from tests.conftest import AUTH, chat, make_settings


def test_openai_compatible_response_shape(client):
    r = chat(client, "hi there")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["total_tokens"] > 0


def test_easy_prompt_routes_to_small_model(client):
    r = chat(client, "What is the capital of France?")
    assert r.headers["x-switchyard-model"] == "small"


def test_hard_prompt_routes_to_large_model(client):
    prompt = ("Prove step by step that the algorithm's complexity is O(n log n), "
              "analyze the trade-offs, and debug this:\n```def f(x):\n  return x ** 2```")
    r = chat(client, prompt)
    assert r.headers["x-switchyard-model"] == "large"


def test_explicit_model(client):
    assert chat(client, "hello", model="medium").headers["x-switchyard-model"] == "medium"


def test_unknown_model_404(client):
    assert chat(client, "hello", model="nope").status_code == 404


def test_auth_required(client):
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401


def test_fallback_when_provider_down(client, mock):
    mock.down.add("small")
    r = chat(client, "What is 2 plus 2?")
    assert r.status_code == 200
    assert r.headers["x-switchyard-model"] == "medium"  # escalates upward first
    assert r.headers["x-switchyard-attempts"] == "2"


def test_all_down_returns_502(client, mock):
    mock.down.update({"small", "medium", "large"})
    assert chat(client, "hello").status_code == 502


def test_circuit_breaker_stops_calling_dead_model(client, mock):
    mock.down.add("small")
    for i in range(6):
        chat(client, f"unique question number {i}?")
    # threshold is 3: after the breaker opens, small is skipped entirely
    assert mock.calls["small"] == 3


def test_exact_and_semantic_cache(client, mock):
    a = chat(client, "How do I reverse a list in Python?")
    assert a.headers["x-switchyard-cache"] == "miss"
    b = chat(client, "How do I reverse a list in Python?")
    assert b.headers["x-switchyard-cache"] == "exact"
    c = chat(client, "how do i reverse a python list")
    assert c.headers["x-switchyard-cache"] == "semantic"
    assert sum(mock.calls.values()) == 1


def test_semantic_cache_rejects_different_numbers(client):
    chat(client, "SELECT top 5 customers by total revenue, joining orders and customers tables")
    r = chat(client, "SELECT top 50 customers by total revenue, joining orders and customers tables")
    assert r.headers["x-switchyard-cache"] == "miss"


def test_long_templated_prompts_not_semantically_matched(client):
    tmpl = ("Design {}, analyze the trade-offs of the storage options, derive capacity "
            "step by step:\n```class Service:\n  pass```")
    chat(client, tmpl.format("a URL shortener"))
    assert chat(client, tmpl.format("a rate limiter")).headers["x-switchyard-cache"] == "miss"


def test_high_temperature_not_cached(client, mock):
    chat(client, "write a poem about trains", temperature=0.9)
    r = chat(client, "write a poem about trains", temperature=0.9)
    assert r.headers["x-switchyard-cache"] == "miss"


def test_cache_respects_conversation_context(client):
    msgs1 = [{"role": "system", "content": "Answer in French."},
             {"role": "user", "content": "What is the capital of Japan?"}]
    msgs2 = [{"role": "system", "content": "Answer in Spanish."},
             {"role": "user", "content": "What is the capital of Japan?"}]
    client.post("/v1/chat/completions", headers=AUTH, json={"messages": msgs1})
    r = client.post("/v1/chat/completions", headers=AUTH, json={"messages": msgs2})
    assert r.headers["x-switchyard-cache"] == "miss"


def test_rate_limit(tmp_path, mock):
    s = make_settings(tmp_path, api_keys=[{"key": "sk-test", "requests_per_minute": 2}])
    c = TestClient(create_app(s, providers={"mock": mock}))
    codes = [chat(c, f"q{i}").status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_budget_enforced(tmp_path, mock):
    s = make_settings(tmp_path, api_keys=[{"key": "sk-test", "monthly_budget_usd": 0.0000001}])
    c = TestClient(create_app(s, providers={"mock": mock}))
    assert chat(c, "first", model="large").status_code == 200
    assert chat(c, "second", model="large").status_code == 402


def test_stats_and_metrics(client):
    chat(client, "hello")
    chat(client, "hello")
    stats = client.get("/v1/stats").json()
    assert stats["requests"] == 2
    assert stats["cache"]["exact"] == 1
    assert stats["saved_usd"] > 0
    m = client.get("/metrics").text
    assert "switchyard_requests_total" in m
    assert "switchyard_latency_ms_bucket" in m


def test_route_explain(client):
    r = client.post("/v1/route/explain", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["target_tier"] == 1
