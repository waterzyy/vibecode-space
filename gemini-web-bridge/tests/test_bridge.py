"""Регрессионные тесты базового функционала: промпт, API, стриминг, ошибки, настройки."""

import pytest
from fastapi.testclient import TestClient
from support import make_settings, sse_events

import server

CHAT = "/v1/chat/completions"


def user(text: str) -> dict:
    return {"role": "user", "content": text}


# ---------------------------------------------------------------------------
# Сборка промпта
# ---------------------------------------------------------------------------


def test_single_user_message_is_sent_raw():
    assert server.build_prompt([server.ChatMessage(role="user", content="Привет")]) == "Привет"


def test_transcript_contains_system_and_keeps_order():
    prompt = server.build_prompt(
        [
            server.ChatMessage(role="system", content="Be terse."),
            server.ChatMessage(role="user", content="Q1"),
            server.ChatMessage(role="assistant", content="A1"),
            server.ChatMessage(role="user", content=[{"type": "text", "text": "Q2a"}, {"type": "text", "text": "Q2b"}]),
        ]
    )
    assert "### System instructions\nBe terse." in prompt
    assert prompt.index("User: Q1") < prompt.index("Assistant: A1") < prompt.index("User: Q2a\nQ2b")
    assert prompt.rstrip().endswith("Assistant:")


def test_developer_role_is_treated_as_system():
    prompt = server.build_prompt(
        [server.ChatMessage(role="developer", content="dev"), server.ChatMessage(role="user", content="hi")]
    )
    assert "### System instructions\ndev" in prompt


def test_image_part_is_rejected_with_400():
    message = server.ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": "x"}}])
    with pytest.raises(server.BridgeError) as exc:
        server.build_prompt([message])
    assert exc.value.status_code == 400


def test_unknown_role_is_rejected_with_400():
    with pytest.raises(server.BridgeError) as exc:
        server.build_prompt([server.ChatMessage(role="wizard", content="x")])
    assert exc.value.status_code == 400


def test_model_prefix_is_stripped():
    assert server.normalize_model_name("openai/gemini-2.5-pro") == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# HTTP API без tools
# ---------------------------------------------------------------------------


def test_models_lists_configured_and_account_models(client):
    response = client.get("/v1/models")
    ids = [model["id"] for model in response.json()["data"]]
    assert response.status_code == 200
    assert ids == ["gemini-2.5-pro", "gemini-3-pro"]


def test_healthz_is_ok_when_session_is_up(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_non_stream_response_shape(client, fake):
    response = client.post(CHAT, json={"model": "gemini-2.5-pro", "messages": [user("Hi")]})
    body = response.json()
    assert response.status_code == 200
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello world!"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert fake.calls[-1]["temporary"] is True


def test_prefixed_model_resolves_to_registry_model(client, fake):
    response = client.post(CHAT, json={"model": "openai/gemini-3-pro", "messages": [user("Hi")]})
    assert fake.calls[-1]["model"] == "gemini-3-pro"
    assert response.json()["model"] == "openai/gemini-3-pro"


def test_unknown_model_falls_back_to_default(client, fake):
    response = client.post(CHAT, json={"model": "gpt-4o", "messages": [user("Hi")]})
    assert response.status_code == 200
    assert fake.calls[-1]["model"] == "gemini-2.5-pro"


def test_stream_without_tools_passes_text_through(client):
    with client.stream(
        "POST",
        CHAT,
        json={
            "stream": True,
            "messages": [{"role": "system", "content": "S"}, user("Hi")],
        },
    ) as response:
        raw = "".join(response.iter_text())
        content_type = response.headers["content-type"]
    events = sse_events(raw)
    chunks = [event for event in events if isinstance(event, dict)]
    assert content_type.startswith("text/event-stream")
    assert events[-1] == "[DONE]"
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Hello world!"
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_consecutive_requests_are_paced(client, fake):
    client.post(CHAT, json={"messages": [user("one")]})
    client.post(CHAT, json={"messages": [user("two")]})
    gap = fake.calls[-1]["t"] - fake.calls[-2]["t"]
    assert gap >= 0.2  # delay_min из make_settings


@pytest.mark.parametrize(
    "payload",
    [{"messages": []}, {"model": "x"}],
    ids=["empty-messages", "missing-messages"],
)
def test_invalid_body_returns_400(client, payload):
    response = client.post(CHAT, json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_invalid_json_returns_400(client):
    response = client.post(CHAT, content="not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Ошибки
# ---------------------------------------------------------------------------


def test_mid_stream_error_is_sent_as_sse_frame(fake):
    fake.fail = "mid"
    with TestClient(server.create_app(make_settings())) as client:
        with client.stream("POST", CHAT, json={"stream": True, "messages": [user("Hi")]}) as response:
            status = response.status_code
            raw = "".join(response.iter_text())
        events = sse_events(raw)
        assert status == 200
        assert any(isinstance(e, dict) and "error" in e for e in events)
        assert events[-1] == "[DONE]"

        response = client.post(CHAT, json={"messages": [user("Hi")]})
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "upstream_error"


def test_auth_error_before_stream_returns_401_json(fake):
    fake.fail = "auth"
    with TestClient(server.create_app(make_settings())) as client:
        response = client.post(CHAT, json={"stream": True, "messages": [user("Hi")]})
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "authentication_error"
        health = client.get("/healthz")
        assert health.status_code == 503
        assert health.json()["status"] == "degraded"


def test_api_key_is_enforced(fake):
    with TestClient(server.create_app(make_settings(api_key="secret123"))) as client:
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer secret123"}).status_code == 200
        assert client.get("/healthz").status_code == 200


# ---------------------------------------------------------------------------
# Настройки из окружения
# ---------------------------------------------------------------------------

ENV_KEYS = [
    "GEMINI_COOKIE_1PSID",
    "GEMINI_COOKIE_1PSIDTS",
    "GEMINI_MODEL",
    "BRIDGE_API_KEY",
    "HOST",
    "PORT",
    "GEMINI_PROXY",
    "GEMINI_REQUEST_DELAY_MIN",
    "GEMINI_REQUEST_DELAY_MAX",
    "GEMINI_TOKEN_ROTATE_MINUTES",
    "GEMINI_TEMPORARY_CHAT",
    "GEMINI_TIMEOUT",
    "LOG_LEVEL",
]


def _clear_env(monkeypatch, **values):
    # Пустая строка трактуется как «не задано», поэтому .env из корня проекта не влияет на тест.
    for key in ENV_KEYS:
        monkeypatch.setenv(key, "")
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_missing_cookie_raises_config_error(monkeypatch):
    _clear_env(monkeypatch)
    with pytest.raises(server.ConfigError, match="GEMINI_COOKIE_1PSID"):
        server.Settings.from_env()


def test_bad_delay_range_raises_config_error(monkeypatch):
    _clear_env(monkeypatch, GEMINI_COOKIE_1PSID="abc", GEMINI_REQUEST_DELAY_MIN="3", GEMINI_REQUEST_DELAY_MAX="1")
    with pytest.raises(server.ConfigError):
        server.Settings.from_env()


def test_defaults_are_applied(monkeypatch):
    _clear_env(monkeypatch, GEMINI_COOKIE_1PSID="abc")
    settings = server.Settings.from_env()
    assert settings.model == "gemini-2.5-pro"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert (settings.delay_min, settings.delay_max) == (1.0, 2.5)
    assert settings.api_key is None
