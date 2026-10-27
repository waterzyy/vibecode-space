"""Общие заглушки для тестов: подмена клиента Gemini и помощники для HTTP-проверок."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Тесты не должны трогать реальный каталог .state/ проекта.
os.environ.setdefault("GEMINI_COOKIE_PATH", str(Path(tempfile.mkdtemp(prefix="gwb-state-"))))

from gemini_webapi import exceptions as gexc  # noqa: E402

import server  # noqa: E402


class Controller:
    """Управляет поведением подменённого клиента: что ответит модель и как упасть."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.chunks: list[str] = ["Hello", " world", "!"]
        self.fail: str | None = None  # None | "auth" | "mid"
        self.models = ["gemini-3-pro", "gemini-2.5-pro"]


class FakeClient:
    """Имитирует интерфейс gemini_webapi.GeminiClient, который использует мост."""

    def __init__(self, ctl: Controller) -> None:
        self._ctl = ctl
        self.client = None
        self.account_status = SimpleNamespace(name="AVAILABLE")

    async def init(self, **kwargs) -> None:
        if self._ctl.fail == "auth":
            raise gexc.AuthError("cookies rejected")
        self.client = object()

    async def close(self, delay: float = 0) -> None:
        self.client = None

    def list_models(self):
        return [SimpleNamespace(model_name=name) for name in self._ctl.models]

    def resolve_model(self, name: str):
        for model in self.list_models():
            if model.model_name == name:
                return model
        raise ValueError(f"Unknown model name: {name}")

    async def generate_content_stream(self, prompt, model=None, temporary=False, **kwargs):
        self._ctl.calls.append(
            {
                "prompt": prompt,
                "model": getattr(model, "model_name", None),
                "temporary": temporary,
                "t": time.monotonic(),
            }
        )
        for index, piece in enumerate(self._ctl.chunks):
            if self._ctl.fail == "mid" and index == 2:
                raise gexc.APIError("boom mid stream")
            yield SimpleNamespace(text_delta=piece)


def install_fake(monkeypatch) -> Controller:
    ctl = Controller()
    monkeypatch.setattr(server, "GeminiClient", lambda **kwargs: FakeClient(ctl))
    return ctl


def make_settings(**overrides) -> server.Settings:
    base = dict(
        cookie_1psid="x",
        cookie_1psidts=None,
        model="gemini-2.5-pro",
        api_key=None,
        host="127.0.0.1",
        port=0,
        proxy=None,
        delay_min=0.2,
        delay_max=0.3,
        temporary_chat=True,
        timeout=30,
        token_rotate_seconds=3600,
        log_level="WARNING",
    )
    base.update(overrides)
    return server.Settings(**base)


def sse_events(text: str) -> list:
    """Разбирает SSE-тело: JSON-объекты и строка '[DONE]'."""
    events: list = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        payload = block[len("data: ") :]
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def stream_text_of(events: list) -> str:
    return "".join(
        event["choices"][0]["delta"].get("content") or ""
        for event in events
        if isinstance(event, dict) and event.get("choices")
    )
