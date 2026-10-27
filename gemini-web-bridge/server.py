"""gemini-web-bridge: локальный OpenAI-совместимый HTTP-прокси к веб-приложению Gemini.

Прокси работает от имени вашей авторизованной сессии gemini.google.com (куки
``__Secure-1PSID`` и ``__Secure-1PSIDTS``, подписка Google AI Pro) и публикует:

    POST /v1/chat/completions   JSON-ответ или SSE при ``"stream": true``
    GET  /v1/models             список моделей аккаунта
    GET  /healthz               состояние веб-сессии

Function calling эмулируется текстом: веб-интерфейс Gemini не умеет вызывать инструменты
через API. Схемы ``tools`` внедряются в системную инструкцию, а ответ модели разбирается
по блокам ``<tool_call>{...}</tool_call>`` и возвращается как ``tool_calls``.

Запуск: ``python server.py``. Настройки читаются из ``.env`` рядом с этим файлом.
"""

import asyncio
import json
import logging
import os
import random
import re
import secrets
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from gemini_webapi import AvailableModel, GeminiClient
from gemini_webapi import exceptions as gexc
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
# gemini-webapi хранит ротированный __Secure-1PSIDTS в каталоге из этой переменной.
# По умолчанию это каталог .state/ рядом с проектом (права 700, в .gitignore),
# а не общий системный /tmp.
os.environ.setdefault("GEMINI_COOKIE_PATH", str(BASE_DIR / ".state"))

log = logging.getLogger("gemini-web-bridge")

MODEL_PREFIXES = ("openai/", "gemini/", "google/")
ROTATION_JITTER = 0.15  # ±15% к интервалу ротации токена
PSIDTS_REFRESH_SECONDS = 600  # интервал фоновой ротации __Secure-1PSIDTS библиотекой

PREAMBLE = (
    "Below is the transcript of a conversation between a user and an AI assistant. "
    "Continue it by writing ONLY the assistant's next reply to the last user message. "
    "Do not repeat the transcript and do not write role labels such as 'User:' or 'Assistant:'."
)

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*[ \t]*\n?(.*?)\n?```$", re.DOTALL)


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------


class ConfigError(RuntimeError):
    """Ошибка в переменных окружения."""


def _env_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должно быть числом, получено {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должно быть целым числом, получено {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name)
    if raw is None:
        return default
    value = raw.lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} должно быть true/false, получено {raw!r}")


@dataclass(frozen=True)
class Settings:
    cookie_1psid: str
    cookie_1psidts: str | None
    model: str
    api_key: str | None
    host: str
    port: int
    proxy: str | None
    delay_min: float
    delay_max: float
    temporary_chat: bool
    timeout: float
    token_rotate_seconds: float
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        psid = _env_str("GEMINI_COOKIE_1PSID")
        if psid is None:
            raise ConfigError("не задан GEMINI_COOKIE_1PSID. Скопируйте .env.example в .env и заполните куки.")
        delay_min = _env_float("GEMINI_REQUEST_DELAY_MIN", 1.0)
        delay_max = _env_float("GEMINI_REQUEST_DELAY_MAX", 2.5)
        if not 0 <= delay_min <= delay_max:
            raise ConfigError("нужно выполнение условия 0 <= GEMINI_REQUEST_DELAY_MIN <= GEMINI_REQUEST_DELAY_MAX")
        rotate_minutes = _env_float("GEMINI_TOKEN_ROTATE_MINUTES", 30.0)
        if rotate_minutes < 1:
            raise ConfigError("GEMINI_TOKEN_ROTATE_MINUTES должно быть не меньше 1")
        log_level = (_env_str("LOG_LEVEL") or "INFO").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ConfigError(f"LOG_LEVEL должен быть DEBUG, INFO, WARNING или ERROR, получено {log_level!r}")
        return cls(
            cookie_1psid=psid,
            cookie_1psidts=_env_str("GEMINI_COOKIE_1PSIDTS"),
            model=_env_str("GEMINI_MODEL") or "gemini-2.5-pro",
            api_key=_env_str("BRIDGE_API_KEY"),
            host=_env_str("HOST") or "127.0.0.1",
            port=_env_int("PORT", 8000),
            proxy=_env_str("GEMINI_PROXY"),
            delay_min=delay_min,
            delay_max=delay_max,
            temporary_chat=_env_bool("GEMINI_TEMPORARY_CHAT", True),
            timeout=_env_float("GEMINI_TIMEOUT", 300.0),
            token_rotate_seconds=rotate_minutes * 60,
            log_level=log_level,
        )


# ---------------------------------------------------------------------------
# Ошибки в формате OpenAI
# ---------------------------------------------------------------------------


class BridgeError(Exception):
    """Ошибка, которую клиент получает в формате OpenAI ``{"error": {...}}``."""

    def __init__(self, status_code: int, message: str, err_type: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.err_type = err_type


def to_bridge_error(exc: Exception) -> BridgeError:
    if isinstance(exc, BridgeError):
        return exc
    if isinstance(exc, gexc.AuthError):
        return BridgeError(
            401,
            "Gemini отклонил куки сессии. Обновите GEMINI_COOKIE_1PSID и GEMINI_COOKIE_1PSIDTS "
            "из свежей сессии браузера и удалите каталог .state/.",
            "authentication_error",
        )
    if isinstance(exc, (gexc.UsageLimitExceededError, gexc.TemporarilyBlockedError)):
        return BridgeError(429, f"Лимит Gemini исчерпан или доступ временно ограничен: {exc}", "rate_limit_exceeded")
    if isinstance(exc, (gexc.TimeoutError, TimeoutError)):
        return BridgeError(504, "Gemini не ответил вовремя.", "timeout")
    detail = str(exc) or type(exc).__name__
    log.error("Ошибка веб-сессии Gemini: %s", detail)
    return BridgeError(502, f"Ошибка веб-сессии Gemini: {detail}", "upstream_error")


def openai_error_response(status_code: int, message: str, err_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type, "param": None, "code": None}},
    )


# ---------------------------------------------------------------------------
# Схемы запроса OpenAI
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    # Поля вроде temperature и max_tokens принимаются и игнорируются:
    # веб-интерфейс Gemini не предоставляет их управления.
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Tools: разбор схем и выбора tool_choice
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    @property
    def required(self) -> tuple[str, ...]:
        required = self.parameters.get("required", [])
        if isinstance(required, list):
            return tuple(item for item in required if isinstance(item, str))
        return ()


@dataclass(frozen=True)
class ToolPlan:
    """Что делать с инструментами в запросе.

    ``specs`` пуст, если инструменты отключены (нет ``tools`` или ``tool_choice="none"``).
    ``mode``: ``auto``, ``required`` или ``forced`` (вызов конкретной функции).
    """

    specs: dict[str, ToolSpec] = field(default_factory=dict)
    mode: str = "auto"
    forced: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.specs)


def normalize_tools(raw_tools: list[dict[str, Any]] | None) -> dict[str, ToolSpec]:
    """Превращает ``tools`` формата OpenAI в словарь имя -> ToolSpec."""
    specs: dict[str, ToolSpec] = {}
    for index, raw in enumerate(raw_tools or []):
        tool_type = raw.get("type", "function")
        function = raw.get("function")
        if tool_type != "function" or not isinstance(function, dict):
            raise BridgeError(
                400,
                f"tools[{index}]: поддерживается только {{'type': 'function', 'function': {{...}}}}.",
                "invalid_request_error",
            )
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise BridgeError(400, f"tools[{index}]: у функции не задано имя.", "invalid_request_error")
        name = name.strip()
        if name in specs:
            raise BridgeError(400, f"Функция {name!r} объявлена в tools дважды.", "invalid_request_error")
        parameters = function.get("parameters") or {"type": "object", "properties": {}}
        if not isinstance(parameters, dict):
            raise BridgeError(
                400, f"tools[{index}]: parameters должен быть JSON-схемой (объектом).", "invalid_request_error"
            )
        description = function.get("description") or ""
        specs[name] = ToolSpec(name=name, description=str(description), parameters=parameters)
    return specs


def resolve_tools(raw_tools: list[dict[str, Any]] | None, tool_choice: Any) -> ToolPlan:
    specs = normalize_tools(raw_tools)

    if tool_choice == "none":
        return ToolPlan(mode="none")
    if tool_choice in (None, "auto"):
        return ToolPlan(specs=specs, mode="auto") if specs else ToolPlan()
    if not specs:
        raise BridgeError(400, "tool_choice задан, но tools пусты.", "invalid_request_error")
    if tool_choice == "required":
        return ToolPlan(specs=specs, mode="required")
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        forced = function.get("name") if isinstance(function, dict) else None
        if tool_choice.get("type") == "function" and isinstance(forced, str):
            if forced not in specs:
                raise BridgeError(
                    400, f"tool_choice ссылается на функцию {forced!r}, которой нет в tools.", "invalid_request_error"
                )
            return ToolPlan(specs={forced: specs[forced]}, mode="forced", forced=forced)
    raise BridgeError(400, f"Неподдерживаемый tool_choice: {tool_choice!r}.", "invalid_request_error")


def tool_instructions(plan: ToolPlan) -> str:
    """Системная инструкция с описанием инструментов и строгим форматом вызова."""
    schema = json.dumps(
        [
            {"name": spec.name, "description": spec.description, "parameters": spec.parameters}
            for spec in plan.specs.values()
        ],
        indent=2,
        ensure_ascii=False,
    )
    if plan.mode == "required":
        requirement = "In this reply you MUST call one of the functions listed below."
    elif plan.mode == "forced":
        requirement = f'In this reply you MUST call the function "{plan.forced}".'
    else:
        requirement = (
            "Call a function only when it is needed to answer. Otherwise answer normally in plain text, "
            "without any <tool_call> tags."
        )
    return "\n".join(
        [
            "## Tool calling",
            "You can call the functions listed below. Calls are written as plain text in exactly this form:",
            "<tool_call>",
            '{"name": "<function_name>", "arguments": {<json arguments>}}',
            "</tool_call>",
            "Rules:",
            "- Inside <tool_call> write ONLY the JSON object. Put no other text inside the tags.",
            "- If you call a function, write no text before or after the <tool_call> blocks.",
            "- To call several independent functions, write several <tool_call> blocks one after another.",
            "- Use only the function names listed below. The arguments must match the function's JSON Schema.",
            "- Never invent a function result. The result arrives later as a message 'Tool Result (...)'; "
            "continue the task from it.",
            requirement,
            "",
            "Available functions (JSON):",
            schema,
        ]
    )


# ---------------------------------------------------------------------------
# Преобразование messages -> один промпт для Gemini
# ---------------------------------------------------------------------------


def message_text(message: ChatMessage) -> str:
    """Извлекает текст сообщения. Нетекстовые части (изображения, аудио) не поддерживаются."""
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for part in content:
        part_type = part.get("type")
        text = part.get("text")
        if part_type != "text" or not isinstance(text, str):
            raise BridgeError(
                400,
                f"Тип содержимого {part_type!r} не поддерживается: прокси передаёт в Gemini только текст.",
                "invalid_request_error",
            )
        texts.append(text)
    return "\n".join(texts)


def render_call_block(name: str, arguments: Any) -> str:
    """Вызов инструмента из истории в том же формате, который просим у модели."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass  # оставляем строку как есть
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{TOOL_OPEN}\n{payload}\n{TOOL_CLOSE}"


def _history_calls(message: ChatMessage) -> list[tuple[str, str, Any]]:
    """Вызовы из поля tool_calls сообщения assistant: (id, имя, аргументы)."""
    calls: list[tuple[str, str, Any]] = []
    for raw in message.tool_calls or []:
        function = raw.get("function") if isinstance(raw, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        calls.append((str(raw.get("id") or ""), function["name"], function.get("arguments", {})))
    return calls


def build_prompt(messages: list[ChatMessage], plan: ToolPlan | None = None) -> str:
    """Собирает промпт для Gemini из списка сообщений OpenAI.

    Одиночное сообщение user без system и без tools отправляется как есть. Иначе история
    склеивается в транскрипт: системные инструкции (при наличии tools — описание функций
    первым блоком), реплики пользователя, ответы ассистента, вызовы инструментов и результаты.
    """
    plan = plan or ToolPlan()
    system_parts: list[str] = []
    if plan.enabled:
        system_parts.append(tool_instructions(plan))

    turns: list[str] = []
    turn_roles: list[str] = []
    call_names: dict[str, str] = {}  # tool_call_id -> имя функции, из истории assistant

    for message in messages:
        role = message.role
        if role in {"system", "developer"}:
            text = message_text(message).strip()
            if text:
                system_parts.append(text)
        elif role == "user":
            text = message_text(message).strip()
            if text:
                turns.append(f"User: {text}")
                turn_roles.append("user")
        elif role == "assistant":
            parts: list[str] = []
            text = message_text(message).strip()
            if text:
                parts.append(text)
            for call_id, name, arguments in _history_calls(message):
                if call_id:
                    call_names[call_id] = name
                parts.append(render_call_block(name, arguments))
            if parts:
                turns.append("Assistant: " + "\n".join(parts))
                turn_roles.append("assistant")
        elif role in {"tool", "function"}:
            text = message_text(message).strip()
            label = message.name or call_names.get(message.tool_call_id or "") or "unknown"
            if message.tool_call_id:
                header = f"Tool Result ({label} / {message.tool_call_id})"
            else:
                header = f"Tool Result ({label})"
            turns.append(f"{header}: {text}")
            turn_roles.append("tool")
        else:
            raise BridgeError(400, f"Неизвестная роль сообщения: {role!r}.", "invalid_request_error")

    if not turns:
        raise BridgeError(400, "В messages нет непустых сообщений user, assistant или tool.", "invalid_request_error")

    if len(turns) == 1 and turn_roles == ["user"] and not system_parts:
        return turns[0][len("User: ") :]

    blocks = [PREAMBLE]
    if system_parts:
        blocks.append("### System instructions\n" + "\n\n".join(system_parts))
    blocks.append("### Conversation\n" + "\n\n".join(turns))
    blocks.append("Assistant:")
    return "\n\n".join(blocks)


def normalize_model_name(requested: str | None) -> str | None:
    if requested is None:
        return None
    name = requested.strip()
    for prefix in MODEL_PREFIXES:
        if name.lower().startswith(prefix):
            name = name[len(prefix) :]
            break
    return name or None


# ---------------------------------------------------------------------------
# Разбор ответа модели: <tool_call> -> tool_calls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContentEvent:
    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    index: int
    id: str
    name: str
    arguments: str  # JSON-строка, как в OpenAI API


DetectorEvent = ContentEvent | ToolCallEvent


def parse_tool_call(body: str, specs: dict[str, ToolSpec]) -> tuple[str, str] | None:
    """Разбирает содержимое блока <tool_call>. Возвращает (имя, JSON аргументов) или None.

    None означает, что блок невалиден: неверный JSON, неизвестная функция, аргументы не
    объект или не хватает обязательных полей.
    """
    text = body.strip()
    fence = _CODE_FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    if not isinstance(name, str) or name not in specs:
        return None
    arguments: Any = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if not isinstance(arguments, dict):
        return None
    if any(key not in arguments for key in specs[name].required):
        return None
    return name, json.dumps(arguments, ensure_ascii=False)


def _partial_tag_len(text: str, tag: str) -> int:
    """Длина самого длинного хвоста text, который может оказаться началом tag."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class ToolCallDetector:
    """Потоковый детектор блоков <tool_call>.

    Текст вне блоков отдаётся сразу; задерживаются только символы, которые могут быть
    началом открывающего тега. Содержимое блока копится до закрывающего тега. Валидный блок
    превращается в ToolCallEvent, невалидный возвращается обычным текстом (ContentEvent).
    Если ``specs`` пуст, детектор работает как прозрачный проход.
    """

    def __init__(self, specs: dict[str, ToolSpec] | None = None) -> None:
        self._specs = specs or {}
        self._pending = ""
        self._in_call = False
        self.calls_emitted = 0

    def feed(self, text: str) -> list[DetectorEvent]:
        if not text:
            return []
        if not self._specs:
            return [ContentEvent(text)]
        self._pending += text
        return self._drain(final=False)

    def finish(self) -> list[DetectorEvent]:
        if not self._specs:
            return []
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[DetectorEvent]:
        events: list[DetectorEvent] = []
        while True:
            if self._in_call:
                end = self._pending.find(TOOL_CLOSE)
                if end < 0:
                    if final:
                        # Блок не закрыт до конца ответа: отдаём текст как есть.
                        events.append(ContentEvent(TOOL_OPEN + self._pending))
                        self._pending = ""
                        self._in_call = False
                    break
                body = self._pending[:end]
                self._pending = self._pending[end + len(TOOL_CLOSE) :]
                self._in_call = False
                events.append(self._finish_block(body))
                continue

            start = self._pending.find(TOOL_OPEN)
            if start >= 0:
                if start:
                    events.append(ContentEvent(self._pending[:start]))
                self._pending = self._pending[start + len(TOOL_OPEN) :]
                self._in_call = True
                continue

            keep = 0 if final else _partial_tag_len(self._pending, TOOL_OPEN)
            emit = self._pending[: len(self._pending) - keep]
            if emit:
                events.append(ContentEvent(emit))
            self._pending = self._pending[len(self._pending) - keep :]
            break
        return events

    def _finish_block(self, body: str) -> DetectorEvent:
        parsed = parse_tool_call(body, self._specs)
        if parsed is None:
            log.warning("Модель вернула невалидный блок <tool_call>; он передан клиенту как текст.")
            return ContentEvent(f"{TOOL_OPEN}{body}{TOOL_CLOSE}")
        name, arguments = parsed
        event = ToolCallEvent(
            index=self.calls_emitted,
            id=f"call_{uuid.uuid4().hex}",
            name=name,
            arguments=arguments,
        )
        self.calls_emitted += 1
        return event


def tool_call_object(event: ToolCallEvent) -> dict[str, Any]:
    """Полный объект tool_calls для нестримингового ответа."""
    return {
        "id": event.id,
        "type": "function",
        "function": {"name": event.name, "arguments": event.arguments},
    }


# ---------------------------------------------------------------------------
# Пейсинг запросов
# ---------------------------------------------------------------------------


class Pacer:
    """Одна веб-сессия: запросы идут строго по очереди, с паузой со случайным jitter."""

    def __init__(self, delay_min: float, delay_max: float) -> None:
        self._lock = asyncio.Lock()
        self._delay_min = delay_min
        self._delay_max = delay_max
        self._last_finished: float | None = None

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._lock:
            if self._last_finished is not None:
                pause = random.uniform(self._delay_min, self._delay_max)
                wait = self._last_finished + pause - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
            try:
                yield
            finally:
                self._last_finished = time.monotonic()


# ---------------------------------------------------------------------------
# Мост к веб-сессии Gemini
# ---------------------------------------------------------------------------


class GeminiBridge:
    """Единственная веб-сессия Gemini, общая для всех запросов."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = GeminiClient(
            secure_1psid=settings.cookie_1psid,
            secure_1psidts=settings.cookie_1psidts,
            proxy=settings.proxy,
        )
        self._pacer = Pacer(settings.delay_min, settings.delay_max)
        self._rotation_task: asyncio.Task[None] | None = None
        self.last_error: str | None = None

    @property
    def ready(self) -> bool:
        return self._client.client is not None

    @property
    def account_status(self) -> str:
        return self._client.account_status.name

    def available_models(self) -> list[str]:
        return [model.model_name for model in self._client.list_models() or []]

    async def _open_session(self) -> None:
        # init() ничего не делает, если сессия уже открыта. После close() он заново
        # получает SNlM0e и использует сохранённые ротированные куки.
        await self._client.init(
            timeout=self._settings.timeout,
            auto_refresh=True,
            refresh_interval=PSIDTS_REFRESH_SECONDS,
            verbose=False,
        )

    async def start(self) -> None:
        try:
            async with self._pacer.slot():
                await self._open_session()
            self.last_error = None
            log.info(
                "Сессия Gemini готова, статус аккаунта: %s. Доступно моделей: %d.",
                self.account_status,
                len(self.available_models()),
            )
        except Exception as exc:
            self.last_error = str(exc) or type(exc).__name__
            log.error(
                "Не удалось открыть сессию Gemini: %s. Сервер продолжит работу и повторит попытку при запросе.",
                self.last_error,
            )
        self._rotation_task = asyncio.create_task(self._rotation_loop(), name="snlm0e-rotation")

    async def _rotation_loop(self) -> None:
        """Периодически пересоздаёт сессию: новый SNlM0e и актуальные куки."""
        base = self._settings.token_rotate_seconds
        while True:
            await asyncio.sleep(base * random.uniform(1 - ROTATION_JITTER, 1 + ROTATION_JITTER))
            try:
                async with self._pacer.slot():
                    await self._client.close()
                    await self._open_session()
                self.last_error = None
                log.info("Токен сессии SNlM0e обновлён.")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc) or type(exc).__name__
                log.warning("Ротация токена не удалась: %s. Следующая попытка при запросе.", self.last_error)

    async def shutdown(self) -> None:
        if self._rotation_task is not None:
            self._rotation_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._rotation_task
        await self._client.close()

    def _resolve_model(self, requested: str | None) -> AvailableModel | None:
        name = normalize_model_name(requested) or self._settings.model
        try:
            return self._client.resolve_model(name)
        except ValueError:
            pass
        if name != self._settings.model:
            log.warning("Модель %r не найдена в аккаунте, используется %r.", name, self._settings.model)
        try:
            return self._client.resolve_model(self._settings.model)
        except ValueError:
            log.warning(
                "Модель %r не найдена в аккаунте, используется модель Gemini по умолчанию.",
                self._settings.model,
            )
            return None

    async def generate(self, prompt: str, requested_model: str | None) -> AsyncIterator[str]:
        """Отдаёт текстовые дельты ответа Gemini по мере их поступления."""
        async with self._pacer.slot():
            await self._open_session()
            model = self._resolve_model(requested_model)
            started = time.monotonic()
            stream = self._client.generate_content_stream(
                prompt,
                model=model,
                temporary=self._settings.temporary_chat,
            )
            try:
                async for output in stream:
                    delta = output.text_delta
                    if delta:
                        yield delta
            except Exception:
                # Сбрасываем сессию: следующий запрос заново получит SNlM0e и актуальные куки.
                await self._client.close()
                raise
            finally:
                # Закрывает внутренний генератор и при отмене, например при разрыве соединения клиентом.
                await stream.aclose()
            log.info(
                "Запрос к Gemini выполнен за %.1f с, промпт %d симв.",
                time.monotonic() - started,
                len(prompt),
            )


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def _sse_frame(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _sse_events(
    deltas: AsyncIterator[str],
    first: str | None,
    completion_id: str,
    created: int,
    model: str,
    detector: ToolCallDetector,
) -> AsyncIterator[str]:
    def frame(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        return _sse_frame(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )

    def render(events: list[DetectorEvent]) -> list[str]:
        frames: list[str] = []
        for event in events:
            if isinstance(event, ContentEvent):
                frames.append(frame({"content": event.text}))
            else:
                frames.append(
                    frame(
                        {
                            "tool_calls": [
                                {
                                    "index": event.index,
                                    "id": event.id,
                                    "type": "function",
                                    "function": {"name": event.name, "arguments": event.arguments},
                                }
                            ]
                        }
                    )
                )
        return frames

    try:
        yield frame({"role": "assistant", "content": ""})
        if first:
            for chunk in render(detector.feed(first)):
                yield chunk
        try:
            async for delta in deltas:
                for chunk in render(detector.feed(delta)):
                    yield chunk
            for chunk in render(detector.finish()):
                yield chunk
        except Exception as exc:
            error = to_bridge_error(exc)
            # Заголовки 200 уже отправлены, поэтому ошибка передаётся внутри потока.
            yield _sse_frame({"error": {"message": error.message, "type": error.err_type}})
            yield "data: [DONE]\n\n"
            return
        finish_reason = "tool_calls" if detector.calls_emitted else "stop"
        yield frame({}, finish_reason=finish_reason)
        yield "data: [DONE]\n\n"
    finally:
        await deltas.aclose()


def create_app(settings: Settings) -> FastAPI:
    started_at = int(time.time())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        bridge = GeminiBridge(settings)
        app.state.bridge = bridge
        await bridge.start()
        try:
            yield
        finally:
            await bridge.shutdown()

    app = FastAPI(title="gemini-web-bridge", version="1.1.0", lifespan=lifespan)
    app.state.settings = settings

    async def require_api_key(request: Request) -> None:
        if settings.api_key is None:
            return
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(token.strip().encode(), settings.api_key.encode()):
            raise BridgeError(
                401,
                "Неверный или отсутствующий ключ. Передайте заголовок 'Authorization: Bearer <BRIDGE_API_KEY>'.",
                "invalid_api_key",
            )

    @app.exception_handler(BridgeError)
    async def _bridge_error_handler(_: Request, exc: BridgeError) -> JSONResponse:
        return openai_error_response(exc.status_code, exc.message, exc.err_type)

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = "; ".join(f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors())
        return openai_error_response(400, f"Некорректное тело запроса: {details}", "invalid_request_error")

    @app.get("/healthz")
    async def healthz(request: Request) -> JSONResponse:
        bridge: GeminiBridge = request.app.state.bridge
        body = {
            "status": "ok" if bridge.ready else "degraded",
            "account_status": bridge.account_status,
            "default_model": settings.model,
            "last_error": bridge.last_error,
        }
        return JSONResponse(body, status_code=200 if bridge.ready else 503)

    @app.get("/v1/models", dependencies=[Depends(require_api_key)])
    async def list_models(request: Request) -> dict[str, Any]:
        bridge: GeminiBridge = request.app.state.bridge
        ids: list[str] = []
        for name in [settings.model, *bridge.available_models()]:
            if name not in ids:
                ids.append(name)
        return {
            "object": "list",
            "data": [{"id": name, "object": "model", "created": started_at, "owned_by": "google"} for name in ids],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
    async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
        bridge: GeminiBridge = request.app.state.bridge
        plan = resolve_tools(body.tools, body.tool_choice)
        prompt = build_prompt(body.messages, plan)
        detector = ToolCallDetector(plan.specs if plan.enabled else None)
        model_name = body.model or settings.model
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        deltas = bridge.generate(prompt, body.model)

        if not body.stream:
            events: list[DetectorEvent] = []
            try:
                async for delta in deltas:
                    events.extend(detector.feed(delta))
                events.extend(detector.finish())
            except Exception as exc:
                raise to_bridge_error(exc) from exc

            text = "".join(event.text for event in events if isinstance(event, ContentEvent))
            calls = [event for event in events if isinstance(event, ToolCallEvent)]
            message: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                message["content"] = text.strip() or None
                message["tool_calls"] = [tool_call_object(call) for call in calls]
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if calls else "stop",
                    }
                ],
            }

        # Первый фрагмент ждём до отправки заголовков: так ошибки до начала ответа
        # (неверные куки, лимит) возвращаются с правильным HTTP-статусом.
        try:
            first = await anext(deltas, None)
        except Exception as exc:
            raise to_bridge_error(exc) from exc

        return StreamingResponse(
            _sse_events(deltas, first, completion_id, created, model_name, detector),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def main() -> None:
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        sys.exit(2)

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cache_dir = Path(os.environ["GEMINI_COOKIE_PATH"])
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with suppress(OSError):
        cache_dir.chmod(0o700)

    if settings.api_key is None and settings.host not in {"127.0.0.1", "localhost", "::1"}:
        log.warning(
            "BRIDGE_API_KEY не задан, а HOST=%s: любой, кто видит порт, сможет пользоваться вашей сессией Gemini.",
            settings.host,
        )

    log.info("Запуск gemini-web-bridge на http://%s:%d", settings.host, settings.port)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
