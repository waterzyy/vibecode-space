"""gemini-web-bridge: локальный OpenAI-совместимый HTTP-прокси к веб-приложению Gemini.

Прокси работает от имени вашей авторизованной сессии gemini.google.com (куки
``__Secure-1PSID`` и ``__Secure-1PSIDTS``, подписка Google AI Pro) и публикует:

    POST /v1/chat/completions   JSON-ответ или SSE при ``"stream": true``
    GET  /v1/models             список моделей аккаунта
    GET  /healthz               состояние веб-сессии

Запуск: ``python server.py``. Настройки читаются из ``.env`` рядом с этим файлом.
"""

import asyncio
import json
import logging
import os
import random
import secrets
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
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


class ChatCompletionRequest(BaseModel):
    # Поля вроде temperature, max_tokens, tools принимаются и игнорируются:
    # веб-интерфейс Gemini не предоставляет их управления.
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False


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


def build_prompt(messages: list[ChatMessage]) -> str:
    """Собирает промпт для Gemini из списка сообщений OpenAI.

    Единственное сообщение user без system отправляется как есть. Иначе история
    склеивается в транскрипт с системными инструкциями, а модели выдаётся указание
    ответить только на последнее сообщение пользователя.
    """
    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in messages:
        role = message.role
        text = message_text(message).strip()
        if role in {"system", "developer"}:
            if text:
                system_parts.append(text)
        elif role in {"user", "assistant", "tool"}:
            if text:
                turns.append((role, text))
        else:
            raise BridgeError(400, f"Неизвестная роль сообщения: {role!r}.", "invalid_request_error")

    if not turns:
        raise BridgeError(400, "В messages нет непустых сообщений user, assistant или tool.", "invalid_request_error")

    if len(turns) == 1 and turns[0][0] == "user" and not system_parts:
        return turns[0][1]

    labels = {"user": "User", "assistant": "Assistant", "tool": "Tool result"}
    blocks = [PREAMBLE]
    if system_parts:
        blocks.append("### System instructions\n" + "\n\n".join(system_parts))
    blocks.append("### Conversation\n" + "\n\n".join(f"{labels[role]}: {text}" for role, text in turns))
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
) -> AsyncIterator[str]:
    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        return _sse_frame(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )

    try:
        yield chunk({"role": "assistant", "content": ""})
        if first:
            yield chunk({"content": first})
        try:
            async for delta in deltas:
                yield chunk({"content": delta})
        except Exception as exc:
            error = to_bridge_error(exc)
            # Заголовки 200 уже отправлены, поэтому ошибка передаётся внутри потока.
            yield _sse_frame({"error": {"message": error.message, "type": error.err_type}})
            yield "data: [DONE]\n\n"
            return
        yield chunk({}, finish_reason="stop")
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

    app = FastAPI(title="gemini-web-bridge", version="1.0.0", lifespan=lifespan)
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
        prompt = build_prompt(body.messages)
        model_name = body.model or settings.model
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        deltas = bridge.generate(prompt, body.model)

        if not body.stream:
            try:
                text = "".join([delta async for delta in deltas])
            except Exception as exc:
                raise to_bridge_error(exc) from exc
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
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
            _sse_events(deltas, first, completion_id, created, model_name),
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
