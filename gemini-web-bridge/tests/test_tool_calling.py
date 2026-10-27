"""Эмуляция tool calling: инъекция tools, история role=tool, разбор <tool_call> (обычный и стриминг)."""

import json

import pytest
from fastapi.testclient import TestClient
from support import make_settings, sse_events, stream_text_of

import server

CHAT = "/v1/chat/completions"

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
            "required": ["city"],
        },
    },
}
TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get local time",
        "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}},
    },
}
WEATHER_CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def specs_for(*tools) -> dict[str, server.ToolSpec]:
    return server.normalize_tools(list(tools))


def run_detector(detector: server.ToolCallDetector, pieces: list[str]) -> list[server.DetectorEvent]:
    events: list[server.DetectorEvent] = []
    for piece in pieces:
        events.extend(detector.feed(piece))
    events.extend(detector.finish())
    return events


def content_of(events) -> str:
    return "".join(e.text for e in events if isinstance(e, server.ContentEvent))


def calls_of(events) -> list[server.ToolCallEvent]:
    return [e for e in events if isinstance(e, server.ToolCallEvent)]


# ---------------------------------------------------------------------------
# Схемы tools и tool_choice
# ---------------------------------------------------------------------------


def test_normalize_tools_builds_specs():
    specs = server.normalize_tools([WEATHER, TIME])
    assert list(specs) == ["get_weather", "get_time"]
    assert specs["get_weather"].required == ("city",)
    assert specs["get_time"].required == ()
    assert specs["get_time"].description == "Get local time"


@pytest.mark.parametrize(
    "bad_tool",
    [
        {"type": "retrieval", "function": {"name": "x"}},
        {"type": "function"},
        {"type": "function", "function": {"description": "no name"}},
        {"type": "function", "function": {"name": "x", "parameters": "not-a-schema"}},
    ],
    ids=["wrong-type", "no-function", "no-name", "bad-parameters"],
)
def test_invalid_tool_schema_returns_400(bad_tool):
    with pytest.raises(server.BridgeError) as exc:
        server.normalize_tools([bad_tool])
    assert exc.value.status_code == 400


def test_duplicate_tool_names_return_400():
    with pytest.raises(server.BridgeError):
        server.normalize_tools([WEATHER, WEATHER])


@pytest.mark.parametrize(
    ("tool_choice", "mode", "enabled"),
    [
        (None, "auto", True),
        ("auto", "auto", True),
        ("required", "required", True),
        ("none", "none", False),
    ],
)
def test_resolve_tools_modes(tool_choice, mode, enabled):
    plan = server.resolve_tools([WEATHER, TIME], tool_choice)
    assert plan.mode == mode
    assert plan.enabled is enabled


def test_forced_tool_choice_restricts_specs():
    plan = server.resolve_tools([WEATHER, TIME], {"type": "function", "function": {"name": "get_time"}})
    assert plan.mode == "forced"
    assert plan.forced == "get_time"
    assert list(plan.specs) == ["get_time"]


def test_no_tools_disables_plan():
    plan = server.resolve_tools(None, None)
    assert plan.enabled is False


@pytest.mark.parametrize(
    ("tools", "tool_choice"),
    [
        ([WEATHER], {"type": "function", "function": {"name": "missing"}}),
        (None, "required"),
        ([WEATHER], "sometimes"),
    ],
    ids=["unknown-forced-name", "required-without-tools", "unknown-string"],
)
def test_invalid_tool_choice_returns_400(tools, tool_choice):
    with pytest.raises(server.BridgeError) as exc:
        server.resolve_tools(tools, tool_choice)
    assert exc.value.status_code == 400


def test_tool_instructions_contain_format_and_schemas():
    plan = server.resolve_tools([WEATHER], None)
    text = server.tool_instructions(plan)
    assert text.startswith("## Tool calling")
    assert "<tool_call>" in text and "</tool_call>" in text
    assert '"name": "get_weather"' in text
    assert '"required": [' in text
    assert "Call a function only when it is needed" in text


def test_tool_instructions_mode_requirements():
    required = server.tool_instructions(server.resolve_tools([WEATHER], "required"))
    forced = server.tool_instructions(
        server.resolve_tools([WEATHER], {"type": "function", "function": {"name": "get_weather"}})
    )
    assert "MUST call one of" in required
    assert 'MUST call the function "get_weather"' in forced


# ---------------------------------------------------------------------------
# Сборка промпта: инъекция и история role=tool
# ---------------------------------------------------------------------------


def test_tools_are_injected_first_in_system_block():
    plan = server.resolve_tools([WEATHER], None)
    prompt = server.build_prompt(
        [server.ChatMessage(role="system", content="Be terse."), server.ChatMessage(role="user", content="Weather?")],
        plan,
    )
    assert "### System instructions\n## Tool calling" in prompt
    assert prompt.index("## Tool calling") < prompt.index("Be terse.")
    assert "get_weather" in prompt


def test_tools_force_transcript_even_for_single_user_message():
    plan = server.resolve_tools([WEATHER], None)
    prompt = server.build_prompt([server.ChatMessage(role="user", content="Weather?")], plan)
    assert prompt != "Weather?"
    assert "User: Weather?" in prompt


def test_tool_result_is_rendered_with_name_and_id():
    prompt = server.build_prompt(
        [
            server.ChatMessage(role="user", content="Weather?"),
            server.ChatMessage(role="tool", content="21C", tool_call_id="call_1", name="get_weather"),
        ]
    )
    assert "Tool Result (get_weather / call_1): 21C" in prompt


def test_tool_result_name_is_looked_up_from_history():
    prompt = server.build_prompt(
        [
            server.ChatMessage(role="user", content="Weather?"),
            server.ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            ),
            server.ChatMessage(role="tool", content="21C", tool_call_id="call_abc"),
        ]
    )
    assert "Tool Result (get_weather / call_abc): 21C" in prompt
    assert '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>' in prompt


def test_legacy_function_role_is_supported():
    prompt = server.build_prompt(
        [
            server.ChatMessage(role="user", content="Weather?"),
            server.ChatMessage(role="function", content="21C", name="get_weather"),
        ]
    )
    assert "Tool Result (get_weather): 21C" in prompt


def test_tool_message_without_name_or_id_uses_unknown_label():
    prompt = server.build_prompt(
        [server.ChatMessage(role="user", content="q"), server.ChatMessage(role="tool", content="r")]
    )
    assert "Tool Result (unknown): r" in prompt


# ---------------------------------------------------------------------------
# Разбор блоков: parse_tool_call
# ---------------------------------------------------------------------------


SPECS = specs_for(WEATHER, TIME)


def test_parse_valid_block():
    name, arguments = server.parse_tool_call('{"name": "get_weather", "arguments": {"city": "Paris"}}', SPECS)
    assert name == "get_weather"
    assert json.loads(arguments) == {"city": "Paris"}


def test_parse_accepts_code_fence_inside_tags():
    body = '\n```json\n{"name": "get_time", "arguments": {}}\n```\n'
    assert server.parse_tool_call(body, SPECS) == ("get_time", "{}")


def test_parse_accepts_arguments_as_json_string():
    name, arguments = server.parse_tool_call('{"name": "get_weather", "arguments": "{\\"city\\": \\"Rome\\"}"}', SPECS)
    assert json.loads(arguments) == {"city": "Rome"}


@pytest.mark.parametrize(
    "body",
    [
        '{"name": "get_weather", "arguments": {"city": "Paris"}',  # битый JSON
        '{"name": "delete_db", "arguments": {}}',  # неизвестная функция
        '{"name": "get_weather", "arguments": {}}',  # нет обязательного city
        '{"name": "get_weather", "arguments": [1, 2]}',  # аргументы не объект
        '["get_weather"]',  # не объект
        "Sure, I will call the weather tool now.",  # текст вместо JSON
    ],
    ids=["broken-json", "unknown-name", "missing-required", "array-arguments", "not-object", "plain-text"],
)
def test_parse_rejects_invalid_blocks(body):
    assert server.parse_tool_call(body, SPECS) is None


# ---------------------------------------------------------------------------
# Детектор: нестриминговый разбор
# ---------------------------------------------------------------------------


def test_detector_extracts_call_and_keeps_text_before_it():
    events = run_detector(server.ToolCallDetector(SPECS), ["Checking.\n" + WEATHER_CALL])
    calls = calls_of(events)
    assert content_of(events) == "Checking.\n"
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].index == 0
    assert calls[0].id.startswith("call_")
    assert json.loads(calls[0].arguments) == {"city": "Paris"}


def test_detector_handles_multiple_calls_with_indexes():
    text = WEATHER_CALL + " and " + '<tool_call>{"name": "get_time", "arguments": {"tz": "UTC"}}</tool_call>'
    events = run_detector(server.ToolCallDetector(SPECS), [text])
    calls = calls_of(events)
    assert [c.index for c in calls] == [0, 1]
    assert [c.name for c in calls] == ["get_weather", "get_time"]
    assert content_of(events) == " and "


def test_detector_returns_invalid_block_as_plain_text():
    text = 'Try this: <tool_call>{"name": "get_weather", "arguments": {"city": </tool_call> done'
    events = run_detector(server.ToolCallDetector(SPECS), [text])
    assert calls_of(events) == []
    assert content_of(events) == text


def test_detector_returns_unknown_function_as_plain_text():
    text = '<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>'
    events = run_detector(server.ToolCallDetector(SPECS), [text])
    assert calls_of(events) == []
    assert content_of(events) == text


def test_unclosed_block_is_flushed_as_text_on_finish():
    text = 'Hi <tool_call>{"name": "get_weather"'
    events = run_detector(server.ToolCallDetector(SPECS), [text])
    assert calls_of(events) == []
    assert content_of(events) == text


def test_disabled_detector_is_transparent():
    detector = server.ToolCallDetector(None)
    assert detector.feed(WEATHER_CALL) == [server.ContentEvent(WEATHER_CALL)]
    assert detector.finish() == []


# ---------------------------------------------------------------------------
# Детектор: стриминг
# ---------------------------------------------------------------------------


def test_stream_char_by_char_hides_tag_and_emits_call_once():
    text = "Checking.\n" + WEATHER_CALL
    events = run_detector(server.ToolCallDetector(SPECS), list(text))
    calls = calls_of(events)
    assert content_of(events) == "Checking.\n"
    assert "<tool_call>" not in content_of(events)
    assert len(calls) == 1
    assert json.loads(calls[0].arguments) == {"city": "Paris"}


def test_stream_holds_back_only_possible_tag_prefix():
    detector = server.ToolCallDetector(SPECS)
    assert detector.feed("a<") == [server.ContentEvent("a")]
    assert detector.feed("b") == [server.ContentEvent("<b")]


def test_stream_without_calls_passes_text_without_delay():
    detector = server.ToolCallDetector(SPECS)
    assert detector.feed("Plain answer") == [server.ContentEvent("Plain answer")]
    assert detector.feed(" continues") == [server.ContentEvent(" continues")]
    assert detector.finish() == []


def test_stream_invalid_block_falls_back_to_content():
    text = 'x <tool_call>{"name": "get_weather", "arguments": nope}</tool_call> y'
    events = run_detector(server.ToolCallDetector(SPECS), list(text))
    assert calls_of(events) == []
    assert content_of(events) == text


# ---------------------------------------------------------------------------
# HTTP: нестриминговый ответ
# ---------------------------------------------------------------------------


def test_non_stream_tool_call_response(client, fake):
    fake.chunks = ["Let me check.\n" + WEATHER_CALL]
    response = client.post(CHAT, json={"messages": [user("Weather in Paris?")], "tools": [WEATHER]})
    body = response.json()
    choice = body["choices"][0]
    message = choice["message"]

    assert response.status_code == 200
    assert choice["finish_reason"] == "tool_calls"
    assert message["content"] == "Let me check."
    assert len(message["tool_calls"]) == 1
    call = message["tool_calls"][0]
    assert call["id"].startswith("call_")
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}


def test_non_stream_tool_call_without_text_has_null_content(client, fake):
    fake.chunks = [WEATHER_CALL]
    body = client.post(CHAT, json={"messages": [user("w?")], "tools": [WEATHER]}).json()
    assert body["choices"][0]["message"]["content"] is None


def test_non_stream_invalid_json_falls_back_to_content(client, fake):
    raw = '<tool_call>\n{"name": "get_weather", "arguments": {"city": }\n</tool_call>'
    fake.chunks = [raw]
    body = client.post(CHAT, json={"messages": [user("w?")], "tools": [WEATHER]}).json()
    message = body["choices"][0]["message"]
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in message
    assert message["content"] == raw


def test_non_stream_without_tools_keeps_tag_as_text(client, fake):
    fake.chunks = [WEATHER_CALL]
    body = client.post(CHAT, json={"messages": [user("w?")]}).json()
    assert body["choices"][0]["message"]["content"] == WEATHER_CALL
    assert body["choices"][0]["finish_reason"] == "stop"


def test_non_stream_parallel_calls(client, fake):
    fake.chunks = [WEATHER_CALL, '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>']
    body = client.post(CHAT, json={"messages": [user("both")], "tools": [WEATHER, TIME]}).json()
    calls = body["choices"][0]["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["get_weather", "get_time"]
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_choice_none_disables_emulation(client, fake):
    fake.chunks = [WEATHER_CALL]
    body = client.post(CHAT, json={"messages": [user("w?")], "tools": [WEATHER], "tool_choice": "none"}).json()
    assert body["choices"][0]["message"]["content"] == WEATHER_CALL
    assert "## Tool calling" not in fake.calls[-1]["prompt"]


def test_required_mode_is_sent_to_model(client, fake):
    client.post(CHAT, json={"messages": [user("w?")], "tools": [WEATHER], "tool_choice": "required"})
    assert "MUST call one of" in fake.calls[-1]["prompt"]


def test_forced_tool_choice_hides_other_tools(client, fake):
    client.post(
        CHAT,
        json={
            "messages": [user("t?")],
            "tools": [WEATHER, TIME],
            "tool_choice": {"type": "function", "function": {"name": "get_time"}},
        },
    )
    prompt = fake.calls[-1]["prompt"]
    assert 'MUST call the function "get_time"' in prompt
    assert '"name": "get_time"' in prompt
    assert '"name": "get_weather"' not in prompt


def test_tools_prompt_reaches_gemini_with_system_first(client, fake):
    client.post(
        CHAT,
        json={"messages": [{"role": "system", "content": "Be terse."}, user("w?")], "tools": [WEATHER]},
    )
    prompt = fake.calls[-1]["prompt"]
    assert prompt.index("### System instructions\n## Tool calling") < prompt.index("Be terse.")


def test_history_with_tool_result_reaches_gemini(client, fake):
    client.post(
        CHAT,
        json={
            "tools": [WEATHER],
            "messages": [
                user("Weather in Paris?"),
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "21C, sunny"},
            ],
        },
    )
    prompt = fake.calls[-1]["prompt"]
    assert "Tool Result (get_weather / call_1): 21C, sunny" in prompt
    assert "<tool_call>" in prompt  # вызов из истории виден модели в том же формате


@pytest.mark.parametrize(
    "payload",
    [
        {"tools": [{"type": "retrieval"}], "messages": [user("x")]},
        {
            "tools": [WEATHER],
            "tool_choice": {"type": "function", "function": {"name": "nope"}},
            "messages": [user("x")],
        },
    ],
    ids=["bad-tool-type", "unknown-forced-function"],
)
def test_invalid_tools_request_returns_400(client, payload):
    response = client.post(CHAT, json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# HTTP: стриминг
# ---------------------------------------------------------------------------


def stream_request(client, body) -> list:
    with client.stream("POST", CHAT, json={**body, "stream": True}) as response:
        assert response.status_code == 200
        raw = "".join(response.iter_text())
    return sse_events(raw)


def test_stream_tool_call_emits_tool_calls_delta_and_finish_reason(client, fake):
    # Токены приходят по одному символу: тег должен собраться внутри детектора.
    fake.chunks = list("Checking.\n" + WEATHER_CALL)
    events = stream_request(client, {"messages": [user("w?")], "tools": [WEATHER]})
    chunks = [e for e in events if isinstance(e, dict)]

    assert events[-1] == "[DONE]"
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"

    text = stream_text_of(events)
    assert text == "Checking.\n"
    assert "<tool_call>" not in text

    tool_deltas = [c["choices"][0]["delta"]["tool_calls"] for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
    assert len(tool_deltas) == 1
    call = tool_deltas[0][0]
    assert call["index"] == 0
    assert call["id"].startswith("call_")
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}

    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_stream_without_calls_is_plain_content_with_stop(client, fake):
    fake.chunks = ["Just ", "text."]
    events = stream_request(client, {"messages": [user("hi")], "tools": [WEATHER]})
    chunks = [e for e in events if isinstance(e, dict)]
    assert stream_text_of(events) == "Just text."
    assert not any("tool_calls" in c["choices"][0]["delta"] for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_stream_invalid_json_falls_back_to_content(client, fake):
    raw = '<tool_call>{"name": "get_weather", "arguments": {"city": </tool_call>'
    fake.chunks = list(raw)
    events = stream_request(client, {"messages": [user("w?")], "tools": [WEATHER]})
    chunks = [e for e in events if isinstance(e, dict)]
    assert stream_text_of(events) == raw
    assert not any("tool_calls" in c["choices"][0]["delta"] for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_stream_parallel_calls_have_sequential_indexes(client, fake):
    fake.chunks = [WEATHER_CALL, '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>']
    events = stream_request(client, {"messages": [user("both")], "tools": [WEATHER, TIME]})
    indexes = [
        call["index"] for e in events if isinstance(e, dict) for call in e["choices"][0]["delta"].get("tool_calls", [])
    ]
    assert indexes == [0, 1]


def test_stream_mid_error_still_reports_error_frame(fake):
    fake.fail = "mid"
    with TestClient(server.create_app(make_settings())) as test_client:
        events = stream_request(test_client, {"messages": [user("w?")], "tools": [WEATHER]})
    assert any(isinstance(e, dict) and "error" in e for e in events)
    assert events[-1] == "[DONE]"
