"""Tests for vLLM tool calling error handling."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import NotFoundError

from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.model._providers._vllm_lora import cleanup_servers
from inspect_ai.model._providers.vllm import VLLMAPI
from inspect_ai.tool import ToolInfo, ToolParams


@pytest.fixture(autouse=True)
def _clean_vllm_servers():
    cleanup_servers()
    yield
    cleanup_servers()


def _make_not_found_error() -> NotFoundError:
    request = httpx.Request("POST", "http://localhost:9999/v1/chat/completions")
    response = httpx.Response(404, request=request)
    return NotFoundError(
        message="Not Found",
        response=response,
        body={"detail": "Not Found"},
    )


def _make_api() -> VLLMAPI:
    return VLLMAPI(
        "some-model",
        base_url="http://localhost:9999/v1",
    )


def _make_tool() -> ToolInfo:
    return ToolInfo(
        name="test_tool",
        description="A test tool",
        parameters=ToolParams(properties={}),
    )


class TestVLLMToolCallingErrors:
    @pytest.mark.anyio
    async def test_404_with_tools_raises_actionable_error(self) -> None:
        """NotFoundError with tools gives RuntimeError with actionable message."""
        api = _make_api()

        with patch(
            "inspect_ai.model._providers.openai_compatible.OpenAICompatibleAPI.generate",
            new=AsyncMock(side_effect=_make_not_found_error()),
        ):
            with pytest.raises(RuntimeError, match="--enable-auto-tool-choice"):
                await api.generate(
                    input=[ChatMessageUser(content="call a tool")],
                    tools=[_make_tool()],
                    tool_choice="auto",
                    config=GenerateConfig(),
                )

    @pytest.mark.anyio
    async def test_404_without_tools_reraises_original(self) -> None:
        """NotFoundError without tools is reraised unchanged."""
        api = _make_api()

        with patch(
            "inspect_ai.model._providers.openai_compatible.OpenAICompatibleAPI.generate",
            new=AsyncMock(side_effect=_make_not_found_error()),
        ):
            with pytest.raises(NotFoundError):
                await api.generate(
                    input=[ChatMessageUser(content="hello")],
                    tools=[],
                    tool_choice="none",
                    config=GenerateConfig(),
                )
