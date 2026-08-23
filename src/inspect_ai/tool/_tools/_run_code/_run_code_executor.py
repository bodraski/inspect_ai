from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, get_args

from pydantic import TypeAdapter

from inspect_ai._util.content import Content, ContentText
from inspect_ai._util.error import pip_dependency_error

from ..._tool import ToolError
from ..._tool_def import ToolDef
from ._bridge import (
    RunCodeInnerToolCallTraceEntry,
    RunCodeToolBridge,
    _is_content,
    _is_content_list,
)

_content_item_adapter: TypeAdapter[Content] = TypeAdapter(Content)

_content_types = {cls.model_fields["type"].default for cls in get_args(Content)}


def _looks_like_content(value: dict[str, Any]) -> bool:
    return value.get("type") in _content_types


def _restore(value: Any) -> Any:
    if isinstance(value, dict) and _looks_like_content(value):
        return _content_item_adapter.validate_python(value)
    if isinstance(value, list):
        return [_restore(v) for v in value]
    if isinstance(value, dict):
        return {k: _restore(v) for k, v in value.items()}
    return value


def _reconstruct_content(value: Any) -> list[Content]:
    """Restoring serialized Content in place."""
    restored = _restore(value)
    if _is_content_list(restored):
        return restored
    if _is_content(restored):
        return [restored]
    return [
        ContentText(
            text=json.dumps(restored) if not isinstance(restored, str) else restored
        )
    ]


@dataclass
class RunCodeResult:
    """Result of a run_code execution."""

    output: list[Content]
    error: str | None = None
    inner_tool_call_trace: list[RunCodeInnerToolCallTraceEntry] = field(
        default_factory=list
    )


class RunCodeExecutor(Protocol):
    """Executor for run_code."""

    async def execute(self, code: str) -> RunCodeResult:
        """Execute code.

        Args:
            code: Python code to execute.

        Returns:
            Result of the code execution.
        """
        ...


class StubRunCodeExecutor:
    """Placeholder executor used until real code execution is implemented."""

    async def execute(self, code: str) -> RunCodeResult:
        return RunCodeResult(
            output=[ContentText(text="run_code execution is not implemented yet")]
        )


class MontyRunCodeExecutor:
    """Run code using Pydantic Monty."""

    def __init__(
        self,
        tool_defs: list[ToolDef] | None = None,
        *,
        max_inner_tool_calls: int | None = None,
    ) -> None:
        self.tool_defs = tool_defs or []
        self.max_tool_calls = max_inner_tool_calls

    async def execute(self, code: str) -> RunCodeResult:
        try:
            import pydantic_monty
            from pydantic_monty import MontyError
        except ImportError:
            raise pip_dependency_error("run_code", ["pydantic-monty"])

        bridge = RunCodeToolBridge(
            self.tool_defs,
            max_inner_tool_calls=self.max_tool_calls,
        )

        try:
            monty = pydantic_monty.Monty(
                code,
                script_name="run_code.py",
                type_check=False,
            )
            output = await monty.run_async(
                external_functions=bridge.external_functions(),
            )

            contents: list[Content] = []

            if output is not None:
                contents.extend(_reconstruct_content(output))
            return RunCodeResult(
                output=contents,
                inner_tool_call_trace=bridge.call_trace,
            )
        except MontyError as exc:
            raise ToolError(str(exc))
        except Exception as exc:
            return RunCodeResult(
                output=[],
                error=str(exc),
                inner_tool_call_trace=bridge.call_trace,
            )
