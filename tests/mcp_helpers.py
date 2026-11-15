from typing import Any, cast

from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from mcp.types import ContentBlock, TextContent


async def call_tool(
    app: FastMCP,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[list[ContentBlock], dict[str, Any]]:
    """Call a tool through FastMCP's public API and expose its wire payload."""
    result = await app.call_tool(tool_name, arguments)
    assert result.structured_content is not None, f"{tool_name} returned no structured content"
    return result.content, result.structured_content


def json_object(value: object) -> dict[str, Any]:
    """Narrow a decoded JSON value to an object."""
    assert isinstance(value, dict)
    return cast("dict[str, Any]", value)


def json_array(value: object) -> list[Any]:
    """Narrow a decoded JSON value to an array."""
    assert isinstance(value, list)
    return cast("list[Any]", value)


def structured_content(result: ToolResult) -> dict[str, Any]:
    """Return a tool result's structured payload, which error results also carry."""
    assert result.structured_content is not None
    return result.structured_content


def text_of(block: ContentBlock) -> str:
    assert isinstance(block, TextContent)
    return block.text
