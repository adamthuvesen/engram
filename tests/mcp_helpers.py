"""Public FastMCP test helpers."""

from typing import Any

from fastmcp import FastMCP
from mcp.types import ContentBlock


async def call_tool(
    app: FastMCP,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[list[ContentBlock], dict[str, Any]]:
    """Call a tool through FastMCP's public API and expose its wire payload."""
    result = await app.call_tool(tool_name, arguments)
    assert result.structured_content is not None, f"{tool_name} returned no structured content"
    return result.content, result.structured_content
