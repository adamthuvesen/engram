"""Public FastMCP test helpers."""

from typing import Any

from fastmcp import FastMCP
from mcp.types import ContentBlock


async def call_tool(
    app: FastMCP,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[list[ContentBlock], dict[str, Any] | None]:
    """Call a tool through FastMCP's public API and expose its wire payload."""
    result = await app.call_tool(tool_name, arguments)
    return result.content, result.structured_content
