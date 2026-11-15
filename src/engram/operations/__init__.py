"""Shared application operations for Engram's MCP and CLI adapters."""

from engram.operations.candidates import approve_candidates, list_candidates, reject_candidates
from engram.operations.facts import (
    correct_memory,
    edit_fact,
    forget,
    inspect,
    mark_stale,
    merge_memories,
    purge,
    rename_project,
    unmark_stale,
)
from engram.operations.maintenance import audit_memories, doctor, memory_stats, sync, upkeep
from engram.operations.memories import import_memories, remember, suggest_memories
from engram.operations.recall import (
    RECALL_CONTEXT_MODES,
    RECALL_MODES,
    recall,
    recall_context,
    recall_stats,
    recall_trace,
)
from engram.operations.results import (
    EXIT_DOCTOR_ERROR,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_VALIDATION,
    OperationResult,
    exception_result,
    invalid_format_result,
)
from engram.operations.stores import async_store

__all__ = [
    "EXIT_DOCTOR_ERROR",
    "EXIT_NOT_FOUND",
    "EXIT_OK",
    "EXIT_RUNTIME",
    "EXIT_VALIDATION",
    "RECALL_CONTEXT_MODES",
    "RECALL_MODES",
    "OperationResult",
    "approve_candidates",
    "async_store",
    "audit_memories",
    "correct_memory",
    "doctor",
    "edit_fact",
    "exception_result",
    "forget",
    "import_memories",
    "inspect",
    "invalid_format_result",
    "list_candidates",
    "mark_stale",
    "memory_stats",
    "merge_memories",
    "purge",
    "recall",
    "recall_context",
    "recall_stats",
    "recall_trace",
    "reject_candidates",
    "remember",
    "rename_project",
    "suggest_memories",
    "sync",
    "unmark_stale",
    "upkeep",
]
