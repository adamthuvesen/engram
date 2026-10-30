"""Read persisted JSONL rows with explicit corruption counts."""

import logging
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from engram.storage.transactions import StoreIntegrityError

logger = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_jsonl_records(
    path: Path, model: type[ModelT], corrupt_label: str, *, strict: bool = False
) -> tuple[list[ModelT], int]:
    """Parse a JSONL file into ``model`` records, returning ``(records, corrupt)``."""
    if not path.exists():
        return [], 0

    records: list[ModelT] = []
    corrupt = 0
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(model.model_validate_json(line))
            except (ValueError, ValidationError) as exc:
                if strict:
                    raise StoreIntegrityError(
                        path, lineno, f"invalid {corrupt_label} record"
                    ) from exc
                corrupt += 1
                logger.warning(
                    "Invalid %s at line %d in %s",
                    corrupt_label,
                    lineno,
                    str(path),
                )
    return records, corrupt
