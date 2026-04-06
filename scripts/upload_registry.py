"""
Upload registry — maps image stem → list of GYD receipt UUIDs.

Written by upload() in etl.py after each successful create() call so that
delete_receipts.py can look up and remove records from the database.

File location: output/.upload_registry.json
"""

import json
from pathlib import Path

REGISTRY_PATH = Path("output") / ".upload_registry.json"


def load() -> dict:
    """Return the full registry dict, or {} if the file is missing/corrupt."""
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save(image_stem: str, ids: list[str]) -> None:
    """Overwrite the ID list for *image_stem* in the registry."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry = load()
    registry[image_stem] = ids
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False),
                             encoding="utf-8")


def remove(image_stem: str) -> None:
    """Remove *image_stem* from the registry (called after successful delete)."""
    registry = load()
    registry.pop(image_stem, None)
    if REGISTRY_PATH.exists():
        REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
