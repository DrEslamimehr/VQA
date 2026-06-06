from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML configs without requiring PyYAML."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional dep
            raise ValueError(
                f"{path} is not JSON-compatible and PyYAML is not installed"
            ) from exc
        return yaml.safe_load(text)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def deep_get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = config
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(config: dict[str, Any], key: str) -> Path:
    raw = deep_get(config, f"paths.{key}")
    if raw is None:
        raise KeyError(f"missing paths.{key}")
    path = Path(raw)
    return path if path.is_absolute() else project_root() / path

