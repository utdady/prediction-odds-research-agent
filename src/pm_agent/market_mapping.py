from __future__ import annotations

from pathlib import Path
import yaml


def load_mapping(path: str = "config/market_entity_map.yml") -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return yaml.safe_load(p.read_text(encoding="utf-8")) or []

