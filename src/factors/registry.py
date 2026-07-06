"""Factor library: persisted factor panels + metadata registry.

Distinct from raw price data (data/raw/): every successful factor run
saves its computed values here so factors can be browsed, compared and
reused without recomputation.

    data/factors/
    ├── registry.jsonl                  one metadata line per factor version
    └── <name>__<run_id>.csv            panel: date, symbol, value
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import factor_data_dir


def _registry_file() -> Path:
    return factor_data_dir() / "registry.jsonl"


def register_factor(
    name: str,
    expression: str,
    panel: pd.DataFrame,  # columns: date, symbol, <name>
    run_id: str,
    spec_summary: Dict[str, Any],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist one factor's values and append its metadata to the registry."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40]
    file_name = f"{safe}__{run_id}.csv"
    out = panel[["date", "symbol", name]].rename(columns={name: "value"})
    out = out.dropna(subset=["value"])
    out.to_csv(factor_data_dir() / file_name, index=False)

    entry = {
        "name": name,
        "expression": expression,
        "file": file_name,
        "run_id": run_id,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(out)),
        **spec_summary,
        "stats": stats,
    }
    with _registry_file().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry


def list_factors(limit: int = 200) -> List[Dict[str, Any]]:
    """Registry entries, newest first."""
    reg = _registry_file()
    if not reg.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for line in reg.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(reversed(entries))[:limit]


def load_factor_panel(file_name: str) -> Optional[pd.DataFrame]:
    """Load a persisted factor panel by registry file name."""
    path = (factor_data_dir() / Path(file_name).name).resolve()
    if not str(path).startswith(str(factor_data_dir().resolve())):
        return None
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])
