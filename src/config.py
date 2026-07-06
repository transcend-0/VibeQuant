"""Workspace and data-directory configuration.

Layout:
    workspace/            run artifacts, experiment log, memory bank
    data/
    ├── raw/<source>/     downloaded original bars (e.g. raw/etf/)
    ├── factors/          computed factor panels + registry.jsonl
    └── sample/           bundled sample CSVs
    config/               llm.yaml, email.yaml (user-editable, gitignored)
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def workspace_dir() -> Path:
    """Workspace root: $VIBEQUANT_WORKSPACE or <repo>/workspace."""
    env = os.environ.get("VIBEQUANT_WORKSPACE")
    path = Path(env) if env else PACKAGE_ROOT / "workspace"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """Data root: $VIBEQUANT_DATA or <repo>/data."""
    env = os.environ.get("VIBEQUANT_DATA")
    path = Path(env) if env else PACKAGE_ROOT / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_data_dir(source: str = "etf") -> Path:
    """Original downloaded bars, grouped by source."""
    path = data_dir() / "raw" / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def factor_data_dir() -> Path:
    """Computed factor panels + registry.jsonl."""
    path = data_dir() / "factors"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_dir() -> Path:
    """User-editable service configs (LLM, email)."""
    path = PACKAGE_ROOT / "config"
    path.mkdir(parents=True, exist_ok=True)
    return path
