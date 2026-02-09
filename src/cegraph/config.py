"""Configuration management for CeGraph."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

CEGRAPH_DIR = ".cegraph"
CONFIG_FILE = "config.json"
GRAPH_DB_FILE = "graph.db"
INDEX_FILE = "index.json"


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5-20250929"
    api_key_env: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0
    base_url: str | None = None

    @property
    def api_key(self) -> str | None:
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        # Try common env vars
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = env_map.get(self.provider, "")
        return os.environ.get(env_var)


class AgentConfig(BaseModel):
    """Agent behavior configuration."""

    max_iterations: int = 15
    auto_verify: bool = True
    require_approval: bool = True


class IndexerConfig(BaseModel):
    """Indexer configuration."""

    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "node_modules",
            "__pycache__",
            ".git",
            ".cegraph",
            "dist",
            "build",
            ".venv",
            "venv",
            ".env",
            "*.pyc",
            "*.pyo",
            "*.so",
            "*.dylib",
            "*.dll",
            "*.exe",
            "*.min.js",
            "*.min.css",
            "*.map",
            "*.lock",
            "package-lock.json",
            "yarn.lock",
        ]
    )
    max_file_size_kb: int = 500
    languages: list[str] = Field(default_factory=list)  # empty = auto-detect


class ProjectConfig(BaseModel):
    """Full project configuration."""

    name: str = ""
    root_path: str = "."
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk up from `start` looking for a .cegraph directory."""
    current = (start or Path.cwd()).resolve()
    while current != current.parent:
        if (current / CEGRAPH_DIR).is_dir():
            return current
        current = current.parent
    if (current / CEGRAPH_DIR).is_dir():
        return current
    return None


def get_cegraph_dir(root: Path) -> Path:
    """Get the .cegraph directory for a project root."""
    return root / CEGRAPH_DIR


def load_config(root: Path) -> ProjectConfig:
    """Load configuration from .cegraph/config.json."""
    config_path = get_cegraph_dir(root) / CONFIG_FILE
    if config_path.exists():
        data = json.loads(config_path.read_text())
        return ProjectConfig(**data)
    return ProjectConfig(name=root.name, root_path=str(root))


def save_config(root: Path, config: ProjectConfig) -> None:
    """Save configuration to .cegraph/config.json."""
    cs_dir = get_cegraph_dir(root)
    cs_dir.mkdir(parents=True, exist_ok=True)
    config_path = cs_dir / CONFIG_FILE
    config_path.write_text(json.dumps(config.model_dump(), indent=2))


def set_config_value(config: ProjectConfig, key: str, value: Any) -> ProjectConfig:
    """Set a nested config value using dot notation (e.g., 'llm.provider')."""
    parts = key.split(".")
    data = config.model_dump()
    target = data
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise KeyError(f"Invalid config key: {key}")
        target = target[part]
    if parts[-1] not in target:
        raise KeyError(f"Invalid config key: {key}")
    target[parts[-1]] = value
    return ProjectConfig(**data)
