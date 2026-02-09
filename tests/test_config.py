"""Tests for configuration management."""

from __future__ import annotations

from pathlib import Path

import pytest

from cegraph.config import (
    ProjectConfig,
    find_project_root,
    load_config,
    save_config,
    set_config_value,
)


class TestConfig:
    def test_default_config(self):
        config = ProjectConfig()
        assert config.llm.provider == "anthropic"
        assert config.agent.max_iterations == 15
        assert len(config.indexer.exclude_patterns) > 0

    def test_save_and_load(self, tmp_path: Path):
        config = ProjectConfig(name="test-project")
        config.llm.provider = "openai"
        config.llm.model = "gpt-4o"

        save_config(tmp_path, config)
        loaded = load_config(tmp_path)

        assert loaded.name == "test-project"
        assert loaded.llm.provider == "openai"
        assert loaded.llm.model == "gpt-4o"

    def test_find_project_root(self, tmp_path: Path):
        # No .cegraph dir - should return None
        assert find_project_root(tmp_path) is None

        # Create .cegraph dir
        (tmp_path / ".cegraph").mkdir()
        assert find_project_root(tmp_path) == tmp_path

        # Should find from subdirectory
        sub = tmp_path / "src" / "module"
        sub.mkdir(parents=True)
        assert find_project_root(sub) == tmp_path

    def test_set_config_value(self):
        config = ProjectConfig()
        updated = set_config_value(config, "llm.provider", "openai")
        assert updated.llm.provider == "openai"

    def test_set_config_nested(self):
        config = ProjectConfig()
        updated = set_config_value(config, "agent.max_iterations", 20)
        assert updated.agent.max_iterations == 20

    def test_set_config_invalid_key(self):
        config = ProjectConfig()
        with pytest.raises(KeyError):
            set_config_value(config, "nonexistent.key", "value")

    def test_exclude_patterns(self):
        config = ProjectConfig()
        assert "node_modules" in config.indexer.exclude_patterns
        assert "__pycache__" in config.indexer.exclude_patterns
        assert ".git" in config.indexer.exclude_patterns
