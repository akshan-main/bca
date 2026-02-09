"""Tests for the CLI interface."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from cegraph.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def indexed_project(tmp_project: Path) -> Path:
    """Create a tmp_project that has been indexed."""
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--path", str(tmp_project)])
    assert result.exit_code == 0, f"Init failed: {result.output}"
    return tmp_project


class TestCLIInit:
    def test_init_basic(self, runner: CliRunner, tmp_project: Path):
        result = runner.invoke(main, ["init", "--path", str(tmp_project)])
        assert result.exit_code == 0
        assert "Initializing" in result.output or "Indexed" in result.output

    def test_init_creates_cegraph_dir(self, runner: CliRunner, tmp_project: Path):
        runner.invoke(main, ["init", "--path", str(tmp_project)])
        assert (tmp_project / ".cegraph").exists()
        assert (tmp_project / ".cegraph" / "config.json").exists()
        assert (tmp_project / ".cegraph" / "graph.db").exists()

    def test_init_nonexistent_path(self, runner: CliRunner):
        result = runner.invoke(main, ["init", "--path", "/nonexistent/path"])
        assert result.exit_code != 0


class TestCLIStatus:
    def test_status(self, runner: CliRunner, indexed_project: Path):
        result = runner.invoke(main, ["status", "--path", str(indexed_project)])
        assert result.exit_code == 0

    def test_status_no_index(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(main, ["status", "--path", str(tmp_path)])
        assert result.exit_code != 0


class TestCLISearch:
    def test_search_symbol(self, runner: CliRunner, indexed_project: Path):
        result = runner.invoke(
            main, ["search", "User", "--path", str(indexed_project)]
        )
        assert result.exit_code == 0
        assert "User" in result.output

    def test_search_no_results(self, runner: CliRunner, indexed_project: Path):
        result = runner.invoke(
            main, ["search", "xyznonexistent", "--path", str(indexed_project)]
        )
        assert result.exit_code == 0
        assert "No" in result.output


class TestCLIWhoCalls:
    def test_who_calls(self, runner: CliRunner, indexed_project: Path):
        result = runner.invoke(
            main, ["who-calls", "helper_function", "--path", str(indexed_project)]
        )
        assert result.exit_code == 0

    def test_who_calls_not_found(self, runner: CliRunner, indexed_project: Path):
        result = runner.invoke(
            main, ["who-calls", "nonexistent_func", "--path", str(indexed_project)]
        )
        assert result.exit_code == 0
        assert "No callers" in result.output


class TestCLIImpact:
    def test_impact(self, runner: CliRunner, indexed_project: Path):
        result = runner.invoke(
            main, ["impact", "calculate_total", "--path", str(indexed_project)]
        )
        assert result.exit_code == 0


class TestCLIConfig:
    def test_config_show(self, runner: CliRunner, indexed_project: Path):
        result = runner.invoke(
            main, ["config", "show", "--path", str(indexed_project)]
        )
        assert result.exit_code == 0
        assert "llm" in result.output

    def test_config_get(self, runner: CliRunner, indexed_project: Path):
        result = runner.invoke(
            main, ["config", "get", "llm.provider", "--path", str(indexed_project)]
        )
        assert result.exit_code == 0

    def test_config_set(self, runner: CliRunner, indexed_project: Path):
        result = runner.invoke(
            main,
            ["config", "set", "llm.provider", "openai", "--path", str(indexed_project)],
        )
        assert result.exit_code == 0
        assert "Set" in result.output


class TestCLIVersion:
    def test_version(self, runner: CliRunner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
