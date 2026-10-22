"""Unit tests for Ticket #03: Codebase Pre-commit, Makefile & CI/CD Security Hardening."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestPreCommitConfig:
    """Verify .pre-commit-config.yaml structure, required hooks, and model checksum protection."""

    def test_pre_commit_config_exists_and_valid_yaml(self) -> None:
        config_path = PROJECT_ROOT / ".pre-commit-config.yaml"
        assert config_path.exists(), ".pre-commit-config.yaml must exist at repository root"
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "repos" in data
        assert isinstance(data["repos"], list)

    def test_pre_commit_required_hooks_present(self) -> None:
        config_path = PROJECT_ROOT / ".pre-commit-config.yaml"
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        hook_ids: set[str] = set()
        for repo in data.get("repos", []):
            for hook in repo.get("hooks", []):
                if "id" in hook:
                    hook_ids.add(hook["id"])

        required_hooks = {
            "trailing-whitespace",
            "end-of-file-fixer",
            "ruff",
            "ruff-format",
            "gitleaks",
        }
        missing = required_hooks - hook_ids
        assert not missing, f"Missing required pre-commit hooks: {missing}"

    def test_pre_commit_protects_exported_model_checksums(self) -> None:
        """Ensure trailing-whitespace and end-of-file-fixer exclude models/exported/."""
        config_path = PROJECT_ROOT / ".pre-commit-config.yaml"
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for repo in data.get("repos", []):
            for hook in repo.get("hooks", []):
                if hook.get("id") in ("trailing-whitespace", "end-of-file-fixer"):
                    exclude_pattern = hook.get("exclude", "")
                    assert "models/exported/" in exclude_pattern, (
                        f"Hook {hook.get('id')} must exclude models/exported/ to preserve MD5 checksums"
                    )


class TestMakefileHardening:
    """Verify Makefile standardized targets and macOS OpenMP thread safety."""

    def test_makefile_required_targets_exist(self) -> None:
        makefile_path = PROJECT_ROOT / "Makefile"
        assert makefile_path.exists(), "Makefile must exist at repository root"
        content = makefile_path.read_text(encoding="utf-8")

        required_targets = [
            "install:",
            "dev:",
            "run-api:",
            "run-ui:",
            "lint:",
            "test:",
            "check:",
            "security:",
            "pre-commit:",
        ]
        for target in required_targets:
            assert target in content, f"Makefile is missing target '{target}'"

    def test_makefile_openmp_thread_safety_on_runtime_targets(self) -> None:
        """Verify run-api, run-ui, and dev set OMP_NUM_THREADS=1 to prevent SIGSEGV 139."""
        makefile_path = PROJECT_ROOT / "Makefile"
        content = makefile_path.read_text(encoding="utf-8")
        assert content.count("OMP_NUM_THREADS=1") >= 4, (
            "Makefile must enforce OMP_NUM_THREADS=1 across dev, run-api, and run-ui targets"
        )


class TestPyprojectSecurityConfig:
    """Verify pyproject.toml dev dependencies and Bandit SAST configuration."""

    def test_dev_dependencies_include_security_and_precommit_tools(self) -> None:
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        dev_deps = data.get("dependency-groups", {}).get("dev", [])
        dev_dep_names = [dep.split(">=")[0].split("==")[0].strip() for dep in dev_deps]

        for tool in ("ruff", "bandit", "mypy", "pytest", "pre-commit", "pip-audit"):
            assert tool in dev_dep_names, f"Missing '{tool}' in [dependency-groups].dev"

    def test_bandit_configuration_present(self) -> None:
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        bandit_cfg = data.get("tool", {}).get("bandit", {})
        assert "tests" in bandit_cfg.get("exclude_dirs", [])
