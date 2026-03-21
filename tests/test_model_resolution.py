"""Tests for clawteam.model_resolution — 7-level priority chain."""

import pytest

from clawteam.model_resolution import (
    AUTO_ROLE_MAP,
    DEFAULT_TIERS,
    resolve_model,
)


class TestResolvePriority:
    """Each test verifies one priority level wins over all lower levels."""

    def test_cli_model_wins_over_everything(self):
        result = resolve_model(
            cli_model="gpt-5.4",
            agent_model="opus",
            agent_model_tier="strong",
            template_model_strategy="auto",
            template_model="sonnet-4.6",
            config_default_model="haiku-4.5",
            agent_type="leader",
        )
        assert result == "gpt-5.4"

    def test_agent_model_wins_over_tier(self):
        result = resolve_model(
            cli_model=None,
            agent_model="codex",
            agent_model_tier="cheap",
            template_model_strategy="auto",
            template_model="sonnet-4.6",
            config_default_model="haiku-4.5",
            agent_type="general-purpose",
        )
        assert result == "codex"

    def test_agent_tier_wins_over_strategy(self):
        result = resolve_model(
            cli_model=None,
            agent_model=None,
            agent_model_tier="cheap",
            template_model_strategy="auto",
            template_model="sonnet-4.6",
            config_default_model="",
            agent_type="leader",
        )
        assert result == DEFAULT_TIERS["cheap"]

    def test_auto_strategy_leader_gets_strong(self):
        result = resolve_model(
            cli_model=None,
            agent_model=None,
            agent_model_tier=None,
            template_model_strategy="auto",
            template_model="sonnet-4.6",
            config_default_model="",
            agent_type="lead-reviewer",
        )
        assert result == DEFAULT_TIERS["strong"]

    def test_auto_strategy_worker_gets_balanced(self):
        result = resolve_model(
            cli_model=None,
            agent_model=None,
            agent_model_tier=None,
            template_model_strategy="auto",
            template_model="sonnet-4.6",
            config_default_model="",
            agent_type="data-collector",
        )
        assert result == DEFAULT_TIERS["balanced"]

    def test_template_model_wins_over_config(self):
        result = resolve_model(
            cli_model=None,
            agent_model=None,
            agent_model_tier=None,
            template_model_strategy=None,
            template_model="sonnet-4.6",
            config_default_model="haiku-4.5",
            agent_type="general-purpose",
        )
        assert result == "sonnet-4.6"

    def test_config_default_used_as_fallback(self):
        result = resolve_model(
            cli_model=None,
            agent_model=None,
            agent_model_tier=None,
            template_model_strategy=None,
            template_model=None,
            config_default_model="haiku-4.5",
            agent_type="general-purpose",
        )
        assert result == "haiku-4.5"

    def test_returns_none_when_nothing_set(self):
        result = resolve_model(
            cli_model=None,
            agent_model=None,
            agent_model_tier=None,
            template_model_strategy=None,
            template_model=None,
            config_default_model="",
            agent_type="general-purpose",
        )
        assert result is None


class TestAutoStrategy:
    def test_reviewer_matches_strong(self):
        result = resolve_model(
            cli_model=None, agent_model=None, agent_model_tier=None,
            template_model_strategy="auto", template_model=None,
            config_default_model="", agent_type="security-reviewer",
        )
        assert result == DEFAULT_TIERS["strong"]

    def test_architect_matches_strong(self):
        result = resolve_model(
            cli_model=None, agent_model=None, agent_model_tier=None,
            template_model_strategy="auto", template_model=None,
            config_default_model="", agent_type="system-architect",
        )
        assert result == DEFAULT_TIERS["strong"]

    def test_manager_matches_strong(self):
        result = resolve_model(
            cli_model=None, agent_model=None, agent_model_tier=None,
            template_model_strategy="auto", template_model=None,
            config_default_model="", agent_type="data-manager",
        )
        assert result == DEFAULT_TIERS["strong"]

    def test_none_strategy_falls_through(self):
        result = resolve_model(
            cli_model=None, agent_model=None, agent_model_tier=None,
            template_model_strategy="none", template_model="sonnet-4.6",
            config_default_model="", agent_type="leader",
        )
        assert result == "sonnet-4.6"


class TestTierOverrides:
    def test_custom_tier_mapping(self):
        result = resolve_model(
            cli_model=None, agent_model=None, agent_model_tier="strong",
            template_model_strategy=None, template_model=None,
            config_default_model="", agent_type="general-purpose",
            tier_overrides={"strong": "gpt-5.4"},
        )
        assert result == "gpt-5.4"

    def test_override_merges_with_defaults(self):
        result = resolve_model(
            cli_model=None, agent_model=None, agent_model_tier="balanced",
            template_model_strategy=None, template_model=None,
            config_default_model="", agent_type="general-purpose",
            tier_overrides={"strong": "gpt-5.4"},
        )
        assert result == DEFAULT_TIERS["balanced"]
