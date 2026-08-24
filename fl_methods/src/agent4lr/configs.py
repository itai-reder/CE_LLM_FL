"""Loading and resolving Agent4LR configurations.

Configurations are persisted in ``fl_methods/configs/lr_configs.json``
under the modular pipeline's shape::

    {
      "config_name": [
        {"role": "planner",     "provider": "openai", "model": "...", ...},
        {"role": "tool_caller", "provider": "openai", "model": "...",
         "iterations": 10, ...},
        {"role": "finisher",    "provider": "openai", "model": "...", ...}
      ],
      ...
    }

``run_lr.py`` accepts either ``--config <name>`` to load a named config
or per-role CLI flags (mutually exclusive). Per-role flags resolve to
a config named ``"<adhoc>"``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from src.agent4lr.agents import AgentRole, AgentSpec, ProviderName

# Imported lazily inside functions to avoid a circular dependency with agent.py
# (which re-exports LRConfig). The typed name is documented here.
LRConfig = tuple[AgentSpec, ...]

DEFAULT_CONFIGS_PATH: Path  # set below

# fl_methods/src/agent4lr/configs.py → fl_methods/src/agent4lr/ →
# fl_methods/src/ → fl_methods/ → fl_methods/configs/lr_configs.json
_THIS_DIR = Path(__file__).resolve().parent
_FL_METHODS_DIR = _THIS_DIR.parent.parent
DEFAULT_CONFIGS_PATH = _FL_METHODS_DIR / "configs" / "lr_configs.json"

ADHOC_CONFIG_NAME = "<adhoc>"


@dataclass(frozen=True)
class PerRoleOverrides:
    """CLI per-role overrides equivalent to a one-off named config.

    All fields default to ``None``; only those set on the command line
    are populated. ``resolve_config`` raises if any required slot is
    incomplete (e.g. ``--planner-model`` given without
    ``--planner-provider``).
    """

    planner_provider: ProviderName | None = None
    planner_model: str | None = None
    planner_temperature: float | None = None
    planner_top_p: float | None = None
    planner_reasoning_effort: str | None = None

    tool_caller_provider: ProviderName | None = None
    tool_caller_model: str | None = None
    tool_caller_iterations: int | None = None
    tool_caller_temperature: float | None = None
    tool_caller_top_p: float | None = None
    tool_caller_reasoning_effort: str | None = None

    finisher_provider: ProviderName | None = None
    finisher_model: str | None = None
    finisher_temperature: float | None = None
    finisher_top_p: float | None = None
    finisher_reasoning_effort: str | None = None

    def has_any(self) -> bool:
        """True iff at least one field is non-``None``."""
        return any(getattr(self, f.name) is not None for f in fields(self))


_V1_ROLE_ORDER: tuple[AgentRole, ...] = ("planner", "tool_caller", "finisher")
_VALID_PROVIDERS: tuple[ProviderName, ...] = ("ollama", "openai")
_AGENT_FIELDS_BY_NAME = {f.name for f in fields(AgentSpec)}


def _agent_from_dict(d: dict[str, Any]) -> AgentSpec:
    """Construct an ``AgentSpec`` from a config JSON entry.

    Unknown keys are rejected (so typos surface), required keys are
    enforced (``role``, ``provider``, ``model``).
    """
    unknown = set(d) - _AGENT_FIELDS_BY_NAME
    if unknown:
        raise ValueError(f"Unknown AgentSpec field(s): {sorted(unknown)}")
    for required in ("role", "provider", "model"):
        if required not in d:
            raise ValueError(f"AgentSpec entry missing required field {required!r}")
    return AgentSpec(**d)


def load_lr_configs(path: Path = DEFAULT_CONFIGS_PATH) -> dict[str, LRConfig]:
    """Load the named configs JSON into ``{name: LRConfig}``.

    Each entry is validated via :func:`validate_v1_invariant`. The
    function tolerates a missing file by returning an empty dict (so the
    user can launch ad-hoc runs without having to create the file
    first); malformed JSON or a config that fails validation raises.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"lr_configs.json at {path} must be a top-level object")
    out: dict[str, LRConfig] = {}
    for name, entries in raw.items():
        if not isinstance(entries, list):
            raise ValueError(f"config {name!r}: entries must be a list of AgentSpec dicts")
        chain = tuple(_agent_from_dict(e) for e in entries)
        validate_v1_invariant(chain)
        out[name] = chain
    return out


def _agent_from_overrides(overrides: PerRoleOverrides, role: AgentRole) -> AgentSpec:
    """Pull one slot's fields off a :class:`PerRoleOverrides` namespace."""
    prefix = role
    provider = getattr(overrides, f"{prefix}_provider")
    model = getattr(overrides, f"{prefix}_model")
    if provider is None or model is None:
        raise ValueError(
            f"{role}: both --{role.replace('_', '-')}-provider and "
            f"--{role.replace('_', '-')}-model are required for per-role configs."
        )
    kwargs: dict[str, Any] = {"role": role, "provider": provider, "model": model}
    for opt in ("temperature", "top_p", "reasoning_effort"):
        v = getattr(overrides, f"{prefix}_{opt}")
        if v is not None:
            kwargs[opt] = v
    if role == "tool_caller":
        kwargs["iterations"] = overrides.tool_caller_iterations or 10
    return AgentSpec(**kwargs)


def resolve_config(
    *,
    config_name: str | None,
    overrides: PerRoleOverrides | None,
    configs: dict[str, LRConfig] | None = None,
) -> tuple[str, LRConfig]:
    """Return ``(name, config)`` for the requested invocation.

    Exactly one of ``config_name`` and a non-empty ``overrides`` must be
    provided; ``ValueError`` otherwise. If ``config_name`` is given,
    ``configs`` (defaults to :func:`load_lr_configs`) is consulted.
    If ``overrides`` is given, a fresh chain is assembled and ``name``
    is :data:`ADHOC_CONFIG_NAME`. The returned chain always satisfies
    :func:`validate_v1_invariant`.
    """
    has_named = bool(config_name)
    has_overrides = overrides is not None and overrides.has_any()
    if has_named == has_overrides:
        raise ValueError(
            "resolve_config requires exactly one of --config or per-role flags "
            "(not both, not neither)."
        )
    if has_named:
        assert config_name is not None
        cfgs = configs if configs is not None else load_lr_configs()
        if config_name not in cfgs:
            raise ValueError(
                f"Configuration {config_name!r} not found in lr_configs.json. "
                f"Available: {sorted(cfgs)}"
            )
        return config_name, cfgs[config_name]
    assert overrides is not None
    chain = tuple(_agent_from_overrides(overrides, r) for r in _V1_ROLE_ORDER)
    validate_v1_invariant(chain)
    return ADHOC_CONFIG_NAME, chain


def validate_v1_invariant(config: LRConfig) -> None:
    """Enforce the v1 chain shape on a configuration.

    Requirements:

    * exactly three slots
    * roles in order: ``planner``, ``tool_caller``, ``finisher``
    * exactly one ``AgentSpec`` per slot
    * ``iterations`` is set (positive int) on ``tool_caller`` only
    * every slot has a non-empty ``model`` and a recognised ``provider``

    Raises ``ValueError`` with a precise message on any violation.
    """
    if len(config) != len(_V1_ROLE_ORDER):
        raise ValueError(
            f"v1 LRConfig must have exactly {len(_V1_ROLE_ORDER)} slots; got {len(config)}"
        )
    for idx, (spec, expected_role) in enumerate(zip(config, _V1_ROLE_ORDER, strict=True)):
        if not isinstance(spec, AgentSpec):
            raise ValueError(f"slot {idx}: expected AgentSpec, got {type(spec).__name__}")
        if spec.role != expected_role:
            raise ValueError(f"slot {idx}: expected role {expected_role!r}, got {spec.role!r}")
        if not spec.model:
            raise ValueError(f"slot {idx} ({spec.role}): model is required")
        if spec.provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"slot {idx} ({spec.role}): provider {spec.provider!r} not in {_VALID_PROVIDERS}"
            )
        if expected_role == "tool_caller":
            if spec.iterations is None or spec.iterations <= 0:
                raise ValueError(
                    f"tool_caller slot must have iterations > 0; got {spec.iterations!r}"
                )
        elif spec.iterations is not None:
            raise ValueError(
                f"slot {idx} ({spec.role}): iterations must be None for non-tool_caller slots"
            )
