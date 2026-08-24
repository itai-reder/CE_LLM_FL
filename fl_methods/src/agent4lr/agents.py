"""Per-slot agent specification and identity helpers.

An ``AgentSpec`` captures the *what* of one slot in an LR configuration —
which provider/model/parameters drive it — without any execution state.
Its canonical-JSON identity feeds the content-addressable checkpoint
store: two slots with identical identities produce reusable checkpoints.

v1 supports three role-bound slots in the canonical order
``planner -> tool_caller -> finisher``. The data model is intentionally
variable-length (``LRConfig = tuple[AgentSpec, ...]``); the runner
validates the v1 invariant at entry, not the type system.

The chain inputs descriptor (``chain_inputs_descriptor``) wraps the
per-bug inputs that *all* checkpoint keys in the chain depend on
(project, bug_id, dataset, sr_model_id, candidate_source,
sorted(input_keys)).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

AgentRole = Literal["planner", "tool_caller", "finisher"]
ProviderName = Literal["ollama", "openai"]


@dataclass(frozen=True)
class AgentSpec:
    """Identity of one slot in an LRConfig chain.

    ``iterations`` is meaningful only for the ``tool_caller`` slot — the
    upper bound on inner ReAct iterations. Other slots leave it as
    ``None``.

    ``reasoning_effort`` is consumed by the OpenAI Responses API for the
    o-series / gpt-5 families; ignored by Ollama and by non-reasoning
    OpenAI models.

    The canonical-JSON form of this dataclass (sorted keys, no
    whitespace) is the per-slot identity used for chain hashing.
    """

    role: AgentRole
    provider: ProviderName
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    reasoning_effort: str | None = None
    iterations: int | None = None
    provider_opts: dict[str, Any] | None = None
    """Provider-specific knobs forwarded as ``**opts`` to the backend.

    Used today for Ollama's ``think`` payload key (e.g.
    ``provider_opts={"think": True}``). Keys not recognised by the
    receiving provider are silently dropped. Included in the canonical
    identity hash so two chains differing only in these knobs do not
    share checkpoints.
    """

    def identity(self) -> dict[str, Any]:
        """Return the canonical-JSON-serializable identity of this slot.

        Equivalent to ``asdict(self)``; documented as the contract for
        downstream hashing so future extensions of this dataclass are
        intentional choices about checkpoint reuse.
        """
        return asdict(self)

    def identity_canonical_json(self) -> str:
        """Return the stable JSON serialisation used as a hash input.

        Sorted keys, no whitespace, no trailing newline. Two
        ``AgentSpec`` instances with structurally equal fields produce
        byte-identical output.
        """
        return json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))


def chain_inputs_descriptor(
    *,
    project: str,
    bug_id: str | int,
    dataset: str,
    sr_model_id: str,
    candidate_source: str,
    input_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Return the per-bug inputs descriptor that anchors a checkpoint chain.

    All checkpoints written for one (bug, sr_model_id, input_keys) combo
    share this descriptor; the chain hash composes it with the slot
    sequence. ``candidate_source`` is a relative POSIX path under the
    processed dir (e.g. ``"FlexFL/SR/rankings/top20/llama3.1_8b.txt"``).

    ``input_keys`` is sorted into the descriptor so the order of
    ``--input`` arguments doesn't affect checkpoint identity.
    """
    return {
        "project": project,
        "bug_id": str(bug_id),
        "dataset": dataset,
        "sr_model_id": sr_model_id,
        "candidate_source": candidate_source,
        "input_keys": sorted(input_keys),
    }
