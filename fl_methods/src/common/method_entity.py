"""Canonical method-identity model for the FL pipeline.

The :class:`MethodEntity` is the primary identity used by behavior-affecting
pipeline logic (statement→method aggregation, BoostN scoring, ranking dedup,
FL→Agent4SR merge).

Identity format (corpus convention): ``pkg$Class.method(SimpleParams)`` — the
``$`` separates package from class, parameters are simple type names, and no
start/end line is encoded in the string.  Equality and hashing are on
``corpus_id`` alone, so a ``dict[MethodEntity, float]`` collapses overload
collisions across the same logical method.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from src.common.java_parser import MethodInfo

logger = logging.getLogger(__name__)

StatementKey = tuple[str, int]


@dataclass(frozen=True, eq=False)
class MethodEntity:
    """A method keyed by its corpus identity."""

    corpus_id: str  # pkg$Class.method(SimpleParams)
    class_fqn_dotted: str  # pkg.Class (for (class, line) statement indexing)
    path: str  # source file path, relative to source root when available
    start_line: int
    end_line: int

    def __hash__(self) -> int:
        return hash(self.corpus_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MethodEntity):
            return NotImplemented
        return self.corpus_id == other.corpus_id


def method_info_to_corpus_id(m: MethodInfo) -> str:
    """Convert a :class:`MethodInfo` to the corpus identity string.

    Input ``method_id``: ``pkg.Class.method(P1,P2).start.end``
    Output: ``pkg$Class.method(P1,P2)``

    Strips the ``.startLine.endLine`` suffix and replaces the last dot before
    the class simple name with ``$``.
    """
    sig = m.method_id
    paren_close = sig.rfind(")")
    if paren_close == -1:
        return sig
    sig = sig[: paren_close + 1]

    class_fqn = m.class_fqn
    last_dot = class_fqn.rfind(".")
    if last_dot == -1:
        return sig

    package = class_fqn[:last_dot]
    rest = sig[len(package) + 1 :]
    return f"{package}${rest}"


def method_entity_from_method_info(
    m: MethodInfo,
    *,
    src_root: Path | None = None,
) -> MethodEntity:
    """Build a :class:`MethodEntity` from a parsed :class:`MethodInfo`.

    If *src_root* is provided, ``path`` is stored relative to that root; this
    matches the convention used by ``method_signatures.csv``.  Otherwise the
    raw ``file_path`` is preserved as-is.
    """
    if src_root is not None and m.file_path:
        try:
            rel = Path(m.file_path).resolve().relative_to(Path(src_root).resolve())
            path = str(rel)
        except (ValueError, OSError):
            path = m.file_path
    else:
        path = m.file_path

    return MethodEntity(
        corpus_id=method_info_to_corpus_id(m),
        class_fqn_dotted=m.class_fqn,
        path=path,
        start_line=m.start_line,
        end_line=m.end_line,
    )


def method_entity_from_python_method_info(
    m: MethodInfo,
    module: str,
    *,
    src_root: Path | None = None,
) -> MethodEntity:
    """Build a :class:`MethodEntity` from a Python :class:`MethodInfo`.

    The Java ``method_info_to_corpus_id`` cannot be reused: it places ``$`` at the
    last dot of ``class_fqn``, which mis-splits multi-segment module paths and
    nested classes.  Here the module/owner boundary is known explicitly, so the
    corpus id is built directly as ``<module>$<qualname>(<param_names>)``:

    - module-level function (``owner == module``) → ``<module>$<func>(...)``;
    - class method (``owner == <module>.<DottedClass>``) →
      ``<module>$<DottedClass>.<method>(...)``.

    ``class_fqn_dotted`` is the owner (``m.class_fqn``), which equals the
    statement-ID ``text-before-#`` so ``(class_fqn_dotted, line)`` aggregation
    joins.
    """
    owner = m.class_fqn
    if owner == module:
        qualname = m.method_name
    else:
        # owner == "<module>.<DottedClass>"; strip the "<module>." prefix.
        class_part = owner[len(module) + 1 :]
        qualname = f"{class_part}.{m.method_name}"
    corpus_id = f"{module}${qualname}({','.join(m.param_types)})"

    if src_root is not None and m.file_path:
        try:
            path = str(Path(m.file_path).resolve().relative_to(Path(src_root).resolve()))
        except (ValueError, OSError):
            path = m.file_path
    else:
        path = m.file_path

    return MethodEntity(
        corpus_id=corpus_id,
        class_fqn_dotted=owner,
        path=path,
        start_line=m.start_line,
        end_line=m.end_line,
    )


def write_method_entities_csv(output_path: Path, entities: list[MethodEntity]) -> None:
    """Write entities to a ``;``-delimited CSV.

    Header: ``corpus_id;path;startLine;endLine`` — same shape as
    ``method_signatures.csv`` but keyed on corpus identity.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("corpus_id;path;startLine;endLine\n")
        for e in entities:
            fh.write(f"{e.corpus_id};{e.path};{e.start_line};{e.end_line}\n")


def load_method_entities(processed_dir: Path) -> list[MethodEntity]:
    """Load ``method_signatures.csv`` from *processed_dir*.

    Recovers ``class_fqn_dotted`` from the ``corpus_id`` (everything before
    ``.method(...)``, with ``$`` replaced by ``.``).
    """
    csv_path = processed_dir / "method_signatures.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"method_signatures.csv not found: {csv_path}")

    results: list[MethodEntity] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for row in reader:
            corpus_id = row.get("corpus_id", "").strip()
            path = row.get("path", "").strip()
            start_raw = row.get("startLine", "").strip()
            end_raw = row.get("endLine", "").strip()

            if not corpus_id or not start_raw.isdigit() or not end_raw.isdigit():
                continue

            class_fqn_dotted = _class_fqn_from_corpus_id(corpus_id)
            results.append(
                MethodEntity(
                    corpus_id=corpus_id,
                    class_fqn_dotted=class_fqn_dotted,
                    path=path,
                    start_line=int(start_raw),
                    end_line=int(end_raw),
                )
            )

    logger.debug("Loaded %d method entities from %s", len(results), csv_path)
    return results


def build_entity_line_index(
    entities: list[MethodEntity],
) -> dict[StatementKey, list[MethodEntity]]:
    """Build ``(dotted_class_fqn, line) -> [MethodEntity]`` index.

    Statement IDs (``pkg.Class#42``) use the dotted class form, so the index
    keys mirror that for direct ``(class, line)`` lookup.
    """
    index: dict[StatementKey, list[MethodEntity]] = {}
    for e in entities:
        for line in range(e.start_line, e.end_line + 1):
            key = (e.class_fqn_dotted, line)
            index.setdefault(key, []).append(e)
    return index


def aggregate_statement_scores_to_entities(
    statement_scores: dict[StatementKey, float],
    index: dict[StatementKey, list[MethodEntity]],
) -> dict[MethodEntity, float]:
    """Aggregate statement scores onto containing :class:`MethodEntity` via max.

    Overload collisions in the corpus identity collapse via the
    ``MethodEntity`` hash, taking the max score across colliding line ranges.
    """
    method_scores: dict[MethodEntity, float] = {}
    for stmt_key, score in statement_scores.items():
        entities = index.get(stmt_key)
        if not entities:
            continue
        for e in entities:
            prev = method_scores.get(e)
            method_scores[e] = score if prev is None else max(prev, score)
    return method_scores


def rank_entities(
    method_scores: dict[MethodEntity, float],
) -> list[tuple[MethodEntity, float, int]]:
    """Sort entities by score descending and assign tied ranks.

    Ties share a rank; the next rank skips accordingly (1, 1, 3, 4).
    """
    sorted_entities = sorted(
        method_scores.items(),
        key=lambda item: (-item[1], item[0].corpus_id, item[0].path),
    )

    ranked: list[tuple[MethodEntity, float, int]] = []
    prev_score: float | None = None
    prev_rank = 0

    for i, (entity, score) in enumerate(sorted_entities, start=1):
        if score != prev_score:
            prev_rank = i
            prev_score = score
        ranked.append((entity, score, prev_rank))

    return ranked


def write_entity_ranking_csv(
    output_path: Path,
    ranked: list[tuple[MethodEntity, float, int]],
) -> None:
    """Write a ``;``-delimited ranking CSV keyed on corpus identity.

    Format: ``rank;signature;path;startLine;endLine;score`` — the ``signature``
    column carries the corpus identity.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("rank;signature;path;startLine;endLine;score\n")
        for entity, score, rank in ranked:
            fh.write(
                f"{rank};{entity.corpus_id};{entity.path}"
                f";{entity.start_line};{entity.end_line};{score}\n"
            )


def _class_fqn_from_corpus_id(corpus_id: str) -> str:
    """Recover the statement-owner FQN from a corpus identity string.

    The ``$`` separates the package/module dotted path from the in-package
    qualified name; the recovered owner is what the statement-ID ``text-before-#``
    must equal for ``(class_fqn_dotted, line)`` aggregation to join (see
    :func:`build_entity_line_index`).

    Handles both benchmark shapes:

    - **Java** (``pkg$Class.method(P)``, ``pkg.Outer$Inner.method(P)``) — the
      qualname after ``$`` always carries a ``.method``; the owner is the part
      before that last dot, with any inner-class ``$`` flattened to ``.``.
      ``pkg$Class.method(P)`` → ``pkg.Class``; ``pkg.Outer$Inner.m()`` →
      ``pkg.Outer.Inner``.
    - **Python** — class methods mirror Java
      (``mod$Class.method(self)`` → ``mod.Class``); a module-level function has
      no class in its qualname (``mod$func(x)``), so the **module is the owner**
      (→ ``mod``).  The class-bearing Java path would amputate a module segment
      here, which is why Python needs this dedicated owner-recovery rule.

    Falls back to the substring before the last ``.`` when no ``$`` is present
    (Java top-level class with the default package).
    """
    paren = corpus_id.find("(")
    head = corpus_id[:paren] if paren != -1 else corpus_id
    dollar = head.find("$")  # first '$' splits package/module from qualname
    if dollar == -1:  # Java default-package legacy: no '$' separator
        last_dot = head.rfind(".")
        return head[:last_dot] if last_dot != -1 else head
    prefix = head[:dollar]  # dotted package / module path (never contains '$')
    qualtail = head[dollar + 1 :]  # Class[.Inner].method  OR  bare function name
    last_dot = qualtail.rfind(".")
    if last_dot == -1:  # Python module-level function: the module owns the line
        return prefix
    class_portion = qualtail[:last_dot].replace("$", ".")  # Java inner '$' -> '.'
    return f"{prefix}.{class_portion}"
