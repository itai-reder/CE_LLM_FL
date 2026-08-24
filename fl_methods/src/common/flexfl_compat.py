"""Helpers for exporting CEFL outputs to FlexFL-compatible method CSVs.

This module handles two compatibility paths:

1. Statement-level rankings (Ochiai/SBIR) -> method-level rows by mapping
   statement lines to extracted Java methods and aggregating with max score.
2. BoostN method IDs -> FlexFL CSV schema (File,Signature,StartLine,EndLine).
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.common.java_parser import MethodInfo, extract_methods_from_java, find_java_files
from src.common.method_entity import load_method_entities

LOGGER = logging.getLogger(__name__)

StatementKey = tuple[str, int]  # (class_fqn, line_number)


@dataclass(frozen=True)
class FlexMethodRow:
    """One method entry in FlexFL CSV schema."""

    file: str
    signature: str
    start_line: int
    end_line: int

    @property
    def method_key(self) -> str:
        """Canonical method key used in comparisons."""
        return f"{self.file}.{self.signature}"


def parse_method_id_to_flex_row(method_id: str) -> FlexMethodRow | None:
    """Parse ``pkg.Class.method(P1,P2).start.end`` into a FlexMethodRow.

    Returns ``None`` when the input cannot be parsed safely.
    """
    close_paren = method_id.rfind(")")
    if close_paren == -1:
        return None

    suffix = method_id[close_paren + 1 :]
    if not suffix.startswith("."):
        return None

    suffix_parts = suffix[1:].split(".")
    if len(suffix_parts) != 2:
        return None
    if not suffix_parts[0].isdigit() or not suffix_parts[1].isdigit():
        return None

    start_line = int(suffix_parts[0])
    end_line = int(suffix_parts[1])
    if start_line <= 0 or end_line < start_line:
        return None

    full_signature = method_id[: close_paren + 1]
    open_paren = full_signature.rfind("(")
    if open_paren == -1:
        return None

    pre_args = full_signature[:open_paren]
    method_dot = pre_args.rfind(".")
    if method_dot == -1:
        return None

    file_part = pre_args[:method_dot]
    method_name = pre_args[method_dot + 1 :]
    params = full_signature[open_paren + 1 : close_paren]
    signature = f"{method_name}({params})"

    if not file_part or not method_name:
        return None

    return FlexMethodRow(
        file=file_part,
        signature=signature,
        start_line=start_line,
        end_line=end_line,
    )


def read_statement_scores_csv(path: Path) -> dict[StatementKey, float]:
    """Read ``Statement,Suspiciousness`` CSV into ``(class,line) -> score``."""
    if not path.exists():
        raise FileNotFoundError(f"Statement ranking CSV not found: {path}")

    scores: dict[StatementKey, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return scores

        id_column = "Statement" if "Statement" in reader.fieldnames else reader.fieldnames[0]
        score_column = (
            "Suspiciousness"
            if "Suspiciousness" in reader.fieldnames
            else (reader.fieldnames[1] if len(reader.fieldnames) > 1 else "")
        )

        for row in reader:
            stmt_id = row.get(id_column)
            raw_score = row.get(score_column)
            if not isinstance(stmt_id, str) or not isinstance(raw_score, str):
                continue
            stmt_id = stmt_id.strip()
            raw_score = raw_score.strip()
            if not stmt_id or not raw_score or "#" not in stmt_id:
                continue

            class_fqn, line_str = stmt_id.rsplit("#", 1)
            if not line_str.isdigit():
                continue

            try:
                score = float(raw_score)
            except ValueError:
                continue

            key = (class_fqn, int(line_str))
            prev = scores.get(key)
            scores[key] = score if prev is None else max(prev, score)

    return scores


def read_statement_scores_csv_ordered(path: Path) -> list[tuple[StatementKey, float]]:
    """Read statement CSV preserving ranking order with stable tie handling.

    Returns a deduplicated ordered list of ``((class_fqn, line), score)`` sorted by:
    1) score descending,
    2) first observed position for the statement when that best score appeared.
    """
    if not path.exists():
        raise FileNotFoundError(f"Statement ranking CSV not found: {path}")

    # key -> (best_score, first_pos_for_best_score)
    best_by_stmt: dict[StatementKey, tuple[float, int]] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []

        id_column = "Statement" if "Statement" in reader.fieldnames else reader.fieldnames[0]
        score_column = (
            "Suspiciousness"
            if "Suspiciousness" in reader.fieldnames
            else (reader.fieldnames[1] if len(reader.fieldnames) > 1 else "")
        )

        for pos, row in enumerate(reader, start=1):
            stmt_id = row.get(id_column)
            raw_score = row.get(score_column)
            if not isinstance(stmt_id, str) or not isinstance(raw_score, str):
                continue
            stmt_id = stmt_id.strip()
            raw_score = raw_score.strip()
            if not stmt_id or not raw_score or "#" not in stmt_id:
                continue

            class_fqn, line_str = stmt_id.rsplit("#", 1)
            if not line_str.isdigit():
                continue

            try:
                score = float(raw_score)
            except ValueError:
                continue

            key = (class_fqn, int(line_str))
            prev = best_by_stmt.get(key)
            if prev is None or score > prev[0] or (score == prev[0] and pos < prev[1]):
                best_by_stmt[key] = (score, pos)

    ordered = sorted(
        best_by_stmt.items(),
        key=lambda item: (
            -item[1][0],
            item[1][1],
            item[0][0],
            item[0][1],
        ),
    )
    return [(stmt_key, score_pos[0]) for stmt_key, score_pos in ordered]


def read_method_scores_csv(path: Path) -> dict[str, float]:
    """Read ``Signature,Suspiciousness`` CSV into ``method_id -> score``."""
    if not path.exists():
        raise FileNotFoundError(f"Method ranking CSV not found: {path}")

    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return scores

        id_column = "Signature" if "Signature" in reader.fieldnames else reader.fieldnames[0]
        score_column = (
            "Suspiciousness"
            if "Suspiciousness" in reader.fieldnames
            else (reader.fieldnames[1] if len(reader.fieldnames) > 1 else "")
        )

        for row in reader:
            method_id = row.get(id_column)
            raw_score = row.get(score_column)
            if not isinstance(method_id, str) or not isinstance(raw_score, str):
                continue
            method_id = method_id.strip()
            raw_score = raw_score.strip()
            if not method_id or not raw_score:
                continue

            try:
                score = float(raw_score)
            except ValueError:
                continue

            prev = scores.get(method_id)
            scores[method_id] = score if prev is None else max(prev, score)

    return scores


def _method_info_to_flex_row(method: MethodInfo) -> FlexMethodRow:
    """Convert MethodInfo into FlexFL row shape."""
    signature = f"{method.method_name}({','.join(method.param_types)})"
    return FlexMethodRow(
        file=method.class_fqn,
        signature=signature,
        start_line=method.start_line,
        end_line=method.end_line,
    )


def _index_rows_by_line(
    method_rows: Iterable[FlexMethodRow],
) -> dict[StatementKey, list[FlexMethodRow]]:
    """Build ``(class,line) -> [methods]`` map from method rows."""
    line_to_methods: dict[StatementKey, list[FlexMethodRow]] = {}

    for row in method_rows:
        if row.start_line <= 0 or row.end_line < row.start_line:
            continue
        for line in range(row.start_line, row.end_line + 1):
            key = (row.file, line)
            line_to_methods.setdefault(key, []).append(row)

    for key, rows in line_to_methods.items():
        unique_rows = sorted(
            set(rows),
            key=lambda item: (
                item.end_line - item.start_line,
                item.start_line,
                item.end_line,
                item.file,
                item.signature,
            ),
        )
        line_to_methods[key] = unique_rows

    return line_to_methods


def build_line_to_method_rows(src_dir: Path) -> dict[StatementKey, list[FlexMethodRow]]:
    """Build ``(class,line) -> [methods]`` map from Java source files."""
    java_files = find_java_files(src_dir, exclude_tests=True)
    method_rows: list[FlexMethodRow] = []

    for java_file in java_files:
        for method in extract_methods_from_java(java_file):
            if method.start_line <= 0 or method.end_line < method.start_line:
                continue
            method_rows.append(_method_info_to_flex_row(method))

    line_to_methods = _index_rows_by_line(method_rows)

    LOGGER.info(
        "Built line-to-method index for %s (%d classes/lines)",
        src_dir,
        len(line_to_methods),
    )
    return line_to_methods


def build_line_to_method_rows_from_method_ids(
    method_ids: Iterable[str],
) -> dict[StatementKey, list[FlexMethodRow]]:
    """Build line-to-method map from method IDs without source checkout access."""
    rows: list[FlexMethodRow] = []
    for method_id in method_ids:
        row = parse_method_id_to_flex_row(method_id)
        if row is not None:
            rows.append(row)
    return _index_rows_by_line(rows)


def build_line_to_method_rows_from_spectra(
    spectra_path: Path,
) -> dict[StatementKey, list[FlexMethodRow]]:
    """Build line-to-method map from GZoltar spectra.csv file.

    The spectra.csv has format:
        name
        pkg.Class#method(params):lineNum
        ...

    We extract the class, method name (signature), and line number to build
    method rows, then index by (class, line) for statement->method lookup.

    Note: Parameter types are simplified to simple class names (no package prefix)
    to match FlexFL reference format.
    """
    if not spectra_path.exists():
        raise FileNotFoundError(f"Spectra file not found: {spectra_path}")

    # (class_fqn, signature) -> [min_line, max_line]
    method_ranges: dict[tuple[str, str], list[int]] = {}

    with spectra_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or header[0] != "name":
            LOGGER.warning("Unexpected spectra.csv header: %s", header)
            return {}

        for row in reader:
            if not row:
                continue
            raw_name = row[0].strip()
            if not raw_name or "#" not in raw_name or ":" not in raw_name:
                continue

            class_part, rest = raw_name.split("#", 1)
            method_and_line = rest.rsplit(":", 1)
            if len(method_and_line) != 2:
                continue

            method_sig, line_str = method_and_line
            if not line_str.isdigit():
                continue
            line_num = int(line_str)

            class_fqn = class_part.replace("$", ".")

            paren_open = method_sig.find("(")
            if paren_open == -1:
                method_name = method_sig
                signature = f"{method_name}()"
            else:
                method_name = method_sig[:paren_open]
                raw_params = method_sig[paren_open + 1 :].rstrip(")")
                if raw_params:
                    simple_params = ",".join(p.rsplit(".", 1)[-1] for p in raw_params.split(","))
                    signature = f"{method_name}({simple_params})"
                else:
                    signature = f"{method_name}()"

            key = (class_fqn, signature)
            current = method_ranges.get(key)
            if current is None:
                method_ranges[key] = [line_num, line_num]
            else:
                if line_num < current[0]:
                    current[0] = line_num
                if line_num > current[1]:
                    current[1] = line_num

    method_rows = [
        FlexMethodRow(
            file=class_fqn,
            signature=signature,
            start_line=line_range[0],
            end_line=line_range[1],
        )
        for (class_fqn, signature), line_range in method_ranges.items()
    ]

    LOGGER.info("Built method index from spectra: %d methods", len(method_rows))
    return _index_rows_by_line(method_rows)


def build_line_to_method_rows_from_flex_csv(
    flex_csv_path: Path,
) -> dict[StatementKey, list[FlexMethodRow]]:
    """Build line-to-method map from a FlexFL-style method CSV."""
    if not flex_csv_path.exists():
        raise FileNotFoundError(f"Flex method CSV not found: {flex_csv_path}")

    method_rows: list[FlexMethodRow] = []
    seen: set[FlexMethodRow] = set()

    with flex_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_part = (row.get("File") or "").strip()
            signature = (row.get("Signature") or "").strip()
            start_raw = (row.get("StartLine") or "").strip()
            end_raw = (row.get("EndLine") or "").strip()
            if not file_part or not signature or not start_raw.isdigit() or not end_raw.isdigit():
                continue

            start_line = int(start_raw)
            end_line = int(end_raw)
            if start_line <= 0 or end_line < start_line:
                continue

            row_obj = FlexMethodRow(
                file=file_part,
                signature=signature,
                start_line=start_line,
                end_line=end_line,
            )
            if row_obj in seen:
                continue
            seen.add(row_obj)
            method_rows.append(row_obj)

    LOGGER.info("Built method index from Flex CSV: %d methods", len(method_rows))
    return _index_rows_by_line(method_rows)


def aggregate_statement_scores_to_methods(
    statement_scores: dict[StatementKey, float],
    line_to_methods: dict[StatementKey, list[FlexMethodRow]],
) -> dict[FlexMethodRow, float]:
    """Aggregate statement scores to methods using max score per mapped method."""
    method_scores: dict[FlexMethodRow, float] = {}

    for key, score in statement_scores.items():
        methods = line_to_methods.get(key)
        if not methods:
            continue

        for method in methods:
            prev = method_scores.get(method)
            method_scores[method] = score if prev is None else max(prev, score)

    return method_scores


def rank_method_scores(
    method_scores: dict[FlexMethodRow, float],
) -> list[tuple[FlexMethodRow, float]]:
    """Return deterministic ranking of method rows by score desc."""
    return sorted(
        method_scores.items(),
        key=lambda item: (
            -item[1],
            item[0].file,
            item[0].signature,
            item[0].start_line,
            item[0].end_line,
        ),
    )


def convert_statement_scores_with_map(
    statement_scores: dict[StatementKey, float],
    line_to_methods: dict[StatementKey, list[FlexMethodRow]],
) -> list[tuple[FlexMethodRow, float]]:
    """Convert statement scores to ranked FlexFL method rows."""
    method_scores = aggregate_statement_scores_to_methods(statement_scores, line_to_methods)
    return rank_method_scores(method_scores)


def _is_filtered_synthetic_signature(signature: str) -> bool:
    return signature.startswith("<clinit>(") or signature == "<clinit>()"


def convert_statement_scores_with_map_parity(
    statement_scores_ordered: list[tuple[StatementKey, float]],
    line_to_methods: dict[StatementKey, list[FlexMethodRow]],
    *,
    filter_synthetic: bool,
) -> list[tuple[FlexMethodRow, float]]:
    """Convert statement scores with FlexFL-parity-oriented ordering.

    Ordering policy:
    1) max suspiciousness per method,
    2) earlier statement position (where that max first appears),
    3) stable lexical fallback.
    """
    # method -> (best_score, first_position_for_best_score)
    best_method_scores: dict[FlexMethodRow, tuple[float, int]] = {}

    for position, (stmt_key, score) in enumerate(statement_scores_ordered, start=1):
        methods = line_to_methods.get(stmt_key)
        if not methods:
            continue

        for method in methods:
            if filter_synthetic and _is_filtered_synthetic_signature(method.signature):
                continue
            prev = best_method_scores.get(method)
            if prev is None or score > prev[0] or (score == prev[0] and position < prev[1]):
                best_method_scores[method] = (score, position)

    ranked = sorted(
        best_method_scores.items(),
        key=lambda item: (
            -item[1][0],
            item[1][1],
            item[0].file,
            item[0].signature,
            item[0].start_line,
            item[0].end_line,
        ),
    )
    return [(method, score_pos[0]) for method, score_pos in ranked]


def convert_boostn_scores(method_scores: dict[str, float]) -> list[tuple[FlexMethodRow, float]]:
    """Convert BoostN method-ID scores to ranked FlexFL method rows.

    Accepts the legacy ``pkg.Class.method(P).start.end`` format. For the
    corpus-identity format BoostN now emits, use
    :func:`convert_boostn_scores_corpus_id` instead.
    """
    parsed_scores: dict[FlexMethodRow, float] = {}

    for method_id, score in method_scores.items():
        row = parse_method_id_to_flex_row(method_id)
        if row is None:
            LOGGER.warning("Skipping unparseable BoostN method ID: %s", method_id)
            continue
        prev = parsed_scores.get(row)
        parsed_scores[row] = score if prev is None else max(prev, score)

    return rank_method_scores(parsed_scores)


def _signature_from_corpus_id(corpus_id: str) -> str | None:
    """Extract ``method(P1,P2)`` from a corpus identity ``pkg$Class.method(P)``."""
    paren = corpus_id.find("(")
    if paren == -1:
        return None
    pre_paren = corpus_id[:paren]
    last_dot = pre_paren.rfind(".")
    if last_dot == -1:
        return None
    method_name = pre_paren[last_dot + 1 :]
    args = corpus_id[paren:]
    if not method_name:
        return None
    return f"{method_name}{args}"


def convert_boostn_scores_corpus_id(
    method_scores: dict[str, float],
    processed_dir: Path,
) -> list[tuple[FlexMethodRow, float]]:
    """Convert corpus-identity BoostN scores to ranked FlexFL method rows.

    Joins on ``corpus_id`` against ``method_signatures.csv`` to recover
    path/start/end metadata.  Mirrors :func:`convert_boostn_scores` for the
    corpus-identity output format BoostN emits post-refactor.
    """
    entities = load_method_entities(processed_dir)
    entity_by_corpus_id = {e.corpus_id: e for e in entities}

    parsed_scores: dict[FlexMethodRow, float] = {}
    missing = 0
    for corpus_id, score in method_scores.items():
        entity = entity_by_corpus_id.get(corpus_id)
        if entity is None:
            missing += 1
            continue
        signature = _signature_from_corpus_id(corpus_id)
        if signature is None:
            LOGGER.warning("Skipping unparseable BoostN corpus_id: %s", corpus_id)
            continue
        row = FlexMethodRow(
            file=entity.class_fqn_dotted,
            signature=signature,
            start_line=entity.start_line,
            end_line=entity.end_line,
        )
        prev = parsed_scores.get(row)
        parsed_scores[row] = score if prev is None else max(prev, score)

    if missing:
        LOGGER.warning(
            "BoostN: %d/%d corpus_ids not present in method_signatures.csv",
            missing,
            len(method_scores),
        )

    return rank_method_scores(parsed_scores)


def write_flex_method_csv(
    output_path: Path,
    ranked_rows: list[tuple[FlexMethodRow, float]],
    *,
    include_suspiciousness: bool,
) -> None:
    """Write ranked methods in FlexFL CSV schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if include_suspiciousness:
            writer.writerow(["File", "Signature", "StartLine", "EndLine", "Suspiciousness"])
            for row, score in ranked_rows:
                writer.writerow([row.file, row.signature, row.start_line, row.end_line, score])
            return

        writer.writerow(["File", "Signature", "StartLine", "EndLine"])
        for row, _ in ranked_rows:
            writer.writerow([row.file, row.signature, row.start_line, row.end_line])
