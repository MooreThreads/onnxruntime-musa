#!/usr/bin/env python3
"""Compare PR base and head model-zoo performance reports.

Single-stream latency and multi-stream QPS are both compared with the same
percentage threshold. A candidate fails when it is slower than the base by
more than the threshold.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportRow:
    model: str
    status: str
    value: float | None


@dataclass(frozen=True)
class Comparison:
    model: str
    base_status: str
    head_status: str
    base_value: float | None
    head_value: float | None
    change_pct: float | None
    result: str

    @property
    def failed(self) -> bool:
        return self.result in (
            "REGRESSION",
            "CANDIDATE_SKIPPED",
            "STATUS_FAILED",
            "METRIC_MISSING",
        )


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_metric(value: str) -> float | None:
    lowered = value.lower()
    if any(word in lowered for word in ("unknown", "skipped", "failed", "n/a")):
        return None
    match = _NUMBER_RE.search(value)
    if match is None:
        return None
    parsed = float(match.group())
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def parse_report(path: Path, metric_column: str) -> dict[str, ReportRow]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = -1
    model_index = status_index = metric_index = -1

    for index, line in enumerate(lines):
        if not line.strip().startswith("|"):
            continue
        cells = _split_markdown_row(line)
        if "Model" not in cells or "Status" not in cells or metric_column not in cells:
            continue
        header_index = index
        model_index = cells.index("Model")
        status_index = cells.index("Status")
        metric_index = cells.index(metric_column)
        break

    if header_index < 0:
        raise ValueError(f"{path}: table with {metric_column!r} column not found")

    rows: dict[str, ReportRow] = {}
    for line in lines[header_index + 1 :]:
        if not line.strip().startswith("|"):
            break
        cells = _split_markdown_row(line)
        if not cells or max(model_index, status_index, metric_index) >= len(cells):
            continue
        if set(cells[0]) <= set("-: "):
            continue
        model = cells[model_index]
        if not model:
            continue
        if model in rows:
            raise ValueError(f"{path}: duplicate model row: {model}")
        rows[model] = ReportRow(
            model=model,
            status=cells[status_index],
            value=_parse_metric(cells[metric_index]),
        )

    if not rows:
        raise ValueError(f"{path}: no model rows found")
    return rows


def compare_reports(
    base: dict[str, ReportRow],
    head: dict[str, ReportRow],
    mode: str,
    threshold_pct: float,
) -> list[Comparison]:
    missing_from_head = sorted(set(base) - set(head))
    missing_from_base = sorted(set(head) - set(base))
    if missing_from_head or missing_from_base:
        details: list[str] = []
        if missing_from_head:
            details.append(f"missing from head: {', '.join(missing_from_head)}")
        if missing_from_base:
            details.append(f"missing from base: {', '.join(missing_from_base)}")
        raise ValueError("model sets differ; " + "; ".join(details))

    comparisons: list[Comparison] = []
    for model in base:
        base_row = base[model]
        head_row = head[model]
        result = "PASS"
        change_pct: float | None = None

        if head_row.status not in ("PASS", "SKIPPED"):
            result = "STATUS_FAILED"
        elif base_row.status == "SKIPPED" and head_row.status == "SKIPPED":
            result = "NOT_MEASURED"
        elif head_row.status == "SKIPPED":
            result = "CANDIDATE_SKIPPED"
        elif base_row.status == "SKIPPED":
            result = "BASE_SKIPPED"
        elif base_row.status != "PASS":
            result = "STATUS_FAILED"
        elif base_row.value is None or head_row.value is None:
            result = "METRIC_MISSING"
        else:
            change_pct = (head_row.value / base_row.value - 1.0) * 100.0
            slower_pct = change_pct if mode == "single" else -change_pct
            if slower_pct > threshold_pct + 1e-9:
                result = "REGRESSION"

        comparisons.append(
            Comparison(
                model=model,
                base_status=base_row.status,
                head_status=head_row.status,
                base_value=base_row.value,
                head_value=head_row.value,
                change_pct=change_pct,
                result=result,
            )
        )
    return comparisons


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _format_change(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def _status_label(row: Comparison, mode: str) -> str:
    if row.result == "PASS":
        return "PASS"
    if row.result == "REGRESSION":
        metric = "latency" if mode == "single" else "QPS"
        return f"REGRESSION ({metric} {_format_change(row.change_pct)})"
    return row.result


def _table_for_mode(
    title: str,
    comparisons: list[Comparison],
    mode: str,
    base_metric: str,
    head_metric: str,
    change_metric: str,
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"| Model | Base {base_metric} | Head {head_metric} | {change_metric} | Result |",
        "|---|---|---|---|---|",
    ]
    for row in comparisons:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.model.replace("|", "\\|"),
                    _format_number(row.base_value),
                    _format_number(row.head_value),
                    _format_change(row.change_pct),
                    _status_label(row, mode),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def build_markdown(
    single: list[Comparison],
    multi: list[Comparison],
    threshold_pct: float,
) -> tuple[str, str]:
    failed = [row for row in single + multi if row.failed]
    unmeasured = [row for row in single + multi if row.result == "NOT_MEASURED"]
    model_count = len(single)

    summary_lines = [
        "## Model Zoo Performance Gate",
        "",
        f"- Models: {model_count}",
        f"- Threshold: more than {threshold_pct:g}% slower than base",
        "- Single stream metric: average latency (lower is better)",
        "- Multi stream metric: QPS (higher is better)",
        f"- Result: {'FAIL' if failed else 'PASS'}",
    ]
    if failed:
        summary_lines.extend(["", "### Regressions", ""])
        for row in failed:
            summary_lines.append(
                f"- {row.model}: {row.result}, change {_format_change(row.change_pct)}"
            )
    if unmeasured:
        summary_lines.extend(
            ["", f"- Skipped in both versions (not measured): {len(unmeasured)}"]
        )
    summary = "\n".join(summary_lines) + "\n"

    lines = [
        "# Model Zoo Performance Comparison",
        "",
        "| Key | Value |",
        "|---|---|",
        f"| Threshold | {threshold_pct:g}% |",
        "| Single stream metric | Average latency, lower is better |",
        "| Multi stream metric | QPS, higher is better |",
        f"| Models compared | {model_count} |",
        f"| Regressions | {len(failed)} |",
        f"| Result | {'FAIL' if failed else 'PASS'} |",
        "",
    ]
    lines.extend(
        _table_for_mode(
            "Single Stream",
            single,
            "single",
            "latency (ms)",
            "latency (ms)",
            "Latency change",
        )
    )
    lines.extend(
        _table_for_mode(
            "Multi Stream (-s 4 -q 128)",
            multi,
            "multi",
            "QPS",
            "QPS",
            "QPS change",
        )
    )
    lines.extend(["## Result", ""])
    if failed:
        lines.append(
            f"FAIL: {len(failed)} model measurement(s) regressed by more than "
            f"{threshold_pct:g}% or could not be measured."
        )
    else:
        lines.append(
            f"PASS: no model regressed by more than {threshold_pct:g}% "
            "in single-stream latency or multi-stream QPS."
        )
    lines.append("")
    return "\n".join(lines) + "\n", summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-single", required=True, type=Path)
    parser.add_argument("--candidate-single", required=True, type=Path)
    parser.add_argument("--baseline-multi", required=True, type=Path)
    parser.add_argument("--candidate-multi", required=True, type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Maximum allowed slowdown in percent (default: 10).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(os.environ["GITHUB_STEP_SUMMARY"])
        if os.environ.get("GITHUB_STEP_SUMMARY")
        else None,
        help="Optional markdown summary file, appended if it exists.",
    )
    args = parser.parse_args()

    if not math.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be a non-negative number")

    try:
        single = compare_reports(
            parse_report(args.baseline_single, "Average Latency"),
            parse_report(args.candidate_single, "Average Latency"),
            "single",
            args.threshold,
        )
        multi = compare_reports(
            parse_report(args.baseline_multi, "QPS"),
            parse_report(args.candidate_multi, "QPS"),
            "multi",
            args.threshold,
        )
    except (OSError, ValueError) as exc:
        print(f"Failed to compare reports: {exc}", file=sys.stderr)
        return 1

    single_models = {row.model for row in single}
    multi_models = {row.model for row in multi}
    if single_models != multi_models:
        missing_from_multi = sorted(single_models - multi_models)
        missing_from_single = sorted(multi_models - single_models)
        print(
            "Single-stream and multi-stream model sets differ; "
            f"single-only={missing_from_multi}, multi-only={missing_from_single}",
            file=sys.stderr,
        )
        return 1

    markdown, summary = build_markdown(single, multi, args.threshold)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Saved performance comparison to {args.output}")
    else:
        print(markdown, end="")

    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as stream:
            stream.write(summary)

    failed = [row for row in single + multi if row.failed]
    if failed:
        print(f"Performance gate failed: {len(failed)} regression(s)", file=sys.stderr)
        for row in failed:
            print(
                f"  {row.model}: {row.result}, change {_format_change(row.change_pct)}",
                file=sys.stderr,
            )
        return 1

    print("Performance gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
