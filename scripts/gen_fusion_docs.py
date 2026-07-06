#!/usr/bin/env python3
"""Generate MUSA fusion docs by extracting fusion structure from C++ sources."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EP_CC = REPO / "musa" / "ep" / "src" / "ep.cc"
COMPUTE_CC = REPO / "musa" / "ep" / "src" / "fusion" / "fusion_node_compute.cc"
FUSION_DIR = REPO / "musa" / "ep" / "src" / "fusion"
OUTPUT_DIR = REPO / "musa" / "docs" / "fusion"
PRIORITY_OUTPUT = REPO / "musa" / "docs" / "fusion_priority.md"

ONNX_OP_RE = re.compile(r'IsOnnxOp\s*\([^,]+,\s*"([^"]+)"\s*\)')


@dataclass
class CapabilityEntry:
    variable: str
    finder: str
    drop_constant_initializers: str


@dataclass
class CompileEntry:
    detector: str
    factory: str


@dataclass
class FactoryInfo:
    factory: str
    source: Path
    compute_type: str
    ops: list[str]


@dataclass
class DetectorInfo:
    detector: str
    source: Path
    ops: list[str]


@dataclass
class FusionInfo:
    key: str
    title: str
    slug: str
    capability: CapabilityEntry
    compile_entry: CompileEntry
    source: Path
    compute_type: str
    extracted_ops: list[str]
    pattern_labels: list[str]


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _extract_function_body(text: str, name: str) -> str:
    search_from = 0
    needle = f"{name}("
    while True:
        start = text.find(needle, search_from)
        if start < 0:
            raise ValueError(f"unable to find function body: {name}")
        open_paren = start + len(name)
        close_paren = _find_matching(text, open_paren, "(", ")")
        if close_paren < 0:
            search_from = start + len(needle)
            continue
        open_brace = text.find("{", close_paren + 1)
        semicolon = text.find(";", close_paren + 1)
        if open_brace >= 0 and (semicolon < 0 or open_brace < semicolon):
            close_brace = _find_matching(text, open_brace, "{", "}")
            if close_brace < 0:
                raise ValueError(f"unterminated function body: {name}")
            return text[open_brace + 1 : close_brace]
        search_from = start + len(needle)


def _find_matching(text: str, open_index: int, open_char: str, close_char: str) -> int:
    if open_index >= len(text) or text[open_index] != open_char:
        return -1
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _ops_from_text(text: str) -> list[str]:
    return _ordered_unique(ONNX_OP_RE.findall(_strip_comments(text)))


def _camel_words(name: str) -> list[str]:
    return re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", name)


def _stem_words(stem: str, known_ops: set[str]) -> list[str]:
    words: list[str] = []
    index = 0
    op_names = sorted(known_ops, key=len, reverse=True)
    while index < len(stem):
        match = next((op for op in op_names if stem.startswith(op, index)), None)
        if match is not None:
            words.append(match)
            index += len(match)
            continue
        m = re.match(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", stem[index:])
        if not m:
            words.append(stem[index])
            index += 1
            continue
        words.append(m.group(0))
        index += len(m.group(0))
    return words


def _stem_from_finder(finder: str) -> str:
    if not finder.startswith("Find") or not finder.endswith("Fusions"):
        raise ValueError(f"unexpected finder name: {finder}")
    return finder[len("Find") : -len("Fusions")]


def _stem_from_factory(factory: str) -> str:
    if not factory.startswith("Create") or not factory.endswith("Fusion"):
        raise ValueError(f"unexpected factory name: {factory}")
    return factory[len("Create") : -len("Fusion")]


def _key_from_stem(stem: str, known_ops: set[str]) -> str:
    return "_".join(word.lower() for word in _stem_words(stem, known_ops))


def _title_from_stem(stem: str, known_ops: set[str]) -> str:
    return " + ".join(_stem_words(stem, known_ops)) + " Fusion"


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO))


def _capability_entries() -> list[CapabilityEntry]:
    text = _strip_comments(EP_CC.read_text())
    body = _extract_function_body(text, "MusaEp::GetCapabilityImpl")
    assignments = {
        m.group("var"): m.group("func")
        for m in re.finditer(
            r"std::vector<std::vector<Ort::ConstNode>>\s+(?P<var>\w+)\s*=\s*"
            r"(?P<func>Find\w+Fusions)\s*\(",
            body,
        )
    }
    entries: list[CapabilityEntry] = []
    for loop in re.finditer(
        r"for\s*\(\s*const auto& fusion_nodes\s*:\s*(?P<var>\w+)\s*\)\s*\{"
        r"(?P<body>.*?EpGraphSupportInfo_AddNodesToFuse\s*\([^;]+;\s*)\}",
        body,
        re.DOTALL,
    ):
        variable = loop.group("var")
        finder = assignments.get(variable)
        if finder is None:
            continue
        drop = re.search(
            r"drop_constant_initializers\s*=\s*(true|false)", loop.group("body")
        )
        entries.append(
            CapabilityEntry(
                variable=variable,
                finder=finder,
                drop_constant_initializers=drop.group(1) if drop else "unknown",
            )
        )
    return entries


def _compile_entries() -> list[CompileEntry]:
    text = _strip_comments(COMPUTE_CC.read_text())
    body = _extract_function_body(text, "MusaEp::CompileImpl")
    entries: list[CompileEntry] = []
    for m in re.finditer(
        r"(?:if|else if)\s*\(\s*(?P<detector>Is\w+FusionGraph)\s*\(\s*graph\s*\)\s*\)"
        r"\s*\{\s*ep->GetFusionComputes\(\)\[fused_node_name\]\s*=\s*"
        r"(?P<factory>Create\w+Fusion)\s*\(",
        body,
        re.DOTALL,
    ):
        entries.append(CompileEntry(detector=m.group("detector"), factory=m.group("factory")))
    fallback = re.search(
        r"else\s*\{\s*ep->GetFusionComputes\(\)\[fused_node_name\]\s*=\s*"
        r"(?P<factory>Create\w+Fusion)\s*\(",
        body,
        re.DOTALL,
    )
    if fallback:
        entries.append(CompileEntry(detector="fallback", factory=fallback.group("factory")))
    return entries


def _fusion_sources() -> list[tuple[Path, str]]:
    paths = sorted(FUSION_DIR.glob("*.cc")) + sorted(FUSION_DIR.glob("*.h"))
    return [(path, _strip_comments(path.read_text())) for path in paths]


def _ops_and_labels_from_function_and_helpers(text: str, function: str) -> tuple[list[str], list[str]]:
    body = _extract_function_body(text, function)
    token_re = re.compile(
        r'IsOnnxOp\s*\([^,]+,\s*"(?P<op>[^"]+)"\s*\)|'
        r"\b(?P<helper>Is\w+|CanFuse\w+)\s*\("
    )
    ops: list[str] = []
    labels: list[str] = []
    seen_labels: set[str] = set()
    for match in token_re.finditer(body):
        op = match.group("op")
        if op is not None:
            ops.append(op)
            if op not in seen_labels:
                labels.append(op)
                seen_labels.add(op)
            continue
        helper = match.group("helper")
        if helper is None or helper == function or helper == "IsOnnxOp":
            continue
        try:
            helper_body = _extract_function_body(text, helper)
        except ValueError:
            continue
        helper_ops = _ops_from_text(helper_body)
        ops.extend(helper_ops)
        if helper_ops:
            label = " / ".join(helper_ops)
            if label not in seen_labels:
                labels.append(label)
                seen_labels.add(label)
    return _ordered_unique(ops), labels


def _ops_from_function_and_helpers(text: str, function: str) -> list[str]:
    ops, _labels = _ops_and_labels_from_function_and_helpers(text, function)
    return ops


def _labels_from_function_and_helpers(text: str, function: str) -> list[str]:
    _ops, labels = _ops_and_labels_from_function_and_helpers(text, function)
    return labels
    for helper in helper_names:
        if helper == function or helper == "IsOnnxOp":
            continue
        try:
            helper_body = _extract_function_body(text, helper)
        except ValueError:
            continue
        ops.extend(_ops_from_text(helper_body))
    return _ordered_unique(ops)


def _source_with_function(name: str, sources: list[tuple[Path, str]]) -> tuple[Path, str, str]:
    for path, text in sources:
        try:
            body = _extract_function_body(text, name)
        except ValueError:
            continue
        return path, text, body
    raise ValueError(f"unable to find function in fusion sources: {name}")


def _factory_infos(sources: list[tuple[Path, str]]) -> dict[str, FactoryInfo]:
    infos: dict[str, FactoryInfo] = {}
    factories = sorted(
        {
            match.group(1)
            for path, text in sources
            if path.suffix == ".cc"
            for match in re.finditer(
                r"std::unique_ptr<FusionNodeCompute>\s+(Create\w+Fusion)\s*\(", text
            )
        }
    )
    for factory in factories:
        source, source_text, body = _source_with_function(factory, sources)
        compute_match = re.search(r"make_unique<(\w+FusionCompute)>", body)
        if compute_match:
            compute_type = compute_match.group(1)
        else:
            compute_types = _ordered_unique(
                re.findall(r"\b(?:struct|class)\s+(\w+FusionCompute)\b", source_text)
            )
            if not compute_types:
                compute_types = _ordered_unique(
                    re.findall(r"\b(\w+FusionCompute)::Compute\s*\(", source_text)
                )
            compute_type = compute_types[0] if compute_types else factory
        infos[factory] = FactoryInfo(
            factory=factory,
            source=source,
            compute_type=compute_type,
            ops=_ops_from_text(source_text),
        )
    return infos


def _detector_infos(sources: list[tuple[Path, str]]) -> dict[str, DetectorInfo]:
    infos: dict[str, DetectorInfo] = {}
    detectors = sorted(
        {
            match.group(1)
            for _path, text in sources
            for match in re.finditer(r"\bbool\s+(Is\w+FusionGraph)\s*\(", text)
        }
    )
    for detector in detectors:
        source, _source_text, body = _source_with_function(detector, sources)
        helper_ops = _ops_from_function_and_helpers(text=_source_text, function=detector)
        infos[detector] = DetectorInfo(detector=detector, source=source, ops=helper_ops)
    return infos


def _finder_ops(finder: str, sources: list[tuple[Path, str]]) -> list[str]:
    ep_text = _strip_comments(EP_CC.read_text())
    ops = _ops_from_function_and_helpers(ep_text, finder)
    if ops:
        return ops

    stem_words = _camel_words(_stem_from_finder(finder))
    candidates = [
        op for op in stem_words if op not in {"Fused", "Activation"} and op[0].isupper()
    ]
    if candidates:
        return _ordered_unique(candidates)

    for _path, text in sources:
        if finder in text:
            return _ops_from_text(text)
    return []


def _finder_labels(finder: str, fallback_ops: list[str]) -> list[str]:
    ep_text = _strip_comments(EP_CC.read_text())
    labels = _labels_from_function_and_helpers(ep_text, finder)
    return labels or fallback_ops


def _compile_entry_for_finder(
    finder: str,
    finder_ops: list[str],
    compile_entries: list[CompileEntry],
    detectors: dict[str, DetectorInfo],
    factories: dict[str, FactoryInfo],
) -> CompileEntry:
    finder_stem = _stem_from_finder(finder)
    finder_set = set(finder_ops)

    for entry in compile_entries:
        if entry.detector == "fallback" and _stem_from_factory(entry.factory) == finder_stem:
            return entry

    best_entry: CompileEntry | None = None
    best_score = -1
    for entry in compile_entries:
        detector_ops = detectors.get(entry.detector, DetectorInfo(entry.detector, Path(), [])).ops
        factory_ops = factories.get(entry.factory, FactoryInfo(entry.factory, Path(), "", [])).ops
        target_set = set(detector_ops) | set(factory_ops)
        score = len(finder_set & target_set)
        if _stem_from_factory(entry.factory) == finder_stem:
            score += 100
        if finder_stem in entry.detector or entry.detector.replace("Is", "").replace("FusionGraph", "") in finder_stem:
            score += 50
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is None or best_score <= 0:
        raise ValueError(f"unable to map {finder} to a CompileImpl factory")
    return best_entry


def _build_fusions() -> tuple[list[FusionInfo], list[CompileEntry]]:
    capability = _capability_entries()
    compile_entries = _compile_entries()
    sources = _fusion_sources()
    known_ops = set(_ops_from_text(EP_CC.read_text()))
    for _path, text in sources:
        known_ops.update(_ops_from_text(text))
    factories = _factory_infos(sources)
    detectors = _detector_infos(sources)

    fusions: list[FusionInfo] = []
    for entry in capability:
        finder_ops = _finder_ops(entry.finder, sources)
        compile_entry = _compile_entry_for_finder(
            entry.finder, finder_ops, compile_entries, detectors, factories
        )
        factory = factories.get(compile_entry.factory)
        if factory is None:
            raise ValueError(f"factory source not found: {compile_entry.factory}")
        if not finder_ops:
            finder_ops = factory.ops
        pattern_labels = _finder_labels(entry.finder, finder_ops)
        stem = _stem_from_finder(entry.finder)
        key = _key_from_stem(stem, known_ops)
        fusions.append(
            FusionInfo(
                key=key,
                title=_title_from_stem(stem, known_ops),
                slug=f"{key}.md",
                capability=entry,
                compile_entry=compile_entry,
                source=factory.source,
                compute_type=factory.compute_type,
                extracted_ops=finder_ops,
                pattern_labels=pattern_labels,
            )
        )
    return fusions, compile_entries


def _mermaid(pattern_labels: list[str], compute_type: str) -> str:
    before = pattern_labels or ["unresolved source pattern"]
    after = ["MUSA fused node", compute_type]
    lines = ["```mermaid", "flowchart LR", "  subgraph Before[Before fusion]"]
    previous = None
    for index, label in enumerate(before):
        node = f"B{index}"
        lines.append(f'    {node}["{label}"]')
        if previous is not None:
            lines.append(f"    {previous} --> {node}")
        previous = node
    lines.extend(["  end", "  subgraph After[After fusion]"])
    previous = None
    for index, label in enumerate(after):
        node = f"A{index}"
        lines.append(f'    {node}["{label}"]')
        if previous is not None:
            lines.append(f"    {previous} --> {node}")
        previous = node
    lines.extend(["  end", "  Before ==> After", "```"])
    return "\n".join(lines)


def _render_fusion_doc(fusion: FusionInfo, priority: int) -> str:
    lines = [
        "<!-- AUTO-GENERATED by scripts/gen_fusion_docs.py. DO NOT EDIT BY HAND. -->",
        f"# {fusion.title}",
        "",
        "This file is generated from the current C++ fusion source. The before graph is derived from ONNX op names found in the finder/detector source; no per-fusion prose metadata is maintained by the generator.",
        "",
        "## Source Mapping",
        "",
        f"- GetCapability priority: **{priority}**",
        f"- Finder: `{fusion.capability.finder}`",
        f"- Compile detector: `{fusion.compile_entry.detector}`",
        f"- Runtime factory: `{fusion.compile_entry.factory}`",
        f"- Runtime compute: `{fusion.compute_type}`",
        f"- Runtime implementation: `{_relative(fusion.source)}`",
        f"- `drop_constant_initializers`: `{fusion.capability.drop_constant_initializers}`",
        "",
        "## Extracted ONNX Ops",
        "",
        ", ".join(f"`{op}`" for op in fusion.extracted_ops) if fusion.extracted_ops else "-",
        "",
        "## Mermaid",
        "",
        _mermaid(fusion.pattern_labels, fusion.compute_type),
        "",
    ]
    return "\n".join(lines)


def _render_priority_doc(fusions: list[FusionInfo], compile_entries: list[CompileEntry]) -> str:
    lines = [
        "<!-- AUTO-GENERATED by scripts/gen_fusion_docs.py. DO NOT EDIT BY HAND. -->",
        "# MUSA Fusion Priority",
        "",
        "This file is generated from `MusaEp::GetCapabilityImpl` and `MusaEp::CompileImpl`.",
        "",
        "## GetCapability Match Order",
        "",
        "| Priority | Fusion | Finder | `drop_constant_initializers` | Doc |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, fusion in enumerate(fusions, start=1):
        lines.append(
            f"| {index} | {fusion.title} | `{fusion.capability.finder}` | "
            f"`{fusion.capability.drop_constant_initializers}` | [{fusion.slug}](fusion/{fusion.slug}) |"
        )

    lines.extend(
        [
            "",
            "## Compile Dispatch Order",
            "",
            "| Priority | Detector | Factory | Related generated docs |",
            "| --- | --- | --- | --- |",
        ]
    )
    for index, entry in enumerate(compile_entries, start=1):
        related = [
            fusion
            for fusion in fusions
            if fusion.compile_entry.detector == entry.detector
            and fusion.compile_entry.factory == entry.factory
        ]
        links = ", ".join(f"[{fusion.key}](fusion/{fusion.slug})" for fusion in related) or "-"
        lines.append(f"| {index} | `{entry.detector}` | `{entry.factory}` | {links} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    try:
        fusions, compile_entries = _build_fusions()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        expected_outputs = {fusion.slug for fusion in fusions}
        for existing in OUTPUT_DIR.glob("*.md"):
            if existing.name not in expected_outputs:
                text = existing.read_text()
                if text.startswith("<!-- AUTO-GENERATED by scripts/gen_fusion_docs.py."):
                    existing.unlink()
        for index, fusion in enumerate(fusions, start=1):
            (OUTPUT_DIR / fusion.slug).write_text(_render_fusion_doc(fusion, index))
        PRIORITY_OUTPUT.write_text(_render_priority_doc(fusions, compile_entries))
    except Exception as exc:
        print(f"[gen_fusion_docs] {exc}", file=sys.stderr)
        return 1

    print(
        f"[gen_fusion_docs] wrote {len(fusions) + 1} files -> "
        f"{OUTPUT_DIR.relative_to(REPO)} and {PRIORITY_OUTPUT.relative_to(REPO)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
