"""Convert the heterogeneous `schema` column from neo4j/text2cypher-2024v1 into kuzu DDL.

Handles two observed formats:

  Format A — markdown / textual:
      "Node properties: - **Product** - `productName`: STRING ...
       Relationships: (:Topic)-[:HAS_TOPIC]->(:Article) ..."

  Format B — APOC-style JSON metadata:
      '{"ASSIGNED_TO": {"count": 27, "type": "relationship"}, ...}'

Strategy: try JSON first; on failure, apply markdown heuristics.
Always succeeds (worst case → minimal Any-Any DDL) so callers don't crash mid-rollout.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

CYPHER_TYPES_TO_KUZU = {
    "STRING": "STRING",
    "INTEGER": "INT64",
    "INT": "INT64",
    "INT64": "INT64",
    "FLOAT": "DOUBLE",
    "DOUBLE": "DOUBLE",
    "BOOLEAN": "BOOL",
    "BOOL": "BOOL",
    "DATE": "DATE",
    "DATETIME": "TIMESTAMP",
    "LIST": "STRING",  # kuzu lists need element type — degrade to STRING (JSON-encoded)
}


class SchemaParseError(Exception):
    pass


@dataclass
class _Schema:
    nodes: dict[str, dict[str, str]] = field(default_factory=dict)  # label -> {prop: type}
    rels: list[tuple[str, str, str, dict[str, str]]] = field(default_factory=list)
    # rels: (rel_type, from_label, to_label, props)

    def to_ddl(self) -> str:
        lines: list[str] = []
        labels = set(self.nodes.keys())
        for _, src, dst, _ in self.rels:
            labels.add(src)
            labels.add(dst)
        if not labels:
            labels = {"Any"}
        for label in sorted(labels):
            props = self.nodes.get(label, {})
            prop_clauses = [f"{p} {CYPHER_TYPES_TO_KUZU.get(t.upper(), 'STRING')}" for p, t in props.items()]
            cols = ", ".join(["id INT64"] + prop_clauses + ["PRIMARY KEY (id)"])
            lines.append(f"CREATE NODE TABLE {label}({cols});")
        seen_rel_keys = set()
        for rel_type, src, dst, props in self.rels:
            key = (rel_type, src, dst)
            if key in seen_rel_keys:
                continue
            seen_rel_keys.add(key)
            prop_clauses = [f"{p} {CYPHER_TYPES_TO_KUZU.get(t.upper(), 'STRING')}" for p, t in props.items()]
            from_to = f"FROM {src} TO {dst}"
            extras = (", " + ", ".join(prop_clauses)) if prop_clauses else ""
            lines.append(f"CREATE REL TABLE {rel_type}({from_to}{extras});")
        return "\n".join(lines)


def _parse_json_format(s: str) -> _Schema:
    data = json.loads(s)
    sch = _Schema()
    for key, info in data.items():
        if not isinstance(info, dict):
            continue
        kind = info.get("type", "node")
        if kind == "relationship":
            # APOC JSON often omits endpoints; default Any→Any. Some variants include 'relationships'.
            from_label = "Any"
            to_label = "Any"
            for endpoint_key in ("from", "start"):
                if endpoint_key in info and isinstance(info[endpoint_key], list) and info[endpoint_key]:
                    from_label = info[endpoint_key][0]
            for endpoint_key in ("to", "end"):
                if endpoint_key in info and isinstance(info[endpoint_key], list) and info[endpoint_key]:
                    to_label = info[endpoint_key][0]
            props = {p: pi.get("type", "STRING") for p, pi in info.get("properties", {}).items() if isinstance(pi, dict)}
            sch.rels.append((_clean_label(key), _clean_label(from_label), _clean_label(to_label), props))
        else:
            label = _clean_label(key)
            props = {p: pi.get("type", "STRING") for p, pi in info.get("properties", {}).items() if isinstance(pi, dict)}
            sch.nodes[label] = props
    return sch


_NODE_BLOCK_RE = re.compile(r"\*\*([A-Za-z_][A-Za-z0-9_]*)\*\*")  # **Label**
_PROP_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`\s*[:\-]\s*([A-Z]+)")  # `name`: TYPE
_REL_RE = re.compile(
    r"\(\s*:?\s*([A-Za-z_][A-Za-z0-9_]*)?\s*\)\s*-\s*\[\s*:?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]\s*-\s*>?\s*\(\s*:?\s*([A-Za-z_][A-Za-z0-9_]*)?\s*\)"
)

# Format C: Cypher-pattern style without markdown formatting
#   Author {author_id: STRING}                          ← node or rel-property block
#   {'start': Article, 'type': PUBLISHED_IN, 'end': J}  ← rel definition
_FMT_C_BLOCK_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\{([^}]+)\}\s*$",
    re.MULTILINE,
)
_FMT_C_PROP_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_]+)"
)
_FMT_C_REL_RE = re.compile(
    r"\{\s*['\"]?start['\"]?\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*,"
    r"\s*['\"]?type['\"]?\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*,"
    r"\s*['\"]?end['\"]?\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}"
)


def _parse_markdown_format(s: str) -> _Schema:
    sch = _Schema()
    # Find node label blocks (**Label**) and the property lines that follow each
    # Approach: split by **Label** and parse the trailing chunk for properties.
    parts = re.split(r"\*\*([A-Za-z_][A-Za-z0-9_]*)\*\*", s)
    # parts is alternating: [pre, label1, body1, label2, body2, ...]
    if len(parts) >= 3:
        for i in range(1, len(parts) - 1, 2):
            label = _clean_label(parts[i])
            body = parts[i + 1]
            # Stop at next "Relationships" / "Relationship" header to avoid eating rels
            body = re.split(r"(?i)relationships?:|relationship properties:", body, maxsplit=1)[0]
            props: dict[str, str] = {}
            for m in _PROP_RE.finditer(body):
                props[m.group(1)] = m.group(2)
            if label and label not in sch.nodes:
                sch.nodes[label] = props
            elif label:
                sch.nodes[label].update(props)
    # Relationship patterns
    for m in _REL_RE.finditer(s):
        src = _clean_label(m.group(1) or "Any")
        rel_type = _clean_label(m.group(2))
        dst = _clean_label(m.group(3) or "Any")
        if rel_type:
            sch.rels.append((rel_type, src, dst, {}))
    return sch


def _clean_label(s: str) -> str:
    s = s.strip()
    # kuzu identifiers — alphanumeric + underscore; replace anything else
    return re.sub(r"[^A-Za-z0-9_]", "_", s) or "Any"


def _parse_format_c(s: str) -> _Schema:
    """Format C — Cypher-pattern lines without markdown formatting.

    Examples:
        Author {author_id: STRING}
        Topic {label: STRING}
        {'start': Article, 'type': PUBLISHED_IN, 'end': Journal}
        PUBLISHED_IN {pages: STRING}     ← rel-property block
    """
    sch = _Schema()
    # First find rel definitions; we need them to disambiguate Label{} blocks below.
    rel_definitions: list[tuple[str, str, str]] = []
    for m in _FMT_C_REL_RE.finditer(s):
        rel_type = _clean_label(m.group(2))
        src = _clean_label(m.group(1))
        dst = _clean_label(m.group(3))
        rel_definitions.append((rel_type, src, dst))
    rel_types = {rt for rt, _, _ in rel_definitions}
    rel_props: dict[str, dict[str, str]] = {rt: {} for rt in rel_types}

    for m in _FMT_C_BLOCK_RE.finditer(s):
        label = _clean_label(m.group(1))
        body = m.group(2)
        props: dict[str, str] = {}
        for pm in _FMT_C_PROP_RE.finditer(body):
            props[pm.group(1)] = pm.group(2)
        if not props:
            continue
        if label in rel_types:
            rel_props[label].update(props)
        else:
            if label in sch.nodes:
                sch.nodes[label].update(props)
            else:
                sch.nodes[label] = props

    for rt, src, dst in rel_definitions:
        sch.rels.append((rt, src, dst, rel_props.get(rt, {})))
    return sch


def _merge_into(dst: _Schema, src: _Schema) -> None:
    for label, props in src.nodes.items():
        if label in dst.nodes:
            dst.nodes[label].update(props)
        else:
            dst.nodes[label] = dict(props)
    seen = {(rt, s_, d_) for rt, s_, d_, _ in dst.rels}
    for rel in src.rels:
        if (rel[0], rel[1], rel[2]) not in seen:
            dst.rels.append(rel)
            seen.add((rel[0], rel[1], rel[2]))


def schema_to_kuzu_ddl(schema_text: str) -> str:
    """Convert a `schema` column entry to executable kuzu DDL.

    Never raises on user data — always returns a string. If the schema is uninterpretable,
    returns minimal Any DDL so the caller can still attempt query execution against it
    (most queries will fail, which the reward function correctly grades as low quality).

    Tries all three known formats (JSON / markdown-bold / Cypher-pattern Format C) and
    merges the results — some schemas mix styles.
    """
    if not schema_text or not isinstance(schema_text, str):
        return "CREATE NODE TABLE Any(id INT64, PRIMARY KEY (id));"
    s = schema_text.strip()

    sch = _Schema()

    # Format B — APOC JSON
    if s.startswith("{") and s.endswith("}"):
        try:
            _merge_into(sch, _parse_json_format(s))
        except json.JSONDecodeError:
            pass

    # Format A — markdown bold + backtick props
    _merge_into(sch, _parse_markdown_format(s))

    # Format C — Cypher-pattern Label {prop: TYPE} lines, plus dict-literal rel definitions
    _merge_into(sch, _parse_format_c(s))

    if not sch.nodes and not sch.rels:
        return "CREATE NODE TABLE Any(id INT64, PRIMARY KEY (id));"
    return sch.to_ddl()


def parse_rate(schemas: list[str]) -> dict:
    """Diagnostic: how many schemas yielded non-trivial DDL.

    Returns {"total": N, "trivial": K, "rate": (N-K)/N}.
    """
    total = len(schemas)
    trivial = 0
    for s in schemas:
        ddl = schema_to_kuzu_ddl(s)
        if ddl.strip() == "CREATE NODE TABLE Any(id INT64, PRIMARY KEY (id));":
            trivial += 1
    return {"total": total, "trivial": trivial, "non_trivial_rate": (total - trivial) / max(total, 1)}
