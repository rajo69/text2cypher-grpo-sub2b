"""Convert a Cypher query into a networkx property-graph for R_GED.

Strategy: regex extraction of MATCH/CREATE/MERGE patterns. No parser dependency
(see plan.md §5 for why — abandoned Python Cypher parsers cause more pain than
they save for our subset of the language).

The graph captures:
  - Nodes labeled by `:Label`
  - Edges typed by `:REL`
  - Variables bound to nodes/edges so subsequent property accesses (WHERE a.foo)
    attach to the right element
"""

from __future__ import annotations

import re
import networkx as nx

# Strip string literals first (they can contain stray parens/brackets)
_STRING_LIT_RE = re.compile(r"'[^']*'|\"[^\"]*\"")

# Tokenize a path: alternating (node) / [rel] segments separated by - or ->
# We extract whole MATCH/CREATE/MERGE clause bodies first, then walk segments.
_CLAUSE_RE = re.compile(
    r"\b(?:MATCH|OPTIONAL\s+MATCH|CREATE|MERGE)\b\s*(.+?)(?=\b(?:WHERE|RETURN|WITH|MATCH|OPTIONAL\s+MATCH|CREATE|MERGE|DELETE|SET|REMOVE|UNWIND|UNION|CALL|FOREACH|ORDER\s+BY|LIMIT|SKIP|$)\b)",
    re.IGNORECASE | re.DOTALL,
)

# Node pattern: (var:Label {props}) — all components optional except enclosing parens
_NODE_RE = re.compile(
    r"\(\s*([A-Za-z_][A-Za-z0-9_]*)?\s*(?::\s*([A-Za-z_][A-Za-z0-9_]*(?:\s*\|\s*[A-Za-z_][A-Za-z0-9_]*)*))?\s*(\{[^}]*\})?\s*\)"
)

# Rel pattern: -[var:TYPE {props}]- or -[var:TYPE]-> etc.
# We don't anchor on direction here; we'll detect the >.
_REL_RE = re.compile(
    r"-\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)?\s*(?::\s*([A-Za-z_][A-Za-z0-9_]*(?:\s*\|\s*[A-Za-z_][A-Za-z0-9_]*)*))?\s*(\*[^]]*)?\s*(\{[^}]*\})?\s*\]\s*(-?\>?)",
    re.DOTALL,
)


def _strip_strings(s: str) -> str:
    return _STRING_LIT_RE.sub("''", s)


def parse_cypher_to_graph(query: str) -> nx.MultiDiGraph:
    """Best-effort conversion of a Cypher query into a property-multigraph.

    Returns an empty graph if the query has no recognizable patterns.
    """
    g = nx.MultiDiGraph()
    if not query or not isinstance(query, str):
        return g

    cleaned = _strip_strings(query)

    # Var → graph-node-id map; we synthesize ids since Cypher is "MATCH (a)-[]->(b)" not numbered
    var_to_id: dict[str, str] = {}
    next_id = [0]

    def node_id_for(var: str | None, label: str | None) -> str:
        if var and var in var_to_id:
            return var_to_id[var]
        nid = f"n{next_id[0]}"
        next_id[0] += 1
        if var:
            var_to_id[var] = nid
        g.add_node(nid, label=label or "_unlabeled_")
        return nid

    for clause_m in _CLAUSE_RE.finditer(cleaned):
        body = clause_m.group(1)
        # Walk the body in linear order, alternating node / rel using a state-machine-ish approach.
        # We collect node positions and rel positions then thread them.
        nodes = [(m.start(), m.end(), m) for m in _NODE_RE.finditer(body)]
        rels = [(m.start(), m.end(), m) for m in _REL_RE.finditer(body)]
        if not nodes:
            continue
        # Process each rel in order: it relates nodes[i] and nodes[i+1] where both appear within
        # the same path segment (no comma between them).
        # Simplification: for each pair of consecutive nodes in the body, check if a rel pattern
        # appears between them. Cypher allows comma-separated patterns; treat each segment.
        # Split body into segments at commas at depth 0 (ignore commas inside braces).
        segments = _split_top_level_commas(body)
        for seg in segments:
            seg_nodes = list(_NODE_RE.finditer(seg))
            seg_rels = list(_REL_RE.finditer(seg))
            if not seg_nodes:
                continue
            # Build node ids in order
            seg_node_ids = []
            for nm in seg_nodes:
                var = nm.group(1)
                label = nm.group(2)
                if label and "|" in label:
                    label = label.split("|")[0].strip()  # take first alt
                nid = node_id_for(var, label)
                seg_node_ids.append(nid)
            # For each rel between two consecutive nodes
            for i, rm in enumerate(seg_rels):
                if i + 1 >= len(seg_node_ids):
                    break
                src = seg_node_ids[i]
                dst = seg_node_ids[i + 1]
                rel_type = rm.group(2) or "_unlabeled_"
                if "|" in rel_type:
                    rel_type = rel_type.split("|")[0].strip()
                arrow = rm.group(5) or ""
                # If the rel ends with -> direction is src->dst; if not arrow, undirected → add both
                if ">" in arrow:
                    g.add_edge(src, dst, type=rel_type)
                else:
                    # Without explicit direction, add both directions so GED is direction-agnostic
                    g.add_edge(src, dst, type=rel_type)
                    g.add_edge(dst, src, type=rel_type)
    return g


def _split_top_level_commas(s: str) -> list[str]:
    """Split a string by top-level commas (ignoring commas inside (), [], or {})."""
    out: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out
