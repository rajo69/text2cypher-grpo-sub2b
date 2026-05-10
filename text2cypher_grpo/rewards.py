"""GRPO reward functions — STRuCT-LLM pattern with execution-based judge replacement.

R = R_judge + R_string + R_GED  (equal weights, untuned, mirroring STRuCT-LLM)

R_judge   — execution grade (replaces STRuCT-LLM's o3-mini judge; we have no API on free tier)
R_string  — difflib SequenceMatcher ratio
R_GED     — networkx graph_edit_distance, normalized

A length-penalty guardrail is provided separately as `reward_length_penalty` to mitigate
GRPO's known length-hacking failure mode.

TRL signature (verified TRL v1.4.0):
    def reward_fn(prompts, completions, completion_ids, trainer_state, **kwargs) -> list[float]
where dataset columns are auto-passed as kwargs (so `cypher` and `schema` arrive named).
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any

import networkx as nx

from text2cypher_grpo.cypher_graph import parse_cypher_to_graph
from text2cypher_grpo.db_backend import CypherBackend, KuzuBackend
from text2cypher_grpo.schema import schema_to_kuzu_ddl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cypher extraction from completions

# The model is expected to emit Cypher. If it wraps in ```cypher ... ``` or other framing,
# extract just the Cypher portion. Otherwise treat the whole completion as Cypher.
_FENCE_RE = re.compile(r"```(?:cypher)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def extract_cypher(completion: str) -> str:
    if not completion:
        return ""
    m = _FENCE_RE.search(completion)
    if m:
        return m.group(1).strip()
    # Otherwise, take everything after the first MATCH/CREATE/MERGE/CALL up to end
    keyword_m = re.search(r"\b(MATCH|OPTIONAL\s+MATCH|CREATE|MERGE|CALL|RETURN|WITH)\b", completion, re.IGNORECASE)
    if keyword_m:
        return completion[keyword_m.start():].strip()
    return completion.strip()


# ---------------------------------------------------------------------------
# R_judge — execution-based replacement for STRuCT-LLM's o3-mini judge

def _judge_one(pred_query: str, gold_query: str, schema_text: str, backend: CypherBackend) -> float:
    """Execute pred and gold against synthesized DB. Grade:
        0.0 — pred fails to parse
        0.3 — pred parses but errors at runtime
        0.7 — pred executes successfully
        1.0 — pred executes AND result-set matches gold (order-insensitive)
    """
    pred_clean = extract_cypher(pred_query)
    if not pred_clean:
        return 0.0
    schema_ddl = schema_to_kuzu_ddl(schema_text)

    # First: just run the prediction
    pred_result = backend.execute(schema_ddl, pred_clean, timeout_ms=2000)
    if not pred_result.success:
        return pred_result.grade()  # 0.0 parse, 0.3 runtime/etc.

    # Now compare to gold to bump 0.7 → 1.0 if results match
    gold_result = backend.execute(schema_ddl, gold_query, timeout_ms=2000)
    if not gold_result.success:
        # If even the gold query fails on synthesized DB, we can only credit execution
        return 0.7

    pred_set = _normalize_rows(pred_result.rows)
    gold_set = _normalize_rows(gold_result.rows)
    if pred_set == gold_set:
        return 1.0
    return 0.7


def _normalize_rows(rows: list | None) -> frozenset:
    if not rows:
        return frozenset()
    out = []
    for r in rows:
        try:
            out.append(tuple(_to_hashable(c) for c in r))
        except Exception:
            out.append(tuple(str(c) for c in r))
    return frozenset(out)


def _to_hashable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool, type(None))):
        return x
    if isinstance(x, (list, tuple)):
        return tuple(_to_hashable(c) for c in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _to_hashable(v)) for k, v in x.items()))
    return str(x)


def reward_judge(backend: CypherBackend | None = None):
    """Build the execution-based judge reward callable for GRPOTrainer.reward_funcs."""
    backend = backend or KuzuBackend()

    def _fn(prompts, completions, completion_ids=None, trainer_state=None, **kwargs):
        cyphers = kwargs.get("cypher")
        schemas = kwargs.get("schema")
        if cyphers is None or schemas is None:
            logger.warning("reward_judge: missing 'cypher' or 'schema' kwargs; returning 0.0")
            return [0.0] * len(completions)
        out: list[float] = []
        for comp, gold, sch in zip(completions, cyphers, schemas):
            try:
                out.append(_judge_one(comp, gold, sch, backend))
            except Exception as e:
                logger.debug("judge exception: %s", e)
                out.append(0.0)
        return out

    _fn.__name__ = "reward_judge"
    return _fn


# ---------------------------------------------------------------------------
# R_string — SequenceMatcher ratio

def _string_one(pred_query: str, gold_query: str) -> float:
    pred_clean = extract_cypher(pred_query)
    if not pred_clean or not gold_query:
        return 0.0
    return difflib.SequenceMatcher(None, gold_query.strip(), pred_clean.strip()).ratio()


def reward_string(prompts, completions, completion_ids=None, trainer_state=None, **kwargs):
    cyphers = kwargs.get("cypher")
    if cyphers is None:
        return [0.0] * len(completions)
    return [_string_one(c, g) for c, g in zip(completions, cyphers)]


# ---------------------------------------------------------------------------
# R_GED — graph_edit_distance via networkx

def _ged_one(pred_query: str, gold_query: str, max_size: int = 20, timeout_s: float = 2.0) -> float:
    pred_clean = extract_cypher(pred_query)
    if not pred_clean or not gold_query:
        return 0.0
    g_pred = parse_cypher_to_graph(pred_clean)
    g_gold = parse_cypher_to_graph(gold_query)
    size_pred = g_pred.number_of_nodes() + g_pred.number_of_edges()
    size_gold = g_gold.number_of_nodes() + g_gold.number_of_edges()
    if size_pred == 0 or size_gold == 0:
        return 0.0
    if size_pred > max_size or size_gold > max_size:
        # Skip GED for outsized queries — return modest match if labels overlap
        pred_labels = {d.get("label") for _, d in g_pred.nodes(data=True)}
        gold_labels = {d.get("label") for _, d in g_gold.nodes(data=True)}
        overlap = len(pred_labels & gold_labels)
        return overlap / max(len(gold_labels), 1) * 0.5
    # optimize_graph_edit_distance is a generator yielding decreasing approximations
    try:
        gen = nx.algorithms.similarity.optimize_graph_edit_distance(
            g_pred, g_gold,
            node_match=_node_match,
            edge_match=_edge_match,
        )
        # Take first approximation only — pathological cases would otherwise hang.
        ged = next(iter(gen), float(max(size_pred, size_gold)))
    except Exception:
        ged = float(max(size_pred, size_gold))
    denom = max(size_pred, size_gold, 1)
    return max(0.0, 1.0 - ged / denom)


def _node_match(a: dict, b: dict) -> bool:
    return a.get("label") == b.get("label")


def _edge_match(a: dict, b: dict) -> bool:
    return a.get("type") == b.get("type")


def reward_ged(prompts, completions, completion_ids=None, trainer_state=None, **kwargs):
    cyphers = kwargs.get("cypher")
    if cyphers is None:
        return [0.0] * len(completions)
    return [_ged_one(c, g) for c, g in zip(completions, cyphers)]


# ---------------------------------------------------------------------------
# Length-penalty guardrail (anti-length-hacking)

def reward_length_penalty(
    prompts, completions, completion_ids=None, trainer_state=None, target_tokens: int = 256, **kwargs
):
    """Soft penalty for runaway-length completions.

    Returns 0 for completions ≤ target_tokens, then linearly decreases to -0.5 at 2× target.
    """
    out: list[float] = []
    for ids in (completion_ids or []):
        n = len(ids) if ids is not None else 0
        if n <= target_tokens:
            out.append(0.0)
        else:
            excess = (n - target_tokens) / max(target_tokens, 1)
            out.append(max(-0.5, -0.5 * min(excess, 1.0)))
    if not out and completions:
        # Fallback if completion_ids not provided
        out = [0.0] * len(completions)
    return out


# ---------------------------------------------------------------------------
# Bundle all four into a list with weights = (1, 1, 1, 1)

def build_reward_funcs(backend: CypherBackend | None = None) -> list:
    """Build the canonical reward list. Pass to GRPOConfig as `reward_funcs=build_reward_funcs()`.

    Note: TRL sums the per-fn rewards (or weighted-sums via GRPOConfig.reward_weights).
    """
    return [
        reward_judge(backend),
        reward_string,
        reward_ged,
        reward_length_penalty,
    ]
