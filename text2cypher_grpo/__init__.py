"""Text2Cypher GRPO — sub-2B specialist training package.

See ../plan.md for design and ../report.md for research/verification.
"""

from text2cypher_grpo.schema import schema_to_kuzu_ddl, SchemaParseError
from text2cypher_grpo.cypher_graph import parse_cypher_to_graph
from text2cypher_grpo.rewards import (
    reward_judge,
    reward_string,
    reward_ged,
    reward_length_penalty,
    build_reward_funcs,
)
from text2cypher_grpo.db_backend import (
    CypherBackend,
    KuzuBackend,
    ExecutionResult,
    get_backend,
)
from text2cypher_grpo.eval_harness import evaluate
from text2cypher_grpo.checkpoint import maybe_resume, push_eval_results

__all__ = [
    "schema_to_kuzu_ddl",
    "SchemaParseError",
    "parse_cypher_to_graph",
    "reward_judge",
    "reward_string",
    "reward_ged",
    "reward_length_penalty",
    "build_reward_funcs",
    "CypherBackend",
    "KuzuBackend",
    "ExecutionResult",
    "get_backend",
    "evaluate",
    "maybe_resume",
    "push_eval_results",
]
