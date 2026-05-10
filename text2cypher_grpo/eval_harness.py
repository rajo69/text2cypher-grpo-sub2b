"""Eval harness for text2cypher-2024v1 test split.

Reports Google-BLEU + Exact Match (Ozsoy 2024 canonical) + Execution Accuracy
(via the same backend abstraction used in training).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from text2cypher_grpo.db_backend import CypherBackend, KuzuBackend
from text2cypher_grpo.rewards import _judge_one, _normalize_rows, extract_cypher
from text2cypher_grpo.schema import schema_to_kuzu_ddl

logger = logging.getLogger(__name__)


def _normalize_for_em(query: str) -> str:
    """Lower-case keywords, collapse whitespace, strip trailing semicolon."""
    if not query:
        return ""
    q = query.strip().rstrip(";").strip()
    q = re.sub(r"\s+", " ", q)
    return q.lower()


def _google_bleu_one(reference: str, hypothesis: str) -> float:
    """Google-BLEU as commonly used in Cypher/SQL eval."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError as e:
        raise ImportError("nltk is required for BLEU; pip install nltk") from e
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not hyp_tokens:
        return 0.0
    return sentence_bleu(
        [ref_tokens],
        hyp_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    )


def _generate_one(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Single-shot generation. Caller should batch for speed."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return completion


def evaluate(
    model,
    tokenizer,
    dataset,
    backend: CypherBackend | None = None,
    max_examples: int | None = None,
    prompt_template: str | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Evaluate model on a HF Dataset.

    Required dataset columns: question, schema, cypher.

    Returns a dict with overall and per-data_source breakdowns:
        {
            "overall": {"bleu": ..., "em": ..., "execution_accuracy": ..., "n": ...},
            "per_source": {<source>: {...}},
            "examples_logged": [<a few sample (gold, pred) pairs for sanity>],
        }
    """
    backend = backend or KuzuBackend()
    if prompt_template is None:
        prompt_template = (
            "You are a Cypher expert. Given the schema and question, return only the Cypher query.\n\n"
            "Schema:\n{schema}\n\n"
            "Question: {question}\n\n"
            "Cypher:\n"
        )

    n = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    bleus: list[float] = []
    ems: list[int] = []
    execs: list[int] = []
    by_source: dict[str, dict[str, list]] = {}
    samples: list[dict] = []

    for i in range(n):
        row = dataset[i]
        question = row["question"]
        schema_text = row["schema"]
        gold = row["cypher"]
        source = row.get("data_source", "unknown")

        prompt = prompt_template.format(schema=schema_text, question=question)
        try:
            completion = _generate_one(model, tokenizer, prompt)
        except Exception as e:
            logger.warning("gen failed at %d: %s", i, e)
            completion = ""

        pred = extract_cypher(completion)
        bleu = _google_bleu_one(gold, pred)
        em = int(_normalize_for_em(pred) == _normalize_for_em(gold))
        # execution accuracy reuses the same judge logic but only credits 1.0 for full match
        try:
            grade = _judge_one(pred, gold, schema_text, backend)
            exec_match = int(grade >= 1.0)
        except Exception as e:
            logger.debug("exec eval failed at %d: %s", i, e)
            exec_match = 0

        bleus.append(bleu)
        ems.append(em)
        execs.append(exec_match)
        by_source.setdefault(source, {"bleu": [], "em": [], "exec": []})
        by_source[source]["bleu"].append(bleu)
        by_source[source]["em"].append(em)
        by_source[source]["exec"].append(exec_match)

        if i < 5:
            samples.append({"question": question, "gold": gold, "pred": pred,
                            "bleu": bleu, "em": em, "exec": exec_match})

        if (i + 1) % 50 == 0:
            logger.info("eval progress %d/%d, bleu=%.3f em=%.3f exec=%.3f",
                        i + 1, n, _mean(bleus), _mean(ems), _mean(execs))

    result: dict[str, Any] = {
        "overall": {
            "bleu": _mean(bleus),
            "em": _mean(ems),
            "execution_accuracy": _mean(execs),
            "n": n,
        },
        "per_source": {
            s: {
                "bleu": _mean(v["bleu"]),
                "em": _mean(v["em"]),
                "execution_accuracy": _mean(v["exec"]),
                "n": len(v["bleu"]),
            } for s, v in by_source.items()
        },
        "examples_logged": samples,
    }
    if output_path is not None:
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result


def _mean(xs: list) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)
