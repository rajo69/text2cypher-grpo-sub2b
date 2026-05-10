# text2cypher-grpo — sub-2B GRPO Cypher specialist (work in progress)

Train a small (≤1.5B) language model to convert natural-language questions into executable
Neo4j Cypher queries, post-trained with **GRPO** (Group Relative Policy Optimization) on
[`neo4j/text2cypher-2024v1`](https://huggingface.co/datasets/neo4j/text2cypher-2024v1).
Designed to run on **free-tier compute** (Kaggle P100 / Colab T4) using
[Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl).

**Status:** scaffold complete; first training run has not been executed yet.

> What this is **not**: the first sub-2B Cypher model (sub-2B SFT models exist), nor the
> first GRPO-on-Cypher work ([STRuCT-LLM, Jun 2025](https://arxiv.org/abs/2506.21575) is the
> closest prior art). What it **is**: the first publicly released sub-2B GRPO Cypher specialist
> with reproducible eval and turnkey LangChain / Neo4j-MCP / Ollama integration.

For the full research and verification trail see [`report.md`](report.md). For the
implementation design see [`plan.md`](plan.md).

---

## Why this exists

GraphRAG developers using LangChain's `GraphCypherQAChain` or the Neo4j MCP server today
reach for GPT-4 / Claude to generate Cypher — fine for prototypes, painful in production
(latency, cost, data-residency / PII concerns for regulated industries that *cannot* send
schema + queries to a hosted API). A 1.5B-or-smaller specialist that runs locally via
Ollama / llama.cpp closes this gap.

Read [`report.md` §2](report.md) for the full who-uses-it / who-benefits / how-it-differs
breakdown.

---

## Repository layout

```
.
├── report.md                     # research + verification (verified URLs)
├── plan.md                       # detailed implementation plan
├── requirements.txt              # pinned deps
├── text2cypher_grpo/             # shared package
│   ├── schema.py                 # schema column → kuzu DDL
│   ├── cypher_graph.py           # Cypher → networkx property-graph (for R_GED)
│   ├── rewards.py                # R_judge + R_string + R_GED + length penalty
│   ├── db_backend.py             # KuzuBackend / FalkorDBLiteBackend
│   ├── eval_harness.py           # Google-BLEU + EM + Execution Accuracy
│   └── checkpoint.py             # HF Hub resume helpers
└── notebooks/
    ├── tier_a_gemma270m.py       # Tier A: Gemma-3-270m  (jupytext .py)
    └── tier_c_qwen15b.py         # Tier C: Qwen2.5-1.5B  (jupytext .py)
```

The notebooks are written as [jupytext](https://jupytext.readthedocs.io/) `.py` files
(percent format). Convert to `.ipynb` if needed:

```bash
pip install jupytext
jupytext --to notebook notebooks/tier_a_gemma270m.py
```

Or upload the `.py` directly to Kaggle as a script — both run identically.

---

## Tiered ladder strategy

We train the smallest model first (cheap to fail fast) and escalate only if reward signal
doesn't emerge.

| Tier | Base model | When to use | Decision gate |
|------|------------|-------------|---------------|
| **A** | `unsloth/gemma-3-270m-it` (0.27B) | Default — start here | After 150 steps: if `R_judge` mean < 0.15 OR composite reward not increasing → escalate |
| **B** | `Qwen2.5-Coder-0.5B-Instruct` (0.5B) | Fallback if Tier A stalls | Same logic |
| **C** | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (1.5B) | Safety net — almost guaranteed to learn | — |

Run **Tier A and Tier C in parallel** on separate Kaggle sessions for fastest path to
a working artifact.

---

## Run protocol

### Prerequisites

- HuggingFace account + write token. Set `HF_TOKEN` in Kaggle Secrets (recommended) or as an env var locally.
- Kaggle account, **phone-verified** (required for notebook internet access).
- Optional: GitHub repo (this repo) for the package — `git clone` into the notebook session.
  Alternative: zip `text2cypher_grpo/` + upload as a Kaggle Dataset called
  `text2cypher-grpo-pkg` and add to your notebook's inputs.

### Phase 0 — plumbing dry-run (~20 min, do this first)

In each notebook, change:

```python
CFG.max_steps = 10
CFG.save_steps = 5
CFG.eval_subset_size = 20
```

Run the notebook end-to-end. This verifies: dataset loads, schema parser yields a
non-trivial parse rate (cell 5 prints it — aim ≥ 70%), kuzu executes a Cypher query,
the four reward functions return non-zero values on gold queries, GRPO takes a step,
HF Hub push succeeds, eval prints reasonable numbers.

Then **kill the kernel and re-run** — the notebook should auto-resume from HF Hub.
If both steps pass, the plumbing is good.

### Phase 1 — Tier A real run (~6–9 hours on Kaggle P100)

Restore the original config (`max_steps=300`, `save_steps=50`, `eval_subset_size=200`).
Run on a Kaggle P100 session. Watch `R_judge` mean in the logs:

- If `R_judge` mean rises past ~0.3 by step 150 → great signal, let it run
- If still ~0.0–0.15 by step 150 → kill and escalate to Tier C

### Phase 1' — Tier C run in parallel (same ~6–9 hours)

Same compute budget, separate Kaggle session. No decision gate — Tier C is the safety net.

### Phase 2 — pick the best checkpoint, run full eval

Identify which tier (A, B, or C) produced the best in-loop eval scores. Uncomment the
"Phase 12 — full 4,830-row test eval" cell in that notebook and run. ~30–90 min.

Push the resulting `eval_results_full.json` to the model repo (the helper does this
automatically). Add a README to the HF model repo with the numbers, the eval script
hash, and a Mind-the-Query secondary eval.

### Phase 3 — ship

- Convert to GGUF via `llama.cpp`'s `convert_hf_to_gguf.py`, push the GGUF to HF Hub
- Write an Ollama Modelfile pointing at the GGUF
- Add a LangChain `GraphCypherQAChain` quickstart and a Neo4j MCP server config example
  to the model repo

---

## Where do the checkpoints live?

| Artifact | Location | Notes |
|---|---|---|
| LoRA adapters + optimizer + scheduler + RNG | HF Hub model repo: `<user>/text2cypher-grpo-{tier}` under `last-checkpoint/` | Pushed automatically every 50 steps via TRL's `hub_strategy="checkpoint"`. Source of truth. |
| Local mirror | `/kaggle/working/checkpoints/` | Faster within-session resume |
| Final merged model | Root of HF repo (`model.safetensors`) | End-of-training |
| Training logs / W&B | `<repo>/logs/` + W&B run if `WANDB_API_KEY` set | Auditable trace |
| `eval_results.json` | Root of HF repo | Pushed by `push_eval_results()` |

If a Kaggle session dies mid-step, worst-case loss = 50 steps. To resume: just re-launch
the notebook, the bootstrap cell pulls `last-checkpoint/` from HF Hub and
`trainer.train(resume_from_checkpoint=True)` continues from the saved step + optimizer state.

---

## Reward function

Composite, equal-weight (mirrors STRuCT-LLM's pattern):

| Component | What it measures | Implementation |
|---|---|---|
| `R_judge` | Execution validity (replaces STRuCT-LLM's o3-mini judge — we have no API on free tier) | Run the predicted Cypher on a synthesized kuzu graph: 0.0 = parse fail, 0.3 = runtime error, 0.7 = executes, 1.0 = result-set matches gold |
| `R_string` | Surface similarity to gold | `difflib.SequenceMatcher(...).ratio()` |
| `R_GED` | Structural similarity (independent of execution) | `networkx.algorithms.similarity.optimize_graph_edit_distance` on regex-extracted Cypher property-graphs |
| `R_length_penalty` | Anti length-hacking guardrail | Soft penalty for completions > 256 tokens |

See `text2cypher_grpo/rewards.py` for the verified TRL signature and the full implementation.

---

## Co-primary benchmarks

| Benchmark | Venue | Why included |
|---|---|---|
| [Neo4j Text2Cypher 2024v1](https://huggingface.co/datasets/neo4j/text2cypher-2024v1) | ACL 2025 GenAIK workshop ([paper](https://aclanthology.org/2025.genaik-1.11/)) | Vendor-official; used by STRuCT-LLM and Neo4j's own SFT models. Maximum comparability. **Primary.** |
| [Mind the Query](https://aclanthology.org/2025.emnlp-industry.133/) | EMNLP 2025 Industry Track | Stronger venue. Ships executable graphs (2024v1 only ships schemas). **Co-primary.** |

Targets at Tier C scale (Qwen2.5-1.5B):
- Google-BLEU ≥ 16 on 2024v1 test (STRuCT-LLM's verified Qwen2.5-1.5B Both-joint = 15.8 ± 2.2)
- First sub-2B model with reproducible Execution Accuracy on this split

---

## Known caveats (read before running)

1. **Unsloth + Gemma-3-270m + GRPO** is an untested combination — Unsloth ships the model and supports GRPO, but no public notebook glues them together yet. If `FastLanguageModel` errors on 270m, fall back to raw `transformers` + TRL `GRPOTrainer`.
2. **Schema parser is heuristic** (the `schema` column has both markdown and APOC JSON formats — the parser handles both, but expect 5–15% of rows to yield trivial DDL). Cell 5 of each notebook prints the parse rate.
3. **kuzu is archived** (Apple acquired Kùzu Inc on 2025-10-10). The 0.11.3 wheel still installs; we wrap it in a backend abstraction so swapping to FalkorDBLite (actively maintained, RedisGraph-team) is one line — but FalkorDBLite needs Python ≥ 3.12, which Kaggle may or may not provide.
4. **STRuCT-LLM's `R_judge` used o3-mini API**, which we don't have on free tier. Our replacement is execution-based grading (0.0/0.3/0.7/1.0). This is a deviation worth documenting in any release.
5. **`vllm_gpu_memory_utilization`** needs empirical tuning per Kaggle GPU. Defaults are conservative guesses.

See [`plan.md` §11](plan.md) for the full risk register.

---

## Acknowledgements / honest prior-art map

- **STRuCT-LLM** (Bouvier et al., [arXiv 2506.21575](https://arxiv.org/abs/2506.21575)) — closest prior art. Joint SQL+Cypher GRPO. Reward design (`R_judge + R_string + R_GED`) and 1.5B Cypher numbers come from their paper. Code: [github.com/bouv/STRuCT-LLM](https://github.com/bouv/STRuCT-LLM).
- **Ozsoy et al. 2024** ([arXiv 2412.10064](https://arxiv.org/abs/2412.10064), Neo4j) — defined the canonical Text2Cypher eval (Google-BLEU + Exact Match) and curated the dataset.
- **Tran et al. 2025** (MDPI Applied Sciences 15(15) 8206) — first GRPO-on-Cypher paper at small scale (Qwen2.5-3B). +5.03% execution-accuracy delta over SFT.
- **Unsloth** (Daniel + Michael Han) — the free-tier GRPO recipe everyone in the small-model RL space leans on.
- **Existing sub-2B Cypher SFT baselines** that we measure against: [`dbands/Qwen2-5-Coder-0-5B_neo4j-text2cypher-2024v1-16B`](https://huggingface.co/dbands/Qwen2-5-Coder-0-5B_neo4j-text2cypher-2024v1-16B), [`VoErik/cypher-gemma`](https://huggingface.co/VoErik/cypher-gemma).

---

## License

Apache-2.0 — matches the dataset license. See [LICENSE](LICENSE).
