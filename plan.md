# Implementation Plan — Text2Cypher GRPO Sub-2B Specialist

**Companion to:** `report.md` (research/verification) and the `text2cypher_grpo/` package + `notebooks/`.
**Compiled:** 2026-05-10. Verified APIs, current as of TRL v1.4.0, Unsloth v0.1.39-beta, kuzu 0.11.3, FalkorDBLite 0.10.0.

---

## 1. Architecture overview

Two parallel training tracks share a single Python package `text2cypher_grpo/` that owns all non-trivial logic. Each notebook is a thin orchestration layer over the package.

```
RL_exp/
├── report.md                         # research + verification
├── plan.md                           # this file
├── requirements.txt                  # pinned deps for both notebooks
├── text2cypher_grpo/                 # shared package (run on Kaggle either via git clone or Kaggle Dataset upload)
│   ├── __init__.py
│   ├── schema.py                     # schema column parser → kuzu DDL (handles Format A markdown + Format B APOC JSON)
│   ├── cypher_graph.py               # Cypher query → networkx property-graph (regex-based, no parser dep)
│   ├── rewards.py                    # R_judge (execution-grade) + R_string (SequenceMatcher) + R_GED (networkx)
│   ├── db_backend.py                 # backend ABC; kuzu primary, FalkorDBLite Kaggle-fallback
│   ├── eval_harness.py               # Google-BLEU + EM + execution accuracy on test split
│   └── checkpoint.py                 # HF Hub resume helpers
└── notebooks/
    ├── tier_a_gemma270m.py           # Gemma-3-270m-it, full-precision LoRA  (jupytext .py — convert to .ipynb if needed)
    └── tier_c_qwen15b.py             # Qwen2.5-1.5B-Instruct, QLoRA 4-bit
```

**Why a package, not inline notebook code:** the reward function and schema parser are non-trivial; bugs there silently corrupt training. Owning them in a tested package keeps the two notebooks short and ensures both tiers get identical reward semantics. Kaggle ingests the package via `git clone` at startup (if you push to GitHub) or by uploading the folder as a Kaggle Dataset.

---

## 2. Tiered ladder — concrete decision gate

| Tier | Model | Why try | Decision gate to escalate |
|---|---|---|---|
| **A** | `unsloth/gemma-3-270m-it` (0.27B, BF16 full-precision LoRA) | Cheapest iteration. Matches `VoErik/cypher-gemma` base — apples-to-apples vs. existing sub-1B SFT baseline. If it works, the deployable artifact is *much* more impressive (CPU-runnable). | After **150 steps** (~3 hr Kaggle P100): if `R_judge` mean is < 0.15 OR composite reward has not increased over rollout-0 baseline, escalate. |
| **B** | `Qwen2.5-Coder-0.5B-Instruct` (0.5B, QLoRA 4-bit) | Fallback. Same family as `dbands` baseline. | Same gate logic as Tier A, applied to Tier B's run. |
| **C** | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (1.5B, QLoRA 4-bit) | Original safe choice. Matches STRuCT-LLM scale for direct-comparison BLEU. | If neither A nor B shows reward signal, treat C as the canonical run — it is almost guaranteed to learn. |

**Run Tier A and Tier C in parallel from day 1** (separate Kaggle accounts or alternating sessions on one account given 30h/week quota). Tier B is held in reserve and only run if Tier A fails *and* you want to debug the size-vs-task interaction before paying Tier C's compute.

---

## 3. Notebook structure (identical for both tiers; only model + LoRA config differ)

Every notebook has these cells in order:

| # | Cell | Purpose |
|---|------|---------|
| 1 | Markdown header | Tier identifier, dataset, expected outcomes, last-known-good Kaggle environment versions |
| 2 | Setup | `pip install`s with pinned versions; `import sys; print(sys.version)` for FalkorDBLite Python ≥ 3.12 check |
| 3 | Auth | HF token from Kaggle Secrets; W&B optional |
| 4 | Config | All hyperparameters in one dataclass — easy to diff between tiers |
| 5 | Repo bootstrap | `git clone` the package OR `pip install -e /kaggle/input/text2cypher-grpo-pkg`  |
| 6 | Dataset load | `load_dataset("neo4j/text2cypher-2024v1")` + filter rows whose `schema` parses cleanly |
| 7 | Backend init | Choose kuzu by default; FalkorDBLite if `--backend falkordb` arg |
| 8 | Reward setup | Compose R_judge + R_string + R_GED with weights = (1,1,1) |
| 9 | Model load | Unsloth `FastLanguageModel.from_pretrained` (Tier A: 270m-it; Tier C: Qwen-1.5B 4-bit) |
| 10 | LoRA setup | `FastLanguageModel.get_peft_model` with target modules per architecture |
| 11 | Resume check | Pull `last-checkpoint` from HF Hub if present |
| 12 | Trainer | `GRPOTrainer(model, reward_funcs=[r_judge, r_string, r_ged], args=GRPOConfig(...))` |
| 13 | Train | `trainer.train(resume_from_checkpoint=True)` if checkpoint exists, else fresh |
| 14 | Eval | Run `eval_harness.evaluate()` on test split, log Google-BLEU + EM + execution accuracy |
| 15 | Push final | `trainer.push_to_hub` + write README with claimed numbers + eval JSON |

---

## 4. Schema → kuzu DDL conversion

The dataset's `schema` column is **heterogeneous** (verified). Two formats observed:

- **Format A (markdown)**: `Node properties: - **Product** - \`productName\`: STRING ... Relationships: (:Topic)-[:HAS_TOPIC]->(:Article) ...`
- **Format B (APOC JSON)**: `{"ASSIGNED_TO": {"count": 27, "type": "relationship"}, "Machine": {"count": 9, "labels": [], "properties": {...}}}`

Strategy in `schema.py`:
1. **Try `json.loads()` first** — if it parses, treat as Format B.
2. **Otherwise apply markdown heuristics** — regex-extract node `**Label**` blocks, property `\`name\`: TYPE` lines, and relationship `(:A)-[:REL]->(:B)` patterns.
3. **Synthesize kuzu DDL** with these rules:
   - For each label, emit `CREATE NODE TABLE Label(id INT64, <props>, PRIMARY KEY(id))` — synthetic `id` because most schemas don't declare one and kuzu requires PK.
   - Map types: STRING→STRING, INTEGER→INT64, FLOAT→DOUBLE, BOOLEAN→BOOL, DATE→DATE, default unknown→STRING.
   - For each relationship `(:A)-[:REL]->(:B)`, emit `CREATE REL TABLE REL(FROM A TO B, <props>)`.
   - When endpoints are ambiguous (Format B's relationships often lack typed endpoints), emit `CREATE REL TABLE REL(FROM Any TO Any)` and add a synthetic `Any` node table at the start.
4. **Track parse-failure rate** — log it; aim for ≥ 80% successful schemas in the test split (verification estimate: 85–95% should parse).
5. **Cache compiled DDL per `database_reference_alias`** — the dataset has 543 unique schemas; building DDL once per schema is a 543x speedup.
6. **Always wrap DDL execution in try/except** — kuzu is strict; one bad type assignment shouldn't crash a rollout. Fallback: skip the row from the batch if DDL fails.

---

## 5. Cypher → property-graph for R_GED

`cypher_graph.py` exposes `parse_cypher_to_graph(query: str) -> networkx.MultiDiGraph`.

Strategy: regex-based extraction, NO parser dependency. The text2cypher-2024v1 queries are dominated by simple patterns:
- Node pattern: `\((\w+)?:?(\w+)?\s*({[^}]*})?\)`
- Rel pattern: `\(...\)-\[(\w+)?:?(\w+)?\s*({[^}]*})?\]-?\>?\(...\)`

Algorithm:
1. Strip string literals first (`'...'`, `"..."`) so they don't confuse the regex.
2. Scan all `MATCH`/`CREATE`/`MERGE` clause bodies with a state machine that tracks alternating `(node)-[rel]-(node)` patterns.
3. For each pattern, emit `nx.MultiDiGraph` nodes labeled by `:Label` and edges typed by `:REL`. Properties become attribute dicts.
4. Variable names (`a`, `b`) are tracked so that `MATCH (a:Person)-[:KNOWS]->(b:Person)` followed by `WHERE a.age > 30` correctly attaches the property to the right node.
5. Fail-soft: if no patterns extract, return an empty graph (R_GED will then be 0).

`reward_ged(pred_query, gold_query) -> float`:
- Build both graphs.
- If either is empty: return 0.
- Use `nx.algorithms.similarity.optimize_graph_edit_distance(g1, g2, timeout=2.0)` — generator yielding approximations.
- Take the *first* approximation (best lower bound) for speed.
- Return `max(0, 1 - ged / max(size_pred, size_gold))` where `size = #nodes + #edges`.
- Per-rollout timeout of 2 s prevents pathological worst-case.

---

## 6. Reward function design

| Component | Formula | Range | Notes |
|---|---|---|---|
| `R_judge` | Execution grade (replaces STRuCT-LLM's o3-mini judge — we have no API on free tier) | 0–1 | 0.0 = Cypher fails to parse on backend; 0.3 = parses but runtime error (missing label/property); 0.7 = executes, returns rows; 1.0 = executes AND result-set matches gold |
| `R_string` | `difflib.SequenceMatcher(None, gold, pred).ratio()` | 0–1 | Continuous, smooth gradient even when execution fails |
| `R_GED` | `1 - GED(g_pred, g_gold) / max(size_pred, size_gold)` via networkx with 2 s timeout | 0–1 | Structural match independent of execution |
| **Composite** | `R = R_judge + R_string + R_GED` | 0–3 | Equal weights, untuned, mirroring STRuCT-LLM |

**Optional auxiliary reward** (Phase 2, only if Tier A baseline doesn't budge): MDPI-style "support task" reward — the model must emit `<extracted_kv>...</extracted_kv>` blocks listing key-value pairs and `<triples>...</triples>` listing (s,p,o) extracted from the question, before the Cypher. Reward bonus for valid extraction. Defer until proven needed.

**TRL signature** (verified):
```python
def reward_judge(prompts, completions, completion_ids, trainer_state, cypher, schema, **kwargs):
    # cypher and schema are auto-passed because they are dataset columns
    return [judge_one(c, gold, sch) for c, gold, sch in zip(completions, cypher, schema)]
```

**Length penalty (guardrail):** `R_length = -0.1 * max(0, len(tokens) - 256) / 256` to prevent length-hacking. Applied as a fourth reward function with weight = 1 (TRL sums all rewards). Track output-length distribution every eval — if it drifts up, increase penalty weight.

---

## 7. Backend abstraction (kuzu primary, FalkorDBLite fallback)

`db_backend.py`:

```python
class CypherBackend(ABC):
    @abstractmethod
    def execute(self, schema_ddl: str, query: str, timeout_ms: int = 1000) -> ExecutionResult: ...

class KuzuBackend(CypherBackend):  # default; works on all platforms
    ...

class FalkorDBLiteBackend(CypherBackend):  # Kaggle/Linux only, requires Python 3.12+
    ...
```

`ExecutionResult` is a small dataclass: `{success: bool, rows: list | None, error: str | None, error_kind: Literal["parse", "runtime", "timeout", None]}`.

Selection:
- Tier A and Tier C default to **kuzu** (works on user's Windows local + Kaggle Linux).
- If Kaggle's Python ≥ 3.12, FalkorDBLite is available as `--backend falkordb` for cross-checking results when kuzu's stricter type system rejects an otherwise-valid query.
- **Never use Aura** for training — the 25 req/min limit (verified) kills GRPO rollouts.

---

## 8. Checkpointing & resume — concrete strategy

### 8.1 Where everything lives

| Artifact | Location | Rationale |
|---|---|---|
| **LoRA adapter checkpoints** | HuggingFace Hub model repo: `<USERNAME>/text2cypher-grpo-gemma-270m` (Tier A) and `<USERNAME>/text2cypher-grpo-qwen-1p5b` (Tier C) | Source of truth. Survives any Kaggle session death. |
| Local mirror | `/kaggle/working/checkpoints/` | Faster resume within a single session; cleared when a fresh notebook session starts unless the notebook is committed |
| Optimizer state, RNG, scheduler | Inside the same HF Hub `last-checkpoint` subfolder (TRL bundles them) | TRL's `Trainer` writes `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json` to the checkpoint dir; `hub_strategy="checkpoint"` mirrors them |
| Training logs / eval results | `<repo>/logs/` (committed alongside model) + W&B run if token provided | Auditable trace |
| Final merged model | Pushed at end of training to root of repo as `model.safetensors` | End-of-training artifact for downstream Ollama / GGUF conversion |

### 8.2 The exact TRL config (verified)

```python
GRPOConfig(
    output_dir="/kaggle/working/checkpoints",
    push_to_hub=True,
    hub_model_id=f"{HF_USERNAME}/text2cypher-grpo-{tier_id}",
    hub_strategy="checkpoint",       # CRITICAL — pushes `last-checkpoint` subfolder
    save_strategy="steps",
    save_steps=50,                   # one push every 50 GRPO steps (~30 min on Kaggle)
    save_total_limit=2,              # keep last 2 checkpoints locally; HF retains all revisions
    save_safetensors=True,
    hub_token=os.environ["HF_TOKEN"],
    hub_private_repo=False,           # public so HF social proof is real; flip to True if shy
    ...
)
```

### 8.3 Resume logic (top of notebook)

```python
from huggingface_hub import snapshot_download, repo_exists

repo_id = f"{HF_USERNAME}/text2cypher-grpo-{tier_id}"
local_ckpt = "/kaggle/working/checkpoints/last-checkpoint"

if repo_exists(repo_id, token=HF_TOKEN):
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_ckpt,
        allow_patterns=["last-checkpoint/*"],
        token=HF_TOKEN,
    )
    resume_arg = local_ckpt if os.path.isdir(local_ckpt) else True
else:
    resume_arg = None

trainer.train(resume_from_checkpoint=resume_arg)
```

### 8.4 Disaster-recovery contract

- **Session dies mid-step**: lose ≤ 50 steps of progress (since last save). Push the last-known-good local checkpoint manually if the auto-push didn't fire.
- **HF push fails (network)**: TRL retries; if all retries fail the local checkpoint at `/kaggle/working/checkpoints/last-checkpoint` is still intact for the rest of the session. Push it manually with `huggingface_cli upload <repo> /kaggle/working/checkpoints/last-checkpoint last-checkpoint`.
- **Kaggle session storage cleared between sessions**: irrelevant — HF Hub is the source of truth. Resume always pulls from there.
- **HF Hub repo deleted accidentally**: run a small daily mirror to a Kaggle Dataset as belt-and-braces (optional; documented but not in default path).

---

## 9. Eval harness

`eval_harness.py` exposes `evaluate(model, tokenizer, dataset, backend, max_examples=None) -> dict`:

| Metric | Implementation | Source |
|---|---|---|
| Google-BLEU | `nltk.translate.bleu_score.sentence_bleu` with `SmoothingFunction().method1` and uniform weights, then averaged. Matches Ozsoy 2024. | NLTK |
| Exact Match | Whitespace-normalized string compare (collapse runs, strip trailing `;`, lowercase keywords) | Custom |
| Execution Accuracy | Synthesize a kuzu graph from `schema` (same DDL converter), execute predicted query, compare result set (as `set(tuple(row))`) to gold. Tolerate row order. | kuzu via `db_backend.py` |

Run on full 4,830-row test split (or `max_examples` for quick checks). Report per-`data_source` breakdown so we see which source datasets the model handles best/worst.

For Mind-the-Query secondary eval: same harness, plug in their dataset loader, use their executable graphs directly (no schema-synthesis needed).

---

## 10. Run protocol — order of operations

### Day 1 (~3 hours): plumbing dry-run

1. Create HF Hub repos (one per tier).
2. Run notebook A (Tier A) end-to-end with `max_steps=10`, `save_steps=5`, `max_examples=20` for eval — verify the entire stack works on Kaggle: schema parsing, kuzu execution, reward computation, GRPO step, HF Hub push, eval harness.
3. Run notebook C (Tier C) the same way.
4. Verify resume by killing the kernel and restarting — both notebooks should auto-resume from HF Hub.
5. Goal: zero training value, full plumbing confidence.

### Day 2 (~9 hours each, parallel sessions): Tier A real run

1. `max_steps=300`, `save_steps=50`, eval every 100 steps on a 200-row eval subset.
2. Decision gate at step 150: if `R_judge` mean < 0.15 or composite reward not increasing → kill and escalate.
3. If learning: continue to 300 steps, run full eval at end.

### Day 2 (parallel): Tier C real run

1. `max_steps=300`, `save_steps=50`, eval every 100 steps on a 200-row eval subset.
2. No decision gate — Tier C is the safety net; let it run.

### Day 3 (~6 hours): consolidate

1. Pick the best-performing tier checkpoint.
2. Run final eval on full 4,830-row test split + Mind-the-Query.
3. Convert to GGUF via `llama.cpp`, push to HF Hub.
4. Write Ollama Modelfile + LangChain `GraphCypherQAChain` + Neo4j MCP example notebooks for the README.
5. Compute SFT-vs-GRPO ablation (one extra Kaggle session: SFT-only run on same data, same compute).

### Day 4: announce

1. Updated README with reproducible numbers and `eval_results.json`.
2. Blog post / X thread / Reddit r/LocalLLaMA — frame as "first sub-2B GRPO Cypher specialist with GGUF + LangChain/MCP integration."

---

## 11. Risks specific to implementation (in addition to project-level risks in `report.md` §7)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Schema parser fails for > 30% of test rows → unable to compute execution-accuracy on enough data | Medium | Track parse rate from cell 6; if < 70%, narrow eval to rows that parse; report this as a known limitation |
| networkx GED is too slow even with `optimize_graph_edit_distance` + 2 s timeout → rollouts hang | Medium | Use first approximation only; cap at 20-node graphs (skip rewards for queries with bigger subgraphs); benchmark on 10 random rollouts before training |
| kuzu rejects valid queries due to typed-schema-mismatch (Neo4j is schemaless, kuzu is typed) | High at first | Start with permissive types (everything STRING); only sharpen if reward signal stalls |
| Unsloth Gemma-3-270m + GRPO is untested combination — may hit a model-architecture-specific bug | Medium | Tier A is the cheap experiment by design; if GRPOTrainer crashes on 270m, fall back to raw `transformers` + TRL (one-line model swap) |
| FalkorDBLite Python ≥ 3.12 requirement breaks Kaggle default → fallback path is fictitious | Low-Medium | First cell of every notebook prints `sys.version`; if Kaggle is < 3.12, document FalkorDBLite as a future-Python-only option and run kuzu only |
| HF Hub push hits 1 GB single-file limit on a checkpoint | Low | LoRA adapters are ~30-100 MB; only the optimizer state could grow. With 4-bit base + LoRA, optimizer state stays well under 500 MB |
| GRPOConfig defaults changed in TRL ≥ 1.4 (`beta=0.0`, `learning_rate=1e-6`) — silent regression vs older notebooks | Documented | Pin TRL == 1.4.0 in `requirements.txt`; always pass `beta` and `learning_rate` explicitly so we never rely on the default |
| Kaggle session disconnect during a 9-hour run | High (frequent) | `hub_strategy="checkpoint"` plus `save_steps=50` means worst case = 50 steps lost. Also enable W&B if available — the run resumes against the same W&B run ID |

---

## 12. What NOT to do (lessons from research)

- **Don't use Aura** for rollouts (25 req/min — verified; would take days for one epoch).
- **Don't use a third-party Cypher parser** — they're abandoned or have wrong API surfaces. Regex extraction is more reliable here.
- **Don't claim "first GRPO Cypher model"** — STRuCT-LLM published Jun 2025. Claim "first publicly released sub-2B GRPO Cypher specialist with GGUF + LangChain/MCP integration."
- **Don't claim "empty niche"** — sub-2B SFT Cypher exists (dbands, VoErik). Empty *sub-niche* is GRPO + sub-2B + deployable integrations.
- **Don't claim numbers from `lakkeo`'s card** as 2024v1 baselines — verified that those are on its own synthetic 5k held-out, not the canonical test split.
- **Don't pre-SFT before GRPO at small scale** unless escalation needs it — the MDPI delta of +5.03% comes from SFT+GRPO over SFT-alone, but at 270m/500m scale the SFT step may eat most of your compute budget.
- **Don't push to HF Hub from a Kaggle notebook without enabling Internet** — phone-verified Kaggle account required; document this in the auth cell.

---

## 13. Open implementation questions (to resolve while running, not before)

1. **GED timeout vs reward smoothness** — 2 s per rollout × G=4 generations × batch 1 × steps 300 = ~40 minutes wall-clock just on GED if every call hits the timeout. May need to drop GED for outsized graphs and rely on R_judge + R_string only for those cases. Benchmark on first 50 steps and tune.
2. **`vllm_gpu_memory_utilization` on Kaggle GPUs** — TRL recommends 0.55–0.7 on T4; P100 is similar VRAM but different compute; tune empirically and log the chosen value.
3. **Tier C: full QLoRA target modules (`q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`) vs attention-only** — start with all 7; if VRAM tight, drop the MLP projections.
4. **`num_generations` (G)** — STRuCT-LLM used 6; Unsloth's free-T4 examples use 4. Start at 4 to fit, raise to 6 if Kaggle's P100 has headroom.
5. **Whether to enable `vllm_enable_sleep_mode=True`** — recommended for OOM avoidance but adds latency between train and rollout phases. Off by default; enable if OOM hits.

---

## 14. What the user owns (cannot be done for them in this session)

- HF Hub account with write token in Kaggle Secrets (env: `HF_TOKEN`).
- Kaggle account with phone verification (required for notebook internet access).
- Optional: GitHub repo to host the `text2cypher_grpo/` package + notebooks (alternative: zip and upload as Kaggle Dataset).
- Optional: W&B account (set `WANDB_API_KEY` to enable; otherwise logging is local-only).
- Cloning STRuCT-LLM (`git clone https://github.com/bouv/STRuCT-LLM`) and reading their `training/` scripts to confirm our R_GED parser produces graphs comparable to theirs. Recommended but not blocking.

---

## Bottom line

The two notebooks are intentionally short (~250 lines each) because all real logic lives in the `text2cypher_grpo/` package. This means: when Tier A succeeds (or fails), iterating on reward / schema / eval logic happens once in the package, both notebooks pick it up, and the comparison stays apples-to-apples. Checkpointing is HF-Hub-first via `hub_strategy="checkpoint"`, with `save_steps=50` giving a worst-case loss of 50 steps if Kaggle disconnects. Both notebooks resume automatically by pulling `last-checkpoint` from HF Hub at startup.
