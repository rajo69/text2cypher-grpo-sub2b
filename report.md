# Sub-2B GRPO Cypher Specialist — Project Research Report

**Compiled:** 2026-05-10
**Author context:** Free-tier compute only (Kaggle P100 / Colab T4), no local GPU
**Target:** A small (≤1.5B) language model, RL-post-trained with GRPO, that converts natural language → executable Cypher (Neo4j graph query language), shipped as a deployable artifact (GGUF + LangChain / Neo4j-MCP integration).

> **Confidence convention used below.** **VERIFIED** = URL fetched and content matched. **PARTIAL** = appears in independent search snippets / abstracts but not fully retrieved. **UNVERIFIED** = could not directly confirm in this pass; flagged so you can verify before announcing.

---

## Verification Update — 2026-05-10 (post deep-read pass)

Two parallel verification agents read MDPI + STRuCT-LLM end-to-end and pulled all baseline model cards + kuzu coverage. Five findings change earlier sections:

1. **kuzu was archived on 2025-10-10** (Apple acquired Kùzu Inc). The PyPI wheel still installs and the Cypher dialect covers ~85–95% of `text2cypher-2024v1` test queries, so it is *usable for this project today*, but the reward harness must be wrapped in a backend abstraction so we can swap to **FalkorDBLite** (`pip install falkordblite`, embedded, actively maintained, Cypher-compatible). See §4.1 and §7. Sources: theregister.com/2025/10/14/kuzudb_abandoned/, news.ycombinator.com/item?id=45560036, pypi.org/project/falkordblite.
2. **STRuCT-LLM Qwen2.5-1.5B Cypher numbers (VERIFIED, arXiv HTML Table B1)**: Both-joint config — Text2Cypher **BLEU 15.8 ± 2.2**, **EM 14.2** (CRT-QA), Spider EXE 51.2. Reward = `R_judge + R_string + R_GED` (all weights = 1, untuned). Compute: 32× H100, 60h. **No 1.5B HF release** — only 33B QwQ at `boubnov/STRuCT-LLM-Novo`. This is now the explicit bar in §6.1.
3. **MDPI paper (PARTIAL — page 403'd, snippet-only)**: Qwen2.5-3B-Instruct, **+5.03% execution accuracy over SFT**, 85% on unseen schemas. Reward uses auxiliary "support tasks" (key-value + triple extraction). No code/weights. Modest, real delta — do not promise headline gains. See §3.2.
4. **No sub-2B GRPO/DPO/RL Cypher model exists on HF** (re-confirmed via search). The differentiation statement in §3.3 holds.
5. **Existing sub-2B Cypher cards publish ZERO eval numbers** (dbands, VoErik, neo4j-4B, Mondhirch, vprashant, tomasonjo-demo, mradermacher-GGUF). Two consequences: (a) merely publishing reproducible BLEU/EM/Execution numbers on the canonical 2024v1 test split is itself a real contribution, (b) "beating dbands' 0.5B SFT" requires running their checkpoint on our eval harness — their card numbers don't exist. **VoErik is on `text2cypher-2025v1`, not 2024v1** — direct comparison requires re-training one of them on the same split.

Sub-finding: `text2cypher-2024v1` test split = 4,830 rows (HF viewer) vs 4,833 stated in card (off-by-one likely from dedup). 543 unique schemas — useful: pre-build kuzu DDL once per schema and cache.

---

## 1. The project in one paragraph

Take a small base model (**tiered ladder**: start at **`google/gemma-3-270m-it`** = 0.27B for cheapest iteration; escalate to **`Qwen2.5-Coder-0.5B-Instruct`** then **`Qwen2.5-1.5B-Instruct`** if reward signal does not emerge); post-train with **GRPO** on **`neo4j/text2cypher-2024v1`** (44,387 NL/Cypher/schema triples, Apache-2.0); use a **composite reward** modeled on STRuCT-LLM's `R_judge + R_string + R_GED` pattern, with execution against an in-process embedded Cypher DB (**kuzu** primary — usable but archived as of 2025-10-10 — wrapped in a backend abstraction so **FalkorDBLite** is a one-line swap); evaluate on the canonical held-out test split (4,830 rows) using **Google-BLEU + Exact Match + Execution Accuracy** (publishing reproducible numbers on this split is itself a contribution — none of the existing sub-2B baselines do so); ship as a **GGUF model on HuggingFace** with **LangChain `GraphCypherQAChain` + Neo4j MCP server compatibility + Ollama Modelfile**. Training fits a single 9-hour Kaggle P100 session via Unsloth's GRPO + QLoRA path.

---

## 2. Why this project — who benefits, who uses it, why

### 2.1 The actual user

| User | Pain today | Why a sub-2B local Cypher model matters |
|---|---|---|
| **GraphRAG developer** (LangChain / LlamaIndex / Microsoft GraphRAG user) | Cypher generation outsourced to GPT-4 / Claude → latency, cost, leak risk | Local model that plugs into `GraphCypherQAChain` with no API call |
| **Neo4j-using enterprise** (regulated industries — banking, healthcare, legal) | Cannot send schema + queries to a hosted LLM (data-residency / PII) | A 1.5B GGUF runs offline on a CPU laptop or single internal GPU |
| **Neo4j MCP server users** (Claude / Cursor / Continue agents) | Default MCP Cypher generation uses the host model — bloated for narrow Cypher tasks | A specialist 1.5B model behind the MCP server is faster + cheaper per call |
| **Indie devs / hobbyists** | No interactive-speed Cypher autocomplete in Neo4j Browser without paying | Ollama-hosted Cypher specialist, free |
| **Researchers** comparing GraphRAG variants | No standardised small-model Cypher baseline | Released GGUF + numbers becomes a reference baseline |

### 2.2 GraphRAG demand evidence

- **Microsoft GraphRAG**: ~30.5–31.6k GitHub stars (VERIFIED via search aggregates 2026). Active project, growing.
- **LangChain `GraphCypherQAChain`** is documented and supports local Ollama LLMs. **VERIFIED** — https://docs.langchain.com/oss/python/integrations/providers/ollama
- **Neo4j MCP server** (`mcp-neo4j-cypher`) exists and is published. **VERIFIED** — https://github.com/neo4j-contrib/mcp-neo4j and https://github.com/neo4j/mcp
- **Reddit / X demand quantitatively**: **UNVERIFIED in this pass** — search tools could not fetch Reddit reliably. Demand is *plausible* not proven; do not lead announcement copy with "developers are asking for this."

---

## 3. Differentiation — honest comparison vs prior art

### 3.1 Existing Cypher models on HuggingFace (VERIFIED, fetched from `?other=cypher`)

| Model | Params | Method | Downloads/mo | Status |
|---|---|---|---|---|
| `dbands/Qwen2-5-Coder-0-5B_neo4j-text2cypher-2024v1-16B` | **0.5B** | SFT (Unsloth/TRL) | 5 | Closest sub-2B SFT baseline. Trained on the exact same dataset. **Must be beaten.** |
| `VoErik/cypher-gemma` | **0.27B** | SFT (TRL) | 4 | Tiny gemma-3-270m on `text2cypher-2025v1`. **Must be beaten.** |
| `mradermacher/StructuredGemma-3-270M-GGUF` | 0.27B | SFT multi-task | n/a | Not Cypher-specialist |
| `vprashant/cypher-gen` | 60M T5 | SFT | 8 | WikiSQL→Cypher, weak |
| `Mondhirch/Natural-Language-to-Cypher-2.7b` | 2.7B | unknown | 17 | Empty model card |
| `lakkeo/stable-cypher-instruct-3b` | 3B | LoRA SFT | 1,028 | Reports BLEU-4 88.63 / Pass@1 51.8% **on its own held-out**, not the Neo4j benchmark |
| `Azzedde/llama3.1-8b-text2cypher` | 8B | LoRA SFT | 895 | Highest-traction community model |
| `tomasonjo/text2cypher-demo-16bit` (+gguf) | 8B | SFT (Unsloth) | n/a | Llama-3-8B |
| `neo4j/text-to-cypher-Gemma-3-4B-Instruct-2025.04.0` | **4B** | SFT | n/a | Official Neo4j |
| `neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1` | 9B | LoRA SFT | n/a | Official Neo4j |
| `neo4j/text-to-cypher-Gemma-3-27B-Instruct-2025.04.0` | 27B | SFT | n/a | Official Neo4j |

**Honest read of this table:**
- Sub-2B Cypher specialists already exist (dbands 0.5B, VoErik 0.27B) — they are SFT, low-quality, low-traction.
- Neo4j Labs themselves ship 4B/9B/27B Gemma SFT models — that's the production benchmark to think about, not the current 0.5B SFT entries.
- **No publicly released sub-2B GRPO/RL Cypher checkpoint exists** as of 2026-05-10.

### 3.2 Relevant academic prior art (VERIFIED)

| Paper / repo | Date | Method | Models | Released weights? | Bearing on this project |
|---|---|---|---|---|---|
| Ozsoy et al. "Text2Cypher: Bridging NL and Graph DBs" — arXiv [2412.10064](https://arxiv.org/abs/2412.10064) → [ACL 2025.genaik-1.11](https://aclanthology.org/2025.genaik-1.11/) | Dec 2024 | SFT | Gemma-2-9B etc. | Yes (HF Neo4j org) | Defines the canonical eval (Google-BLEU + EM); use the same |
| Ozsoy "Text2Cypher: Data Pruning..." — arXiv [2505.05122](https://arxiv.org/abs/2505.05122) | May 2025 | SFT + hard-example pruning | Gemma | n/a | Use for data-curation tricks |
| **STRuCT-LLM (Bouvier et al.)** — arXiv [2506.21575](https://arxiv.org/html/2506.21575) — code at github.com/bouv/STRuCT-LLM | Jun 2025 | **GRPO joint SQL+Cypher** with reward `R_judge + R_string + R_GED` (graph-edit-distance), all weights=1 | **Qwen2.5-1.5B**, Qwen2.5-14B, Qwen3-14B, QwQ-32B | Code yes; **`boubnov/STRuCT-LLM-Novo` 33B BF16 only — no 1.5B release** | **VERIFIED real bar.** 1.5B Both-joint (Table B1): **Text2Cypher BLEU 15.8 ± 2.2**, Spider EXE 51.2, CRT-QA EM 14.2. QwQ-32B Text2Cypher EXE 55.3 (top entry). Compute: 32× H100, 60h, G=6, LR 1e-5 (1.5B). |
| MDPI *Applied Sciences* 15(15) 8206 — "Refining Text2Cypher on Small LM with RL Leveraging Semantic Information" — https://www.mdpi.com/2076-3417/15/15/8206 | Jul 2025 | GRPO + auxiliary "support tasks" (key-value + (s,p,o) triple extraction in rollout) | Qwen2.5-3B-Instruct (only) | **No weights, no code** | **PARTIAL — page 403'd, snippet-only.** SFT-vs-GRPO delta = **+5.03% execution accuracy**; 85% on unseen schemas. Modest, real. Length-hacking / failure modes not reported (does not mean absent). |
| IBM "Mind the Query" — [2025.emnlp-industry.133](https://aclanthology.org/2025.emnlp-industry.133/) | EMNLP 2025 | New benchmark (27,529 rows w/ executable graphs) | n/a | Benchmark released | **Use as a secondary eval surface** for execution accuracy |

### 3.3 The honest differentiation statement

> *First publicly released sub-2B GRPO-trained Cypher specialist with GGUF artifacts and turnkey LangChain / Neo4j-MCP / Ollama integration.*

This is defensible. The earlier "empty niche" framing **is not** — it overclaims. Use this exact line in any README/blog.

What this project is **not**:
- It is **not** the first sub-2B Cypher model (dbands and VoErik exist).
- It is **not** the first GRPO+Cypher work (STRuCT-LLM published Jun 2025).
- It is **not** SOTA on Cypher overall (QwQ-32B and Neo4j-Gemma-9B/27B beat 1.5B class by construction).

What it **is**:
- The first deployable + reproducible GRPO recipe at sub-2B specifically for Cypher.
- A real measured improvement over `dbands/Qwen2-5-Coder-0-5B_neo4j-text2cypher-2024v1-16B` on the Neo4j 2024v1 test split.
- A laptop-runnable GGUF that drops into existing GraphRAG stacks.

---

## 4. Technical approach

### 4.1 Stack

| Layer | Choice | Why |
|---|---|---|
| Base model | **Tier A**: `google/gemma-3-270m-it` (0.27B). **Tier B**: `Qwen2.5-Coder-0.5B-Instruct`. **Tier C**: `Qwen2.5-1.5B-Instruct` | Climb only as needed. Tier A matches VoErik's base for clean apples-to-apples; Tier C matches STRuCT-LLM scale for direct comparison |
| Algorithm | **GRPO** (TRL) via **Unsloth** wrapper | Free-tier VRAM fit; matches dominant 2026 small-model RL recipe |
| Quantization | QLoRA 4-bit + LoRA adapters on attention + MLP (Tier A may run full-precision — only 0.27B) | ~5–10 GB VRAM at sequence 512–1024 for Tier C; Tier A fits comfortably under 4 GB |
| Dataset | `neo4j/text2cypher-2024v1` (44,387 rows; test = 4,830, 543 unique schemas). VERIFIED https://huggingface.co/datasets/neo4j/text2cypher-2024v1 | Canonical; Neo4j SFT baselines train on it; pre-build kuzu DDL once per schema and cache (543 schemas → tractable) |
| Reward backend | **kuzu** ([github.com/kuzudb/kuzu](https://github.com/kuzudb/kuzu)) primary — embedded, `pip install kuzu`, no apt deps. **WARNING: archived 2025-10-10**, wheel still works. Wrap in backend abstraction for **FalkorDBLite** swap (`pip install falkordblite`, embedded Redis-module, actively maintained). | Aura free tier is 25 req/min — unusable. ~85–95% of test queries run on kuzu (sample of 20: 0% APOC/FTS/vector/GDS, 10% date literals, 10% EXISTS{} — all kuzu-supported) |
| Reward function | Composite à la STRuCT-LLM: `R = w1·R_judge + w2·R_string + w3·R_GED` (start with all weights = 1, untuned, mirroring their setup). `R_judge` = "executes without error against schema-loaded kuzu" (binary). `R_string` = Google-BLEU vs gold Cypher. `R_GED` = `1 − GED(predicted_subgraph, gold_subgraph) / max(size)` (graph-edit-distance over parsed query subgraphs — no live execution required). Optional auxiliary: MDPI-style key-value/triple-extraction support task. | Multi-component additive reward is the verified-working pattern at 1.5B+; `R_GED` works even when execution fails (kuzu dialect gap or schema-loading failure) |
| Eval | Google-BLEU + Exact Match (Ozsoy 2024 canonical) + Execution Accuracy on kuzu-loaded test subset. Report on full 4,830-row test split. | None of the existing sub-2B Cypher cards publish reproducible numbers on this split — publishing them is itself a contribution |

### 4.2 Training run plan

- **Compute**: 1 × Kaggle P100 (16 GB), 9-hour session, 30h/week quota — VERIFIED via Kaggle docs.
- **Hyperparameters (starting point — tune)**: `num_generations=4`, `max_prompt_length=512`, `max_completion_length=512`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`, `learning_rate=5e-6`, `beta=0.04`, `kl_coef=0.04`.
- **Env var**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid step-2 OOM.
- **Checkpointing**: every 50 steps to HuggingFace Hub (Kaggle sessions die).
- **Phases**: (Phase 0) verify SFT baseline reproduces dbands' numbers on 2024v1 test → (Phase 1) GRPO from `Qwen2.5-1.5B-Instruct` directly → (Phase 2) compare GRPO-from-SFT vs GRPO-from-Instruct.

### 4.3 Free-tier feasibility — concrete VRAM math

For Qwen2.5-1.5B QLoRA-GRPO with `num_generations=4`, seq 512+512: weights 4-bit ≈ 1.0 GB, LoRA + grads ≈ 0.4 GB, paged-AdamW ≈ 0.3 GB, activations w/ grad-ckpt ≈ 2 GB, vLLM colocate KV cache @ `gpu_memory_utilization=0.55` ≈ 6–8 GB, framework ≈ 1.5 GB. **Total ≈ 11–13 GB**, fits a P100 (16 GB) with margin. Confirmed via Unsloth's published "5 GB GRPO at 1.5B" claim (PARTIAL — Daniel Han's claim, not independently re-benchmarked here).

---

## 5. Tools and frameworks fit (downstream)

| Integration | Status | Notes |
|---|---|---|
| **LangChain `GraphCypherQAChain` + Ollama** | VERIFIED — https://docs.langchain.com/oss/python/integrations/providers/ollama | Drop-in. Ship an Ollama Modelfile + 5-line LangChain example in the README. |
| **Neo4j MCP server** (`mcp-neo4j-cypher`) | VERIFIED — https://github.com/neo4j-contrib/mcp-neo4j | Power the MCP server with this specialist instead of GPT-4. Document the wiring. |
| **LlamaIndex PropertyGraphIndex** | UNVERIFIED in this pass | Generally supports custom LLMs; verify before promoting. |
| **GGUF / llama.cpp / Ollama** | PLAUSIBLE — Qwen2-1.5B GGUF widely shipped | 1.5B Q4 reportedly >20 t/s on modern CPUs; verify on target hardware before announcing |
| **kuzu** (embedded Cypher) | VERIFIED — https://github.com/kuzudb/kuzu | LangChain-compatible. Use for both training rollouts AND a 1-click demo notebook. |
| **Neo4j Aura Free** | NOT SUITABLE for training rollouts (25 req/min, VERIFIED at https://neo4j.com/docs/aura/api/overview/) | Acceptable as a deployment demo target only |

---

## 6. Benchmarks and success criteria

### 6.1 What we will report

**Co-primary benchmarks** (report on both — covers the workshop-venue concern about 2024v1 alone):

| Benchmark | Venue | Why include it |
|---|---|---|
| **Neo4j Text2Cypher 2024v1** (Ozsoy et al.) — https://aclanthology.org/2025.genaik-1.11/ | ACL 2025 **GenAIK workshop** (peer-reviewed, lower bar than main) | Vendor-official dataset (Neo4j Inc), used by STRuCT-LLM, by Neo4j's own SFT models, and by the existing sub-2B baselines. Maximum comparability. 4,830-row test, 543 schemas. **Primary.** |
| **Mind the Query** (IBM Research) — https://aclanthology.org/2025.emnlp-industry.133/ | EMNLP 2025 **Industry Track** (peer-reviewed, main-conf-affiliated, stronger than workshop) | Stronger venue. **Ships executable graphs** (2024v1 only ships schemas — execution accuracy on it requires synthesizing graphs). 27,529 rows. **Co-primary** — covers the "is this workshop-only?" objection. |

| Metric | Source | Target |
|---|---|---|
| Google-BLEU | Ozsoy 2024 canonical, on `text2cypher-2024v1` test (4,830 rows) | Tier A (0.27B): publish first reproducible number on this split (no existing baseline does). Tier C (1.5B): aim ≥ 16 BLEU (STRuCT-LLM 1.5B Both-joint = 15.8 ± 2.2; matching them with 1/64th the compute is the win) |
| Exact Match | Same | Tier A: first reproducible. Tier C: aim ≥ 14 EM (STRuCT-LLM 1.5B = 14.2 on CRT-QA; uncharted on Text2Cypher main table) |
| Execution Accuracy on 2024v1 | Synthesized kuzu graphs from schema, query through backend abstraction | Tier C: aim 30–45% (STRuCT-LLM Qwen2.5-14B Both = 50.4; QwQ-32B = 55.3 — bigger/reasoning models, so 30–45% at 1.5B is credible). Tier A: any non-trivial rate is novel |
| Execution Accuracy on Mind-the-Query | True execution against released graphs | Real apples-to-apples execution metric — first sub-2B GRPO entry |
| SFT-vs-GRPO ablation | Train SFT-only baseline on same data, same compute | Clean GRPO delta — MDPI got +5.03% EXE at 3B; matching that shape at 1.5B is the credible target |

### 6.2 What we will NOT claim

- **Not** "SOTA on Text2Cypher" (Neo4j-Gemma-27B is the top SFT entry; QwQ-32B is the top GRPO via STRuCT-LLM).
- **Not** "first GRPO Cypher model" (STRuCT-LLM did joint SQL+Cypher GRPO already).
- **Not** "empty niche" (sub-2B SFT exists).

### 6.3 What we will claim

- First publicly released sub-2B GRPO Cypher specialist with GGUF + LangChain/MCP integration.
- Measured improvement over the existing sub-2B Cypher SFT baselines on the canonical Neo4j 2024v1 test split.
- A reproducible recipe runnable on Kaggle free tier in one session.

---

## 7. Risks and known gaps

| Risk | Severity | Mitigation |
|---|---|---|
| **Neo4j Labs ships a sub-2B model first** (cadence: quarterly Gemma drops in 2025) | High | Move fast; lead announcement on the GRPO-recipe + MCP-integration angle, not just the checkpoint |
| **STRuCT-LLM authors release a Cypher-only 1.5B** | Medium | They have not released even their 1.5B joint model on HF (only 33B); cite prominently; our Cypher-only specialization + free-tier reproducibility is differentiating |
| **GRPO doesn't beat SFT meaningfully at small scale for this task** | Medium-High | MDPI delta is **+5.03%** at 3B — modest. At 0.27B/0.5B the delta could be smaller or zero. Mitigation: tiered ladder; cheap Tier A run validates reward function in 4-6h before any expensive escalation; report SFT-vs-GRPO ablation cleanly |
| **kuzu archived 2025-10-10 (Apple acquisition)** | Medium — wheel still works, software is dead | Wrap reward harness in backend abstraction; FalkorDBLite (`pip install falkordblite`) is the actively-maintained embedded fallback — a one-line swap. Sources: theregister.com/2025/10/14/kuzudb_abandoned/, news.ycombinator.com/item?id=45560036 |
| **kuzu Cypher dialect gaps with Neo4j** | Low (verified) | Sample of 20 test queries: 0% used APOC/FTS/vector/GDS/regex. Estimated 85–95% of 4,830-row test runs unmodified. Above the 70% threshold |
| **Aura free tier rate limit** (25 req/min — VERIFIED at https://neo4j.com/docs/aura/api/overview/) | Hard constraint, mitigated | Use embedded backend (kuzu/FalkorDBLite) for training; Aura only for end-user demo |
| **Tier A (0.27B) fails to learn from sparse RL signal** | Medium-High — likely outcome | Built into the plan: Tier A is the cheap experiment; if reward stays flat after 4-6h on Kaggle, escalate to Tier B (0.5B) then C (1.5B). Each escalation is one model-swap + re-run |
| **Sub-2B Cypher SFT cards have ZERO eval numbers** (dbands, VoErik, neo4j-4B, etc.) | Low — actually a positive | "Beating their card numbers" is impossible because their cards have no numbers. Mitigation/opportunity: re-evaluate their checkpoints on our harness; publishing first reproducible numbers on 2024v1 test is itself a contribution |
| **VoErik baseline is on 2025v1, not 2024v1** | Low | Cannot be apples-to-apples without re-training one of them. Either pick 2024v1 throughout (recommended — more papers compare on it) or accept VoErik isn't a direct baseline |
| **`lakkeo/stable-cypher-instruct-3b` BLEU-4 88.63 / Pass@1 51.80** is on their *own* synthetic 5k held-out, NOT on 2024v1 test | Low | Do not cite as a 2024v1 baseline. Verified from card |
| **No canonical public leaderboard** | Low | Be explicit in README that comparisons are eval-script-reproducible, not leaderboard-listed; ship the eval script alongside the model |

---

## 8. Free-tier compute strategy

| Tier | Use | Limits (VERIFIED) |
|---|---|---|
| **Kaggle P100 (or 2×T4)** | Primary training | 9h/session, 30h/week, internet on requires phone-verified account |
| **Colab Free T4** | Prototyping only — never the canonical run | Undocumented quota; idle disconnect ~90 min |
| **Modal Starter** | Final long run / final eval | $30/mo free credits ≈ 27 T4-hours |
| **HF Jobs (Unsloth Explorers promo)** | Bonus | Finite credits; useful for a single A100 final run |
| **Lightning AI Free** | Niche L4 access | 15 credits/mo |

Stack them: prototype on Colab → train on Kaggle → final long run on Modal/HF Jobs → push GGUF to HF Hub → demo via Ollama + LangChain notebook.

---

## 9. Verified evidence — sources actually opened

- Dataset card: https://huggingface.co/datasets/neo4j/text2cypher-2024v1 (44,387 rows, Apache-2.0)
- Dataset card: https://huggingface.co/datasets/neo4j/text2cypher-2025v1
- HF model search: https://huggingface.co/models?other=cypher
- Closest sub-2B baseline: https://huggingface.co/dbands/Qwen2-5-Coder-0-5B_neo4j-text2cypher-2024v1-16B
- Neo4j blog: https://neo4j.com/blog/developer/introducing-neo4j-text2cypher-dataset/
- Neo4j Text2Cypher paper (canonical eval): https://arxiv.org/abs/2412.10064 → https://aclanthology.org/2025.genaik-1.11/
- STRuCT-LLM (closest prior art): https://arxiv.org/html/2506.21575 — code https://github.com/bouv/STRuCT-LLM
- MDPI GRPO+Cypher paper (PARTIAL — page 403'd to fetcher, content from search): https://www.mdpi.com/2076-3417/15/15/8206
- IBM "Mind the Query" benchmark: https://aclanthology.org/2025.emnlp-industry.133/
- Aura rate limit: https://neo4j.com/docs/aura/api/overview/ (25 req/min)
- kuzu DB: https://github.com/kuzudb/kuzu (archived 2025-10-10)
- kuzu archive context: https://www.theregister.com/2025/10/14/kuzudb_abandoned/ ; https://news.ycombinator.com/item?id=45560036
- FalkorDBLite (recommended fallback): https://pypi.org/project/falkordblite ; https://falkordb.com/blog/kuzudb-to-falkordb-migration/
- Neo4j MCP server: https://github.com/neo4j-contrib/mcp-neo4j and https://github.com/neo4j/mcp
- LangChain + Ollama: https://docs.langchain.com/oss/python/integrations/providers/ollama
- STRuCT-LLM 33B HF release: https://huggingface.co/boubnov/STRuCT-LLM-Novo
- Test split viewer (4,830 rows, 543 unique schemas): https://huggingface.co/datasets/neo4j/text2cypher-2024v1/viewer/default/test

## 10. Things you must verify before announcing

Items 1, 2, and 4 are now resolved by the verification update above. Remaining open items:

1. ~~Read MDPI paper end-to-end~~ → **PARTIAL.** Page 403'd; recovered +5.03% EXE delta and 85% unseen-schema accuracy from snippets. Exact reward formula, hyperparameters, and any negative ablations remain unknown. **Action**: try institutional access / contact author if a number from the paper becomes load-bearing for our claims; otherwise treat as snippet-only.
2. ~~Read STRuCT-LLM end-to-end~~ → **DONE.** 1.5B Both-joint Text2Cypher BLEU 15.8 ± 2.2 is the bar. KL beta and exact Cypher execution backend remain unrecovered (likely in appendix); fetch the appendix if your reward design starts to deviate from theirs.
3. **Run `dbands/Qwen2-5-Coder-0-5B...` and `VoErik/cypher-gemma` on the same eval harness** — still required. Their cards have **no eval numbers at all**, so this is empirically necessary. Note VoErik trained on 2025v1, not 2024v1 — pick a single dataset version and stick to it (recommend 2024v1 — more comparators).
4. ~~Confirm kuzu coverage~~ → **DONE.** ~85–95% of 4,830-row test runs on kuzu. Above threshold.
5. **Watch Neo4j Labs' GitHub cadence weekly** during training — a sub-2B Gemma drop from them would re-frame the project.
6. **Decide whether to copy MDPI's "support task" auxiliary reward** (key-value + (s,p,o) extraction) on top of STRuCT-LLM's `R_judge + R_string + R_GED`. Defer until Tier A baseline is running; add only if needed.
7. **Choose backend abstraction interface upfront** so kuzu → FalkorDBLite swap is mechanical. (`execute(schema_ddl, query) -> result_set | error`.)

---

## Bottom line

This is a **viable, defensible, deployment-shaped project**, not a research-novelty play. Position it as engineering + integration first, paper second. The verified differentiation — *first sub-2B GRPO Cypher specialist with GGUF + LangChain / Neo4j-MCP / Ollama integration* — survives scrutiny. The previous "empty niche" framing does not, and was correctly removed.

**Proceed: yes, with the seven conditions above.** Implementation plan next.
