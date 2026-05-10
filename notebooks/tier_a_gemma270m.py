# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Text2Cypher GRPO — Tier A — `unsloth/gemma-3-270m-it`
#
# **Goal**: Cheapest-iteration GRPO run on the smallest viable base. If reward signal
# emerges by step 150, this is the artifact (CPU-runnable GGUF, ~10 t/s on a laptop).
# If it stalls, escalate to Tier B (0.5B) or Tier C (1.5B).
#
# **Where checkpoints live**: HuggingFace Hub repo `<HF_USERNAME>/text2cypher-grpo-gemma-270m`.
# Auto-push every 50 steps via `hub_strategy="checkpoint"`. Auto-resume on notebook restart.
#
# **Conversion**: this is a jupytext `.py`. To run as `.ipynb`:
#     `pip install jupytext && jupytext --to notebook tier_a_gemma270m.py`
# Or upload the `.py` directly as a Kaggle script — both work.

# %% [markdown]
# ## 1. Setup — pinned installs + Python version check

# %%
# !pip install -q -U trl==1.4.0 unsloth==2026.5.5 'transformers>=4.50,<4.60' accelerate peft \
#                   'datasets>=3.2' bitsandbytes 'vllm>=0.6,<0.8' \
#                   networkx nltk kuzu==0.11.3 huggingface_hub

import sys
print("Python:", sys.version)

import nltk
nltk.download("punkt", quiet=True)

# %% [markdown]
# ## 2. Auth — pull HF token from Kaggle Secrets (or env on local)

# %%
import os

try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("Loaded HF_TOKEN from Kaggle Secrets.")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN in Kaggle Secrets (preferred) or as env var.")
    print("Using HF_TOKEN from env.")

# Determine HF username from token
from huggingface_hub import whoami
HF_USERNAME = whoami(token=HF_TOKEN)["name"]
print("HF user:", HF_USERNAME)

# %% [markdown]
# ## 3. Bootstrap the shared package
#
# Two options on Kaggle:
#   - `git clone <your-github-repo>` and `pip install -e ./RL_exp`
#   - Upload the `text2cypher_grpo/` folder as a Kaggle Dataset and add to inputs
# Locally for iteration, just `pip install -e .` from the project root.

# If not yet on this Kaggle session, clone the repo:
# !git clone https://github.com/<YOUR_USERNAME>/text2cypher-grpo /kaggle/working/text2cypher-grpo

# %%
import glob
import os
import sys

# Auto-discover the package across common locations
search_globs = [
    "/kaggle/input/*/text2cypher_grpo",
    "/kaggle/working/*/text2cypher_grpo",
    os.path.expanduser("~/**/text2cypher_grpo"),
    "./**/text2cypher_grpo",
]
matches = []
for pat in search_globs:
    matches.extend(glob.glob(pat, recursive=True))
matches = [os.path.dirname(m) for m in matches if os.path.isdir(m)]
if not matches:
    raise RuntimeError(
        "text2cypher_grpo/ not found. Either:\n"
        "  (a) !git clone https://github.com/<YOUR_USERNAME>/text2cypher-grpo /kaggle/working/text2cypher-grpo\n"
        "  (b) upload the folder as a Kaggle Dataset and add it to your notebook's inputs"
    )
pkg_root = matches[0]
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)
print("Using package at:", pkg_root)

from text2cypher_grpo import (
    schema_to_kuzu_ddl, build_reward_funcs, KuzuBackend, evaluate, maybe_resume, push_eval_results,
)

# %% [markdown]
# ## 4. Config — every tier-specific knob in one place

# %%
from dataclasses import dataclass

@dataclass
class TierConfig:
    tier_id: str = "gemma-270m"
    model_id: str = "unsloth/gemma-3-270m-it"
    repo_id: str = f"{HF_USERNAME}/text2cypher-grpo-gemma-270m"
    output_dir: str = "/kaggle/working/checkpoints"
    max_seq_length: int = 1024
    load_in_4bit: bool = False  # 270m is small enough for full BF16 LoRA
    full_finetune: bool = False  # LoRA always — keeps adapters small for HF Hub push
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # GRPO knobs
    num_generations: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 256
    learning_rate: float = 1e-5     # higher than TRL default 1e-6 — small model, more capacity to learn
    beta: float = 0.04              # KL weight — TRL default is 0.0; we want some regularization
    epsilon: float = 0.2
    grpo_batch_size: int = 1
    grad_accum: int = 8
    max_steps: int = 300
    save_steps: int = 50
    eval_steps: int = 100
    eval_subset_size: int = 200     # rows from test split for in-loop eval
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.50  # leave headroom for trainer; tune empirically

CFG = TierConfig()
print(CFG)

# %% [markdown]
# ## 5. Dataset — load and format prompts
#
# We attach a `prompt` column. TRL auto-passes every other column (including `cypher`
# and `schema`) to reward functions as keyword args.

# %%
from datasets import load_dataset

PROMPT_TEMPLATE = (
    "You are a Cypher expert. Given the schema and question, return only the Cypher query.\n\n"
    "Schema:\n{schema}\n\n"
    "Question: {question}\n\n"
    "Cypher:\n"
)

ds = load_dataset("neo4j/text2cypher-2024v1")
print(ds)

def add_prompt(row):
    row["prompt"] = PROMPT_TEMPLATE.format(schema=row["schema"], question=row["question"])
    return row

train_ds = ds["train"].map(add_prompt)
test_ds = ds["test"].map(add_prompt)

# Drop columns the trainer / reward functions don't need
train_ds = train_ds.remove_columns([c for c in train_ds.column_names
                                    if c not in ("prompt", "cypher", "schema", "question")])
test_ds = test_ds.remove_columns([c for c in test_ds.column_names
                                  if c not in ("prompt", "cypher", "schema", "question", "data_source")])

print("Train rows:", len(train_ds), "  Test rows:", len(test_ds))

# Quick sanity check on the schema parser — % of train schemas that yield non-trivial DDL
from text2cypher_grpo.schema import parse_rate
sample_schemas = [train_ds[i]["schema"] for i in range(0, len(train_ds), max(1, len(train_ds) // 200))]
parse_stats = parse_rate(sample_schemas)
print("Schema parse rate (sample):", parse_stats)

# %% [markdown]
# ## 6. Reward functions

# %%
backend = KuzuBackend()
reward_funcs = build_reward_funcs(backend=backend)
print("Reward functions:", [getattr(f, "__name__", repr(f)) for f in reward_funcs])

# Quick smoke test of the reward chain on one row
row = train_ds[0]
fake_completions = [row["cypher"]]  # the gold query — should score high on all rewards
fake_completion_ids = [list(range(50))]  # fake ids
r_judge = reward_funcs[0]
print("Smoke r_judge on gold:", r_judge(
    prompts=[row["prompt"]],
    completions=fake_completions,
    completion_ids=fake_completion_ids,
    cypher=[row["cypher"]],
    schema=[row["schema"]],
))

# %% [markdown]
# ## 7. Model load — Unsloth FastLanguageModel

# %%
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CFG.model_id,
    max_seq_length=CFG.max_seq_length,
    load_in_4bit=CFG.load_in_4bit,
    fast_inference=CFG.use_vllm,
    gpu_memory_utilization=CFG.vllm_gpu_memory_utilization,
)
print("Loaded model:", CFG.model_id)

model = FastLanguageModel.get_peft_model(
    model,
    r=CFG.lora_r,
    lora_alpha=CFG.lora_alpha,
    lora_dropout=CFG.lora_dropout,
    target_modules=list(CFG.lora_target_modules),
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print("LoRA adapter attached.")

# %% [markdown]
# ## 8. Resume from HF Hub if a previous run exists

# %%
resumed = maybe_resume(CFG.repo_id, CFG.output_dir, hf_token=HF_TOKEN)
print("Resuming from previous checkpoint:", resumed)

# %% [markdown]
# ## 9. GRPOConfig + Trainer

# %%
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir=CFG.output_dir,
    num_train_epochs=1,
    max_steps=CFG.max_steps,
    per_device_train_batch_size=CFG.grpo_batch_size,
    gradient_accumulation_steps=CFG.grad_accum,
    learning_rate=CFG.learning_rate,
    beta=CFG.beta,
    epsilon=CFG.epsilon,
    num_generations=CFG.num_generations,
    max_prompt_length=CFG.max_prompt_length,
    max_completion_length=CFG.max_completion_length,
    use_vllm=CFG.use_vllm,
    vllm_gpu_memory_utilization=CFG.vllm_gpu_memory_utilization,
    logging_steps=5,
    save_strategy="steps",
    save_steps=CFG.save_steps,
    save_total_limit=2,
    save_safetensors=True,
    push_to_hub=True,
    hub_model_id=CFG.repo_id,
    hub_strategy="checkpoint",      # critical — pushes `last-checkpoint` subfolder
    hub_token=HF_TOKEN,
    hub_private_repo=False,
    remove_unused_columns=False,    # keep `cypher` and `schema` columns for reward funcs
    bf16=True,
    optim="paged_adamw_8bit",
    warmup_ratio=0.05,
    seed=42,
    report_to="none",               # set to "wandb" if WANDB_API_KEY set
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    reward_funcs=reward_funcs,
)

# %% [markdown]
# ## 10. Train — auto-resume if checkpoint pulled in step 8

# %%
trainer.train(resume_from_checkpoint=resumed if resumed else None)

# Final push (in addition to the periodic `last-checkpoint` pushes)
trainer.push_to_hub(commit_message=f"end of training — tier {CFG.tier_id}")

# %% [markdown]
# ## 11. Eval on test split — Google-BLEU + EM + Execution Accuracy
#
# Decision gate: if `R_judge` mean was < 0.15 by step 150 (visible in `trainer_state.json`),
# escalate to Tier B/C. Otherwise run full eval here.

# %%
FastLanguageModel.for_inference(model)  # switch off training optimizations

eval_results = evaluate(
    model=model,
    tokenizer=tokenizer,
    dataset=test_ds,
    backend=backend,
    max_examples=CFG.eval_subset_size,  # small subset for fast run; expand to None for full 4,830
    output_path=os.path.join(CFG.output_dir, "eval_results.json"),
)
print("Overall:", eval_results["overall"])
print("Per source:")
for k, v in eval_results["per_source"].items():
    print(f"  {k}: {v}")

# Push eval to the model repo so the README can link to it
push_eval_results(CFG.repo_id, eval_results, hf_token=HF_TOKEN)

# %% [markdown]
# ## 12. (Optional) Run the full 4,830-row test eval
#
# Only after the small-subset eval looks reasonable. ~30-90 minutes depending on GPU.

# %%
# eval_results_full = evaluate(
#     model=model,
#     tokenizer=tokenizer,
#     dataset=test_ds,
#     backend=backend,
#     max_examples=None,
#     output_path=os.path.join(CFG.output_dir, "eval_results_full.json"),
# )
# push_eval_results(CFG.repo_id, eval_results_full, hf_token=HF_TOKEN)
