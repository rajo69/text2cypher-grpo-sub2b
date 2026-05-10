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
# # Text2Cypher GRPO — Tier C — `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`
#
# **Goal**: safety-net run at the size where reward signal is almost guaranteed to emerge.
# Matches STRuCT-LLM's smallest model size for direct comparison (their Qwen2.5-1.5B
# Both-joint Text2Cypher BLEU = 15.8 ± 2.2 — that's the bar at this scale).
#
# **Where checkpoints live**: HuggingFace Hub repo `<HF_USERNAME>/text2cypher-grpo-qwen-1p5b`.
# Auto-push every 50 steps via `hub_strategy="checkpoint"`. Auto-resume on restart.
#
# **Run in parallel with Tier A** — separate HF repo, separate Kaggle session.
#
# **Conversion**: this is a jupytext `.py`. Convert to `.ipynb` if needed:
#     `pip install jupytext && jupytext --to notebook tier_c_qwen15b.py`

# %% [markdown]
# ## 1. Setup

# %%
# !pip install -q -U trl==1.4.0 unsloth==2026.5.5 'transformers>=4.50,<4.60' accelerate peft \
#                   'datasets>=3.2' bitsandbytes 'vllm>=0.6,<0.8' \
#                   networkx nltk kuzu==0.11.3 huggingface_hub

import sys
print("Python:", sys.version)

import nltk
nltk.download("punkt", quiet=True)

# %% [markdown]
# ## 2. Auth

# %%
import os
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = HF_TOKEN
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN in Kaggle Secrets or env var.")
from huggingface_hub import whoami
HF_USERNAME = whoami(token=HF_TOKEN)["name"]
print("HF user:", HF_USERNAME)

# %% [markdown]
# ## 3. Bootstrap shared package

# If not yet on this Kaggle session, clone the repo:
# !git clone https://github.com/rajo69/text2cypher-grpo /kaggle/working/text2cypher-grpo

# %%
import glob
import os
import sys

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
    build_reward_funcs, KuzuBackend, evaluate, maybe_resume, push_eval_results,
)

# %% [markdown]
# ## 4. Config — Tier C specifics

# %%
from dataclasses import dataclass

@dataclass
class TierConfig:
    tier_id: str = "qwen-1p5b"
    model_id: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    repo_id: str = f"{HF_USERNAME}/text2cypher-grpo-qwen-1p5b"
    output_dir: str = "/kaggle/working/checkpoints"
    max_seq_length: int = 2048              # Qwen handles longer context comfortably
    load_in_4bit: bool = True               # QLoRA — needed to fit on T4/P100
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    num_generations: int = 4                # 4 fits T4; raise to 6 on P100 if VRAM allows
    max_prompt_length: int = 768
    max_completion_length: int = 384
    learning_rate: float = 5e-6             # smaller than Tier A — bigger model, slower drift
    beta: float = 0.04
    epsilon: float = 0.2
    grpo_batch_size: int = 1
    grad_accum: int = 8
    max_steps: int = 300
    save_steps: int = 50
    eval_steps: int = 100
    eval_subset_size: int = 200
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.40   # tighter than Tier A — bigger model, more grad memory

CFG = TierConfig()
print(CFG)

# %% [markdown]
# ## 5. Dataset

# %%
from datasets import load_dataset

PROMPT_TEMPLATE = (
    "You are a Cypher expert. Given the schema and question, return only the Cypher query.\n\n"
    "Schema:\n{schema}\n\n"
    "Question: {question}\n\n"
    "Cypher:\n"
)

ds = load_dataset("neo4j/text2cypher-2024v1")

def add_prompt(row):
    row["prompt"] = PROMPT_TEMPLATE.format(schema=row["schema"], question=row["question"])
    return row

train_ds = ds["train"].map(add_prompt)
test_ds = ds["test"].map(add_prompt)
train_ds = train_ds.remove_columns([c for c in train_ds.column_names
                                    if c not in ("prompt", "cypher", "schema", "question")])
test_ds = test_ds.remove_columns([c for c in test_ds.column_names
                                  if c not in ("prompt", "cypher", "schema", "question", "data_source")])
print("Train rows:", len(train_ds), "  Test rows:", len(test_ds))

# %% [markdown]
# ## 6. Reward functions

# %%
backend = KuzuBackend()
reward_funcs = build_reward_funcs(backend=backend)
print("Reward functions:", [getattr(f, "__name__", repr(f)) for f in reward_funcs])

# %% [markdown]
# ## 7. Model load — Unsloth FastLanguageModel (4-bit QLoRA)

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

# %% [markdown]
# ## 8. Resume

# %%
resumed = maybe_resume(CFG.repo_id, CFG.output_dir, hf_token=HF_TOKEN)
print("Resuming:", resumed)

# %% [markdown]
# ## 9. Trainer

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
    hub_strategy="checkpoint",
    hub_token=HF_TOKEN,
    hub_private_repo=False,
    remove_unused_columns=False,
    bf16=True,
    optim="paged_adamw_8bit",
    warmup_ratio=0.05,
    seed=42,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    reward_funcs=reward_funcs,
)

# %% [markdown]
# ## 10. Train

# %%
trainer.train(resume_from_checkpoint=resumed if resumed else None)
trainer.push_to_hub(commit_message=f"end of training — tier {CFG.tier_id}")

# %% [markdown]
# ## 11. Eval

# %%
FastLanguageModel.for_inference(model)

eval_results = evaluate(
    model=model,
    tokenizer=tokenizer,
    dataset=test_ds,
    backend=backend,
    max_examples=CFG.eval_subset_size,
    output_path=os.path.join(CFG.output_dir, "eval_results.json"),
)
print("Overall:", eval_results["overall"])
for k, v in eval_results["per_source"].items():
    print(f"  {k}: {v}")

push_eval_results(CFG.repo_id, eval_results, hf_token=HF_TOKEN)

# %% [markdown]
# ## 12. (Optional) full 4,830-row eval — uncomment after subset looks reasonable

# %%
# eval_results_full = evaluate(
#     model=model, tokenizer=tokenizer, dataset=test_ds, backend=backend,
#     max_examples=None,
#     output_path=os.path.join(CFG.output_dir, "eval_results_full.json"),
# )
# push_eval_results(CFG.repo_id, eval_results_full, hf_token=HF_TOKEN)
