"""HF Hub checkpoint helpers — works with TRL's `hub_strategy="checkpoint"` flow.

Key idea: TRL's Trainer with `hub_strategy="checkpoint"` automatically pushes the
`last-checkpoint` subfolder of `output_dir` to the hub repo at every save. On notebook
restart we just need to pull that subfolder back into `output_dir/last-checkpoint`
and pass `resume_from_checkpoint=True` to `trainer.train()`.

The verified TRL signature is:
    Trainer.train(resume_from_checkpoint: str | bool | None = None)
where `True` means "load the last checkpoint in output_dir".
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def maybe_resume(repo_id: str, output_dir: str | Path, hf_token: str | None = None) -> bool:
    """Pull `last-checkpoint` from HF Hub into `output_dir/last-checkpoint`.

    Returns True if a checkpoint was successfully pulled, False otherwise.
    Caller should pass `resume_from_checkpoint=True` to trainer.train() if True.
    """
    try:
        from huggingface_hub import snapshot_download, repo_exists
    except ImportError:
        logger.warning("huggingface_hub missing; skipping resume check")
        return False
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = hf_token or os.environ.get("HF_TOKEN")
    try:
        if not repo_exists(repo_id, repo_type="model", token=token):
            logger.info("No existing repo %s — fresh start.", repo_id)
            return False
    except Exception as e:
        logger.warning("repo_exists check failed (%s); assuming fresh start.", e)
        return False
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_dir),
            allow_patterns=["last-checkpoint/*", "last-checkpoint/**"],
            token=token,
        )
    except Exception as e:
        logger.warning("snapshot_download failed (%s); fresh start.", e)
        return False
    last_ckpt = output_dir / "last-checkpoint"
    if not last_ckpt.is_dir() or not any(last_ckpt.iterdir()):
        logger.info("No last-checkpoint dir on hub — fresh start.")
        return False
    logger.info("Resumed checkpoint at %s", last_ckpt)
    return True


def push_eval_results(repo_id: str, results: dict[str, Any], hf_token: str | None = None) -> None:
    """Upload eval_results.json to the model repo so the README can link to it."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub missing; skipping eval push")
        return
    token = hf_token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    payload = json.dumps(results, indent=2).encode("utf-8")
    api.upload_file(
        path_or_fileobj=payload,
        path_in_repo="eval_results.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update eval_results.json",
    )
    logger.info("Pushed eval_results.json to %s", repo_id)
