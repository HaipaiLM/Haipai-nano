from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, Optional

import wandb


def should_use_wandb(disable: bool) -> bool:
    if disable:
        return False
    return bool(os.environ.get("WANDB_API_KEY"))


@contextmanager
def wandb_run(project: str, name: str, config: Optional[Dict] = None, disable: bool = False):
    if not should_use_wandb(disable):
        yield None
        return
    run = wandb.init(project=project, name=name, config=config or {}, reinit=True)
    try:
        yield run
    finally:
        run.finish()
