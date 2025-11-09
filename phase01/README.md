# Phase 0 & 1 (MoR-First) Implementation

This folder contains the executable code for **Stage 0** (data + tokenizer + low-precision prep) and **Stage 1** (tiny Mixture-of-Recursions backbone) exactly as described in `plan.md`.

Key principles:

- Every long-running step streams progress with `tqdm` (dataset scoring, MGA reformulation, tokenizer fitting, dataloader batching, training epochs, logging flushes). You’ll never stare at a blank terminal.
- All training/inference scripts log metrics, router stats, and precision info to Weights & Biases. Set `WANDB_API_KEY` before running or pass `--disable_wandb`.
- Tokenizer training is baked in (50 K vocab). Use `python data/tokenizer_50k.py --dataset ...` once you supply the dataset name/path.
- Dataset selection is your call; plug in any Hugging Face dataset or custom text folder when invoking the scripts. Curriculum + MGA layers sit on top.

Contents (high-level):

- `data/` – curriculum scheduler, MGA reformulation, tokenizer trainer, limited-memory store glue.
- `models/` – Tiny MoR, Router-Tuning MoD, ACM gate, DTR/LoT routing, SelfBudgeter head.
- `training/` – Phase 0 driver (`prepare_phase0.py`) and Phase 1 training loop (`train_mor.py`), both with detailed progress bars + W&B hooks.
- `utils/` – logging context, low-precision helpers, shared progress-bar wrappers, config loading.

Fill in dataset paths + hyperparameters via CLI flags or YAML under `config/`. The defaults stay minimal (≤200 M parameters) so experiments remain inside the $15 GPU budget target.
