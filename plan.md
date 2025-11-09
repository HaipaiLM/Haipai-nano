# Haipai Hybrid Efficiency Plan (MoR-First)

Goal: build a lean, $15-budget, ~1B-class Haipai entirely around conditional computation (Mixture-of-Recursions + MoD + MoE) without ever running a standalone dense 500 M training. Every stage cites the research grounding the approach.

---

## Stage 0 – Data & Low-Precision Training Setup

1. **Curriculum + Reformulation** – Prepare training data using curriculum techniques so the model sees easy tokens first, accelerating convergence at tiny scales (arXiv:2506.11300). Complement this with Massive Genre-Audience reformulation (arXiv:2502.04235) so your existing corpus yields more “useful” samples without scraping.
2. **Limited-Memory LM mindset** – Tag factual snippets and keep them in an external store as suggested by LMLM (arXiv:2505.15962). The model focuses on reasoning, not memorization, reducing the required parameter budget.
3. **INT4/INT8 Training** – Implement Q-GaLore / low-precision optimizer tricks (arXiv:2505.01043) in the training loop so gradients/optimizer states live in INT4/INT8 from day one.

Deliverable: Updated `train.py` (or `train_mor.py`) that loads curriculum shards, reformulated text, and uses low-precision GaLore.

---

## Stage 1 – Train a Tiny MoR Backbone from Scratch

1. **Mixture-of-Recursions base** – Use the MoR architecture (arXiv:2507.10524) as the default Haipai block stack: shared layers across recursive steps, token-choice & expert-choice routing, recursion-wise KV caching, and per-token depth budgets.
2. **Built-in MoD / ACM** – Embed Router-Tuning Mixture-of-Depths or Adaptive Computation Modules directly inside the initial model so tokens skip entire blocks immediately (arXiv:2410.13184, 2312.10193).
3. **Dynamic Token Routing** – Integrate DTRNet / Leap-of-Thought gating (arXiv:2509.00925; OpenReview 0SF6Kr1lrx) so attention cost is conditional without deleting information.
4. **SelfBudgeter control** – Add a budget head to regulate how much compute each sequence receives (arXiv:2505.11274), feeding its signals into MoR/MoD thresholds.

Deliverables: `modeling_haipai.py` rewritten around MoR blocks; `train_mor.py` handling recursion, routers, and budget logging. This tiny MoR (150–200 M) runs for ~3–4 GPU-hours in low precision, staying within a few dollars.

---

## Stage 2 – Cheaply Scale via HyperCloning

1. **HyperClone / LESA expansion** – Once the tiny MoR converges, expand hidden dims and FF widths using HyperCloning/LESA (arXiv:2409.12903, 2502.13794). Because MoR ties weights across recursion, the cloning cost is minimal.
2. **Short stabilization run** – Do a brief low-LR fine-tune (≤1 hour) to settle the widened parameters while keeping routers active.

Deliverable: `hyperclone.py` + configs describing the scale-up target (~1B effective capacity) and stabilization schedule.

---

## Stage 3 – Sparse Experts & Adaptive Token Compression

1. **Expert-Choice MoE / Mixture-of-Tokens** – Replace half the MLPs with expert-choice routers (arXiv:2406.18219) or Mixture-of-Tokens layers (NeurIPS 2024, paper ID bc427eb3…) so each token only activates a couple of experts per recursion step.
2. **Adaptive Token Merging** – Integrate QuickMerge++ (arXiv:2508.13204) plus the training-free Pareto-optimal merging schemes for edge contexts (arXiv:2509.09955, 2509.09168) so sequence length shrinks dynamically without harming accuracy.
3. **Evaluator-Head Compression / REFORM** – Use retrieval-head analysis (arXiv:2501.12959) or the REFORM pipeline (arXiv:2506.01215) to drop irrelevant KV entries before each recursion, preserving only the core context.

Deliverables: `experts.py`, `token_merging.py`, telemetry dashboards tracking router balance, merge ratios, and perplexity impact.

---

## Stage 4 – Long-Context Hybridization

1. **Transformer + SSM Hybrid** – Interleave Haipai attention blocks with Mamba/StripedHyena-style state-space modules to capture the 4× speedups observed beyond ~57K tokens (arXiv:2507.12442, 2403.19887). MoR’s recursion loop treats both block types uniformly.
2. **Core-Context Attention** – Adopt core-context aware attention so quadratic compute targets salient windows and peripheral tokens are summarized cheaply (arXiv:2412.12465).

Deliverable: Extended model definition supporting hybrid blocks and context-aware routing flags.

---

## Stage 5 – Quantization & Inference Stack

1. **SpinQuant / GuidedQuant W4A4KV4** – Quantize the expanded MoR model using learned rotations (arXiv:2405.16406) plus GuidedQuant refinements (arXiv:2505.07004) to keep accuracy within ~2–3 points of FP while slashing costs.
2. **Speculative Decoding** – Implement draft/verify decoding per the 2024 survey (arXiv:2401.07851). Use recursion boundaries as natural verification checkpoints so speculative tokens align with MoR steps.
3. **Token-Budget-Aware Reasoning** – Train with TALE-style losses so generation length obeys explicit budgets (ACL Findings 2025 “Token-Budget-Aware LLM Reasoning”).

Deliverables: `quantize.py`, updated `inference.py` supporting draft models, recursion-aware verification, and budget enforcement.

---

## Stage 6 – Cost Envelope & Repo Workflow

1. **Compute budget**  
   - Tiny MoR training (low-precision): 3–4 GPU-hours (~$3–4).  
   - HyperClone + stabilization: ~1 hour ($1).  
   - MoE/MoT + token routing fine-tunes: 3 hours ($3).  
   - Token merging / hybrid blocks / quantization passes: 3–4 hours ($3–4).  
   - Total ≈10–12 hours → $10–12 on a $1/hr L4 (staying under the $15 ceiling).

2. **Repo org**  
   - Keep `older_version/` as the dense baseline.  
   - Add `next_gen/` (or similar) housing MoR-first code: `train_mor.py`, `hyperclone.py`, `experts.py`, `token_merging.py`, `quantize.py`.  
   - Update README and docs with the arXiv references above and instructions for reproducing each stage.

Outcome: a Mixture-of-Recursions-first Haipai that never trains a large dense baseline yet achieves 1B-class intelligence via conditional depth, recursive reuse, sparse experts, adaptive token routing/merging, hybrid SSM blocks, and aggressive quantization—all within a ~$15 compute budget.
