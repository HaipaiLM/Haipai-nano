# 🧠 **model_safe — A Readable, Modern GPT-Style Transformer**

**Haipai** (“high-pie”) is a compact, fully readable **decoder-only Transformer** you can understand, train, and extend yourself.
It’s designed to behave like a scaled-down GPT-style model — but remain small enough to fit comfortably on a single modern GPU (or even CPU inference).

The model implements the same modern components as today’s large LMs — **RMSNorm**, **SwiGLU**, **Rotary Position Embeddings (RoPE)**, **causal SDPA attention**, **tied embeddings**, and **EMA-smoothed weights** — in about 200 lines of PyTorch.

This release includes:

* ✅ **EMA-averaged weights** in `.safetensors`
* ✅ **Tokenizer** (trained on mixed Wikipedia + CNN/DailyMail)
* ✅ **Minimal PyTorch model implementation**
* ✅ **Inference script** with temperature/top-p sampling

---

## ⚡ Highlights

| Feature             | Description                                       |
| :------------------ | :------------------------------------------------ |
| **Architecture**    | Decoder-only Transformer (GPT-style)              |
| **Modern features** | RMSNorm, SwiGLU, RoPE, SDPA, EMA                  |
| **Parameter count** | ~50M (depending on vocab)                      |
| **Context length**  | 1,024 tokens (RoPE scaled ×2)                     |
| **Precision**       | fp16 / bf16                                       |
| **Tokenizer**       | BPE, 50k vocab, lowercase + whitespace            |
| **Export format**   | `.safetensors` (EMA-smoothed)                     |
| **Intended use**    | Pretraining demos, teaching, small-scale research |
| **License**         | (choose: MIT / Apache-2.0 / custom)               |

---

## 🧱 Architecture Overview

**HaipaiLM** is a GPT-style decoder block stack with **RMSNorm → Attention → Residual** and **RMSNorm → SwiGLU → Residual**, finalized by an RMSNorm and tied embedding head.

It uses **RoPE** for positional encoding applied to both Q and K, and **causal SDPA** (PyTorch fused attention) for fast inference.

```
Input IDs
   ↓
Embedding
   ↓
[× N blocks]
   ├─ RMSNorm → Multi-Head Attention (RoPE on QK causal SDPA) → Add Residual
   └─ RMSNorm → SwiGLU → Add Residual
   ↓
Final RMSNorm
   ↓
Tied LM Head (shared with Embedding)
```

---

### 🧩 Diagram

This schematic shows how **HaipaiLM** processes tokens:

<img width="938" height="665" alt="diagram-export-11-8-2025-9_11_04-PM" src="https://github.com/user-attachments/assets/39e4b6b1-c889-4c2c-a94e-f8386a0cae26" />


Each block uses:

* **RMSNorm 1 → RoPE → Multi-Head Attention → Residual**
* **RMSNorm 2 → SwiGLU → Residual**
* **Final RMSNorm** normalizes before the output head
* **Embedding ↔ LM Head** share the same weights

---

## ⚙️ Training Configuration

| Component                    | Setting                           |
| :--------------------------- | :-------------------------------- |
| **Layers**                   | 12                                |
| **Hidden size (d_model)**    | 384                               |
| **Attention heads (n_head)** | 6                                 |
| **Head dimension**           | 64                                |
| **Feed-forward (SwiGLU)**    | 2048                              |
| **Norm**                     | RMSNorm (ε=1e-6)                  |
| **Dropout**                  | 0.0–0.01                          |
| **RoPE scale**               | 2.0                               |
| **Optimizer**                | AdamW (β₁=0.9, β₂=0.95, ε=1e-8)   |
| **Weight decay**             | 0.1                               |
| **Gradient clip**            | 0.8                               |
| **LR schedule**              | Linear warmup → Cosine decay      |
| **Warmup steps**             | 1000                              |
| **Max LR**                   | 3e-3                              |
| **EMA decay**                | 0.9995                            |
| **Precision**                | AMP (bf16/fp16)                   |
| **Objective**                | Causal LM (next-token prediction) |

---

## 📊 Training Recipe Summary

**Dataset mix:**

* 60% Wikipedia (latest English dump)
* 40% CNN/DailyMail (news articles)
* ~200–500M total tokens

**Tokenizer:**

* Trained BPE (50k vocab)
* Lowercased, whitespace pre-tokenized
* Saved in `/tokenizer/` folder

**Validation perplexity:**

* ~60 after early stage (1k–2k steps)
* ~25–30 after full 500M-token run
* **EMA weights** yield smoother generations

---

## 🧪 Local Inference

```python
import torch, os, json
from safetensors.torch import load_file
from transformers import AutoTokenizer
from modeling_haipai import HaipaiLM

repo_dir = "./model_safe"

# 1. Load config + tokenizer
with open(os.path.join(repo_dir, "config.json")) as f:
    cfg = json.load(f)
tok = AutoTokenizer.from_pretrained(os.path.join(repo_dir, "tokenizer"), local_files_only=True)
if tok.pad_token is None and tok.eos_token is not None:
    tok.pad_token = tok.eos_token

# 2. Load model
state = load_file(os.path.join(repo_dir, "model.safetensors"))
model = HaipaiLM(
    vocab_size=tok.vocab_size,
    d_model=cfg["d_model"], n_layer=cfg["n_layer"], n_head=cfg["n_head"],
    d_ff=cfg["d_ff"], max_seq_len=cfg["max_position_embeddings"],
    rope_scale=cfg.get("rope_scale", 2.0),
).to("cpu").eval()
model.load_state_dict(state, strict=False)

# 3. Generate
prompt = "Doctor told"
ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    logits, _ = model(ids)
```

---

## 💻 CLI Inference

```bash
python inference.py --local_dir "./model_safe" --prompt "Doctor told" --device cpu
```

Options:

```
--local_dir            Path to model + tokenizer
--prompt               Input prompt
--device               cuda / cpu
--max_new_tokens       (default 120)
--temperature          (default 0.7)
--top_p                (default 0.9)
--repetition_penalty   (default 1.15)
```

---

## 🔥 Sampling Defaults

```python
temperature = 0.7
top_p = 0.9
repetition_penalty = 1.15
max_new_tokens = 150
```

You can also experiment with:

* Lower temperature (0.6) for factual output
* Higher top_p (0.95) for more variety
* Larger context windows for longer text generation

---

## 📈 Evaluate Perplexity

```python
import math, torch
@torch.no_grad()
def ppl_on_texts(model, tok, texts):
    losses = []
    for text in texts:
        enc = tok(text, return_tensors="pt")
        ids = enc.input_ids
        labels = ids.clone()
        labels[:, :-1] = ids[:, 1:]
        labels[:, -1] = -100
        _, loss = model(ids, labels)
        losses.append(loss.item())
    return math.exp(sum(losses)/len(losses))
```

---

## 🧩 Design Notes

**RMSNorm** – lightweight alternative to LayerNorm, numerically stable for small models.
**SwiGLU** – nonlinearity `SiLU(W₁x) ⊙ W₂x → W₃`, better expressiveness.
**RoPE** – encodes relative positions via sinusoidal rotation of Q/K.
**Causal SDPA** – built-in PyTorch scaled dot-product attention backend (FlashAttention on supported GPUs).
**EMA** – maintains a running exponential average of weights during training, improving validation loss and sample smoothness.

---

## ⚠️ Limitations

* ~50M parameters — for research, not production deployment.
* Not instruction-tuned; won’t follow chat-style prompts.
* Quality depends on corpus balance (Wikipedia + news).
* May produce factual or stylistic inconsistencies.
  Use responsibly with human review.

---

## 🧭 Extending Haipai

You can scale the model easily:

* 100M: double `d_model` and `d_ff`
* 1B+: use Mixture-of-Experts (MoE) for efficient scaling
* Replace `HaipaiBlock` with sparse MoE or gated variants
* Try longer context windows by rebuilding RoPE cache

---

## 📘 Citation

```bibtex
@software{haipai2025,
  title   = {Haipai: A Minimal, Modern GPT-Style Language Model (~50M, EMA)},
  author  = {Md Rakibul Islam Rocky},
  year    = {2025},
  url     = {https://huggingface.co/rocky1410/haipai-nano}
}
```

---

## ⚖️ License

This repository is open for research and educational use on Apache 2.0 License

---
