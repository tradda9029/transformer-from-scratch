# Transformer from Scratch

Implementing a Transformer from scratch while learning Python, PyTorch, and the mathematical foundations of Transformer architectures.

The long-term goal is to build reusable Transformer components for research on In-Context Learning (ICL), including Synthetic Linear Regression and Looped Transformers.

## Progress

- [x] 01 — Python and PyTorch Basics
- [x] 02 — Single-Head Causal Self-Attention
- [x] 03 — Minimal Language Model (Bigram LM)
- [x] 04 — Attention-based Language Model
- [ ] 05 — Transformer Block (Residual Connections, LayerNorm, FFN)
- [ ] 06 — Multi-Head Attention and Multi-Layer Transformer
- [ ] 07 — Synthetic Linear Regression ICL
- [ ] 08 — Looped Transformer
- [ ] 09 — Position Corruption Experiments
- [ ] 10 — Hidden-State Analysis and Ablations

## Repository Structure

- `exercises/` — Learning exercises, model training, and experiments
- `src/` — Reusable model components, including `SelfAttention`
- `README.md` — Project overview and implementation progress

## Chapter 04: Attention Language Model

Implemented a character-level causal language model with:

- Token Embedding
- Optional Learned Positional Embedding
- Single-Head Causal Self-Attention
- Linear projection to vocabulary logits
- Cross-Entropy Loss and AdamW training
- Autoregressive token generation with a context-length limit

The model uses a reusable `SelfAttention` module from `src/self_attention.py`.

### Tests

Successfully tested:

- Forward output shapes
- Causal independence from future tokens
- Gradient propagation
- Positional sensitivity
- Maximum context length
- Positional embedding ablation
- Parameter counting
- Training and evaluation

### Model Comparison

Configuration: embedding dimension 32, head dimension 32, block size 8, 1000 training steps, batch size 8, AdamW with learning rate 0.001.

| Model | Parameters | Evaluation Loss |
|---|---:|---:|
| Bigram LM | 100 | 1.6671 |
| Attention LM without positional embeddings | 3,722 | 0.2263 |
| Attention LM with positional embeddings | 3,978 | 0.1732 |

All models were trained using the same sequence of mini-batches.

The evaluation uses all valid context windows from the original training text. Therefore, these results measure fitting to the training source, not generalization to unseen data.

### Running the Comparison

From the repository root:

`PYTHONPATH=.:exercises/03_minimal_language_model python exercises/04_attention_language_model/09_compare_models.py`

## Next Steps

Implement Transformer Blocks using residual connections, LayerNorm, and feed-forward networks. Then extend the architecture to multiple attention heads and multiple layers before adapting it to Synthetic Linear Regression ICL.