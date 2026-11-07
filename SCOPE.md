# Project Scope: LLMs Explained for Software Developers

## Audience
Working software developers who know code and APIs but not necessarily ML or heavy math.

## Goal
After reading, a developer can explain how an LLM works end to end and reason about its behavior: why it hallucinates, why context length matters, why temperature, prompts and tooling behave the way they do.

**Non-goals:** research-level derivations, training a frontier model, vendor-specific tutorials.

## Decisions (agreed)
| Topic | Decision |
|---|---|
| Math depth | Intuition-first, minimal formulas. Scaled dot-product attention is shown and explained explicitly. |
| Examples | Python as the main language. JavaScript only for application-level topics (RAG, tool use). |
| Application layer | One handout (module 8), not a full track. |
| Language | English |
| Delivery | Plain files in this folder (Markdown, Mermaid, SVG, code). No site generator for now. |
| Model taxonomy | Module 3 classifies the most popular models by architecture type (encoder-only, decoder-only, encoder-decoder, and variants such as MoE). |

## Modules
| # | Module | Covers |
|---|---|---|
| 0 | Orientation | Mental model, pipeline overview, how to use the handouts |
| 1 | History | n-grams → RNN/LSTM → seq2seq and attention → Transformer → GPT/BERT/T5 → scaling laws → instruction tuning/RLHF → open models, MoE, reasoning models |
| 2 | Text to numbers | Tokenization (BPE and similar), embeddings, positional encoding |
| 3 | The Transformer | Attention (scaled dot-product), multi-head, MLP blocks, residuals, normalization, KV cache; **model taxonomy**: encoder-only vs decoder-only vs encoder-decoder, with a table of popular models |
| 4 | Generation | Logits, softmax, sampling (temperature, top-k, top-p), beam search, structured output |
| 5 | Training | Pretraining, data, loss, scaling laws, fine-tuning, LoRA, RLHF/DPO, evaluation |
| 6 | Inference and serving | Quantization, batching, speculative decoding, latency vs cost, context limits |
| 7 | Architecture variants | MoE, long-context tricks, multimodal, state-space models (Mamba) |
| 8 | Building with LLMs | One handout: prompting patterns, RAG, tool use, agents, evals, guardrails |
| 9 | Pitfalls and best practices | Hallucination, prompt injection, non-determinism, testing, cost control |
| 10 | Further reading | Papers, courses, repos |

## Handout template
Every handout uses the same sections: **Key idea → Intuition → How it works → Pseudo-code → Example → Pitfalls → Summary → Further reading**. Target 5–15 minutes of reading.

## Conventions
- **Diagrams:** Mermaid in the Markdown (renders on GitHub and most editors). Hand-written SVG in `diagrams/` where Mermaid is too limiting.
- **Code:** short, runnable, standalone files under `examples/python/` and `examples/js/`, referenced from the handouts.
- **Accuracy rule:** simplify the explanation, never the facts. Flag simplifications explicitly. Mark fast-changing facts (model names, sizes, dates) as "as of <date>".

## Layout
```
llm-explained-handouts/
  README.md        index and reading paths
  SCOPE.md         this file
  glossary.md
  docs/            00-orientation.md … 10-further-reading.md
  diagrams/
  examples/python/  examples/js/
```

## Status
- [x] Scope agreed
- [x] 00 Orientation, 01 History (style sample, review before continuing)
- [x] 02 Text to numbers (draft)
- [ ] 03–10
