# 10 · Further Reading

Curated next steps, grouped by purpose. Papers link to arXiv. Resources change, so if a link breaks, search by title.

## Pick a path
| You want to... | Go to |
|---|---|
| Build intuition quickly | [Start here](#start-here-intuition-first) |
| Understand the Transformer deeply | [Learn by building](#learn-by-building) and [Foundations](#foundations-papers-in-reading-order) |
| Ship LLM features | [Building and operating](#building-and-operating-llm-applications) |
| Run or optimize models | [Inference and efficiency](#inference-and-efficiency) |
| Keep up with the field | [Staying current](#staying-current) |

---

## Start here (intuition first)
- Andrej Karpathy, [*Intro to Large Language Models*](https://www.youtube.com/watch?v=zjkBMFhNj_g) and [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI) (videos): the whole pipeline in about an hour or three.
- Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/), [*The Illustrated GPT-2*](https://jalammar.github.io/illustrated-gpt2/) and [*The Illustrated Word2vec*](https://jalammar.github.io/illustrated-word2vec/): the best visual walk-throughs of attention and embeddings.
- Jay Alammar, [*Visualizing A Neural Machine Translation Model*](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/): animated seq2seq and attention, the step before the Transformer.
- 3Blue1Brown, [*Neural networks* series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (its chapters on GPT and transformers): animated explanations of embeddings and attention.
- [*LLM Visualization*](https://bbycroft.net/llm): an interactive 3D walkthrough of a GPT model (a tiny 85,000-parameter one, plus GPT-2 and GPT-3 views), from embedding through attention and the MLP to softmax. Best on a desktop browser.
- Hugging Face, [*LLM Course*](https://huggingface.co/learn): hands-on, with code.

## Learn by building
- Karpathy, [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html) (videos), especially [*Let's build GPT: from scratch, in code*](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE).
- Code to read: [nanoGPT](https://github.com/karpathy/nanoGPT) (a small, readable GPT), [minbpe](https://github.com/karpathy/minbpe) (a BPE tokenizer in about 300 lines).
- Sebastian Raschka, [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) (book).
- Stanford courses: [CS224N](https://web.stanford.edu/class/cs224n/) (NLP with deep learning) and [CS336](https://cs336.stanford.edu/) (Language Modeling from Scratch). Lecture materials are public.

---

## Foundations: papers in reading order
Matching the handouts. Start with the first and the starred ones. Skim the rest as needed.

### 01 History and 03 Transformer
| Paper | Why read it |
|---|---|
| Bahdanau et al., [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473) (2014) | Where attention started |
| Sutskever et al., [*Sequence to Sequence Learning with Neural Networks*](https://arxiv.org/abs/1409.3215) (2014) | The encoder-decoder RNN setup |
| Luong et al., [*Effective Approaches to Attention-based Neural Machine Translation*](https://arxiv.org/abs/1508.04025) (2015) | Simpler (dot-product) attention scores |
| ⭐ Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017) | The Transformer |
| Devlin et al., [*BERT*](https://arxiv.org/abs/1810.04805) (2018) | Encoder-only pretraining |
| Raffel et al., [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5)](https://arxiv.org/abs/1910.10683) (2019) | Encoder-decoder, text-to-text |
| Brown et al., [*Language Models are Few-Shot Learners* (GPT-3)](https://arxiv.org/abs/2005.14165) (2020) | In-context learning at scale |
| Touvron et al., [*LLaMA*](https://arxiv.org/abs/2302.13971) (2023) | The open-weight wave |

### 02 Tokens, embeddings, positions
| Paper | Why read it |
|---|---|
| Sennrich et al., [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909) (2015) | BPE for language models |
| Su et al., [*RoFormer* (RoPE)](https://arxiv.org/abs/2104.09864) (2021) | Rotary position embeddings |

### 04 Generation
| Paper | Why read it |
|---|---|
| Holtzman et al., [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751) (2019) | Top-p (nucleus) sampling |
| Wei et al., [*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*](https://arxiv.org/abs/2201.11903) (2022) | Why "think step by step" helps |

### 05 Training
| Paper | Why read it |
|---|---|
| Kaplan et al., [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361) (2020) | Loss falls predictably with scale |
| ⭐ Hoffmann et al., [*Training Compute-Optimal Large Language Models* (Chinchilla)](https://arxiv.org/abs/2203.15556) (2022) | Balance parameters and data |
| ⭐ Ouyang et al., [*Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155) (2022) | Instruction tuning and RLHF |
| Bai et al., [*Constitutional AI*](https://arxiv.org/abs/2212.08073) (2022) | Feedback from AI guided by principles |
| Rafailov et al., [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290) (2023) | Preference tuning without RL |
| Hu et al., [*LoRA*](https://arxiv.org/abs/2106.09685) (2021) | Cheap fine-tuning |
| Dettmers et al., [*QLoRA*](https://arxiv.org/abs/2305.14314) (2023) | LoRA on a 4-bit base |
| DeepSeek-AI, [*DeepSeek-R1*](https://arxiv.org/abs/2501.12948) (2025) | Reinforcement learning for reasoning |
| Schaeffer et al., [*Are Emergent Abilities of Large Language Models a Mirage?*](https://arxiv.org/abs/2304.15004) (2023) | A critical view of "emergence" |

### 06 Inference and serving
| Paper | Why read it |
|---|---|
| Shazeer, [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150) (2019) | Multi-query attention |
| Ainslie et al., [*GQA*](https://arxiv.org/abs/2305.13245) (2023) | Grouped-query attention |
| Dao et al., [*FlashAttention*](https://arxiv.org/abs/2205.14135) (2022) | Exact, memory-efficient attention |
| Kwon et al., [*Efficient Memory Management for LLM Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (2023) | The idea behind vLLM |
| Leviathan et al., [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192) (2022) | Draft-and-verify decoding |
| Frantar et al., [*GPTQ*](https://arxiv.org/abs/2210.17323) (2022); Lin et al., [*AWQ*](https://arxiv.org/abs/2306.00978) (2023) | Weight quantization |

### 07 Architecture variants
| Paper | Why read it |
|---|---|
| Shazeer et al., [*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538) (2017) | MoE for deep nets |
| Jiang et al., [*Mixtral of Experts*](https://arxiv.org/abs/2401.04088) (2024) | A widely used open MoE |
| DeepSeek-AI, [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437) (2024) | MoE plus multi-head latent attention at scale |
| Chen et al., [*Extending Context Window via Positional Interpolation*](https://arxiv.org/abs/2306.15595) (2023); Peng et al., [*YaRN*](https://arxiv.org/abs/2309.00071) (2023) | Longer contexts with RoPE |
| Beltagy et al., [*Longformer*](https://arxiv.org/abs/2004.05150) (2020); Jiang et al., [*Mistral 7B*](https://arxiv.org/abs/2310.06825) (2023) | Sliding-window attention |
| Liu et al., [*Visual Instruction Tuning* (LLaVA)](https://arxiv.org/abs/2304.08485) (2023) | Vision plus LLM |
| Gu and Dao, [*Mamba*](https://arxiv.org/abs/2312.00752) (2023); Lieber et al., [*Jamba*](https://arxiv.org/abs/2403.19887) (2024) | State-space models and hybrids |

### 08 and 09 Building, safety, evaluation
| Paper | Why read it |
|---|---|
| Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401) (2020) | The original RAG paper |
| Yao et al., [*ReAct*](https://arxiv.org/abs/2210.03629) (2022) | Reasoning plus acting with tools |
| Liu et al., [*Lost in the Middle*](https://arxiv.org/abs/2307.03172) (2023) | Models use long context unevenly |
| Zheng et al., [*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*](https://arxiv.org/abs/2306.05685) (2023) | Strengths and biases of model judges |
| Ji et al., [*Survey of Hallucination in Natural Language Generation*](https://arxiv.org/abs/2202.03629) (2022) | A map of hallucination research |

---

## Building and operating LLM applications
- Anthropic, [*Building effective agents*](https://www.anthropic.com/engineering/building-effective-agents): workflows vs agents, with patterns.
- [Model Context Protocol](https://modelcontextprotocol.io/): documentation and specification.
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/): the standard security checklist.
- NIST, [*AI Risk Management Framework*](https://www.nist.gov/itl/ai-risk-management-framework): governance and risk vocabulary.
- [Simon Willison's blog](https://simonwillison.net/): practical writing on prompt injection and the ["lethal trifecta"](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/).
- Your provider's documentation (for example [Anthropic](https://platform.claude.com/docs/en/home) or [OpenAI](https://developers.openai.com/api/docs)): prompting guides, tool use, structured outputs, evals, data handling.

## Inference and efficiency
- [vLLM](https://github.com/vllm-project/vllm) (continuous batching, PagedAttention) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (local inference, GGUF quantization): read the docs and issues to see real trade-offs.
- Hugging Face blog, *How to generate text: using different decoding methods*: decoding in code.

## Staying current
- arXiv categories [cs.CL](https://arxiv.org/list/cs.CL/recent) (computation and language) and [cs.LG](https://arxiv.org/list/cs.LG/recent) (machine learning), plus Hugging Face's [daily papers](https://huggingface.co/papers) page.
- Model cards and technical reports from labs. They are the primary source for architecture and training facts, and the best antidote to rumor.
- Leaderboards help to shortlist models, but verify on your own evals (handouts 05 and 09).

## How to read a paper efficiently
1. Read the abstract, then the figures and tables (they carry the claims).
2. Read the introduction and conclusion.
3. Read the method only as far as you need to.
4. Ask: what is the baseline, what is the data, and would this hold for my use case?
5. Prefer official technical reports and reproductions over summaries.

Back to the [index](../README.md) · [Glossary](../glossary.md).
