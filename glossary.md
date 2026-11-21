# Glossary and Index

Terms and abbreviations used in the handouts, with a pointer to where each is explained. Handout numbers: [00](docs/00-orientation.md) Orientation · [01](docs/01-history.md) History · [02](docs/02-text-to-numbers.md) Tokens and embeddings · [03](docs/03-transformer.md) Transformer · [04](docs/04-generation.md) Generation · [05](docs/05-training.md) Training · [06](docs/06-inference.md) Inference and serving · [07](docs/07-architecture-variants.md) Architecture variants · [08](docs/08-building-with-llms.md) Building with LLMs · [09](docs/09-pitfalls-and-practices.md) Pitfalls and best practices.

Entries marked *(planned)* will be covered in a later handout. This file is updated with every new handout.

## Abbreviations
| Abbr. | Stands for | See |
|---|---|---|
| ALiBi | Attention with Linear Biases (positional method) | 02 |
| AWQ | Activation-aware Weight Quantization | 06 |
| BM25 | classic keyword-ranking function used in search | 08 |
| BERT | Bidirectional Encoder Representations from Transformers | 01, 03 |
| bf16 | bfloat16, a 16-bit number format common in training | 05 |
| BPE | Byte-Pair Encoding (tokenizer algorithm) | 02 |
| DPO | Direct Preference Optimization | 05 |
| FLOPs | Floating-point operations (amount of compute) | 05 |
| FP8 | 8-bit floating point | 06 |
| FSDP | Fully Sharded Data Parallel | 05 |
| GELU | Gaussian Error Linear Unit (activation function) | 03 |
| GGUF | file format for quantized models used by llama.cpp | 06 |
| GPT | Generative Pre-trained Transformer | 01, 03 |
| GPTQ | post-training weight quantization method | 06 |
| GQA | Grouped-Query Attention | 03 |
| HNSW | Hierarchical Navigable Small World (approximate nearest-neighbor index) | 08 |
| KV cache | Key-Value cache | 03 |
| LLM | Large Language Model | 00 |
| LoRA | Low-Rank Adaptation | 05 |
| LSTM | Long Short-Term Memory (recurrent network) | 01 |
| MCP | Model Context Protocol (open protocol for exposing tools and data to LLM apps) | 08 |
| MHA | Multi-Head Attention | 03 |
| MLA | Multi-head Latent Attention (compressed KV cache) | 07 |
| MLP | Multi-Layer Perceptron (the feed-forward part of a block) | 03 |
| MoE | Mixture of Experts | 01, 03, 07 |
| MQA | Multi-Query Attention | 03 |
| NER | Named-Entity Recognition | 03 |
| PPO | Proximal Policy Optimization (RL algorithm) | 05 |
| QLoRA | Quantized LoRA | 05 |
| RAG | Retrieval-Augmented Generation | 05, 08 |
| RLAIF | Reinforcement Learning from AI Feedback | 05 |
| RLHF | Reinforcement Learning from Human Feedback | 01, 05 |
| RLVR | Reinforcement Learning with Verifiable Rewards | 05 |
| RMSNorm | Root-Mean-Square Normalization | 03 |
| RNN | Recurrent Neural Network | 01 |
| RoPE | Rotary Position Embedding | 02 |
| SFT | Supervised Fine-Tuning | 05 |
| SSM | State-Space Model | 07 |
| T5 | Text-to-Text Transfer Transformer | 01, 03 |
| TPOT | Time Per Output Token | 06 |
| TTFT | Time To First Token | 06 |
| ViT | Vision Transformer | 03 |
| ZeRO | Zero Redundancy Optimizer | 05 |

## Math notation (handout 03)
| Symbol | Meaning |
|---|---|
| $n$ | number of tokens in the sequence |
| $d_{\text{model}}$ | length of the token vectors through the model (embedding size, hidden size) |
| $h$ | number of attention heads |
| $d_k$, $d_v$ | length of key/query and value vectors per head, $d_{\text{model}} / h$ |
| $d_{\text{ff}}$ | width of the MLP hidden layer, about $4 \times d_{\text{model}}$ |
| $N$, $D$ | parameters, training tokens (compute $\approx 6ND$ FLOPs, handout 05) |
| $W_Q$, $W_K$, $W_V$ | learned matrices that produce queries, keys, values |
| $T$ | temperature (handout 04) |
| $z_i$ | logit (raw score) of token $i$ (handout 04) |
| $p$, $k$ | cumulative-probability threshold of top-p sampling, and number of tokens kept in top-k sampling (handout 04) |
| $r$, $\alpha$ | LoRA rank and scaling constant (handout 05) |
| $O(n^2)$ | cost growing with the square of the sequence length |

---

## A
- **Activation function:** nonlinearity inside the MLP (e.g., GELU). Without it, stacked layers would collapse into one linear map. 03
- **AdamW:** the standard optimizer for LLMs: adaptive per-weight step sizes, momentum, plus weight decay. 05
- **Alignment:** shaping a model's behavior toward helpful, honest, safe responses (post-training). 05
- **Attention:** mechanism where each token takes a weighted average of other tokens' value vectors, with weights from query-key matches. 01, 03
- **Agent:** an LLM running a tool-use loop and choosing its own next steps; contrast with a *workflow*. 08
- **Automation bias:** over-trusting fluent machine output. 09
- **Active parameters:** the parameters actually used for one token in an MoE model (a fraction of the total). 07
- **Attention sink:** the first few tokens kept permanently in a streaming setup alongside a recent window. 07
- **Attention weights:** the softmax output of the score table; each row sums to 1. 03
- **Autoregressive:** generating one token at a time, feeding each output back as input. 00, 04

## B
- **Backpropagation:** algorithm that computes how each weight should change to reduce the loss. 05
- **Base model:** a pretrained model that only continues text; not yet an assistant. 05
- **Batch size:** number of sequences (or tokens) processed per training step. 05
- **Batching (static / continuous):** serving many requests per model pass. Static waits for the whole batch; continuous swaps finished requests for waiting ones after every step. 06
- **Beam search:** decoding that keeps the best $k$ partial sequences instead of one. 04
- **Benchmark:** a standard test set with automatic scoring (MMLU, HumanEval, ...). 05
- **BPE (Byte-Pair Encoding):** builds a sub-word vocabulary by repeatedly merging the most frequent adjacent pair. 02

## C
- **Causal mask:** blocks attention to later positions so token $i$ only sees tokens at positions $\le i$. 03
- **Chinchilla:** 2022 result: scale parameters and training tokens together (about 20 tokens per parameter). 01, 05
- **Chunking:** splitting documents into overlapping passages for embedding and retrieval. 08
- **Compute-bound / memory-bandwidth-bound:** limited by arithmetic speed (prefill) / by how fast data can be read from memory (decode). 06
- **Constitutional AI:** a method where a model uses written principles to critique and generate preference labels. 05
- **Constrained decoding:** masking illegal tokens at each step to guarantee a grammar or JSON schema. 04
- **Context vector:** in classic seq2seq, the single fixed-size vector (the encoder's last hidden state) summarizing the source sentence; with attention, a new context vector $c_t$ is built from all encoder states at every decoding step. 01
- **Context window:** the maximum number of tokens the model can read at once; its only working memory. 00, 03
- **Contamination:** benchmark questions leaking into training data, inflating scores. 05
- **Cosine similarity:** how closely two vectors point in the same direction (−1 to 1). 02
- **Cross-attention:** attention where queries come from the decoder and keys/values from the encoder. 03
- **Cross-entropy:** the loss $-\log p(\text{correct token})$, averaged over positions. 05

## D
- **Decode (phase):** generating tokens one at a time after the prompt; memory-bandwidth-bound. 03
- **Decoder-only:** Transformer using only causal attention; the dominant LLM family. 03
- **Decoding:** turning logits into the chosen next token (greedy, sampling, beam search...). 04
- **Draft model:** small, fast model that proposes tokens for the large target model to verify (speculative decoding). 06
- **DPO:** preference tuning with a direct loss on (chosen, rejected) pairs, no reward model. 05

## E
- **Eval (evaluation set):** fixed test inputs with automatic checks, run repeatedly to measure an LLM feature statistically. 09
- **Embedding:** a learned vector representing a token (or a whole text, for embedding models). 02
- **Expert parallelism:** placing different MoE experts on different GPUs; the runtime sends each token to the GPU holding its expert and brings the result back (an all-to-all exchange). 05, 07
- **Expert:** one of several parallel MLPs in an MoE layer; the router picks a few per token. 07
- **Embedding model:** model that maps a whole text to one vector, used for search and RAG. 02
- **Emergent abilities:** skills that seem to appear suddenly at larger scale; the sharpness is debated. 05
- **Encoder-decoder:** Transformer with a bidirectional encoder and a causal decoder linked by cross-attention (T5, Whisper). 03
- **Encoder-only:** Transformer with bidirectional attention, for understanding tasks (BERT). 03
- **EOS token (end of sequence):** special token the model emits to say it is finished. 04

## F
- **Few-shot prompting:** giving a few input-output examples in the prompt (in-context learning). 08
- **Fine-tuning:** further training of a pretrained model on a narrower dataset. 05
- **FlashAttention:** exact attention computed in on-chip tiles, avoiding the full $n \times n$ table in slow memory. 06
- **FLOPs:** count of arithmetic operations; training $\approx 6ND$. 05

## G
- **Function calling / tool use:** the model emits a structured request naming a tool and arguments; your code validates and runs it. 08
- **Gates (LSTM, GRU):** small learned switches that decide what a recurrent cell keeps, overwrites or outputs, letting information survive more steps. 01
- **GELU:** a smooth activation function used in many Transformers. 03
- **Gradient / gradient descent:** the direction of steepest loss increase / repeatedly stepping weights the opposite way. 05
- **Gradient clipping:** capping very large gradients to avoid unstable training. 05
- **Greedy decoding:** always pick the most probable token. 04
- **GQA (grouped-query attention):** several query heads share one set of keys/values, shrinking the KV cache. 03

## H
- **Hallucination:** fluent but false or unsupported output; a result of optimizing for plausible text. Mitigate by grounding, verification and an "I don't know" path. 00, 09
- **Head (attention head):** one independent attention computation on a slice of the vector. 03
- **Hybrid search:** combining vector (semantic) search with keyword search. 08
- **Hybrid model:** architecture interleaving attention layers with SSM (and sometimes MoE) layers. 07
- **Guardrails:** checks and limits around the model (input/output validation, moderation, tool permissions). 09
- **Hidden state:** the vector a network carries internally. In an RNN, the fixed-size memory $h_t$ passed from one step to the next; in a Transformer, a token's vector after at least one block (size $d_{\text{model}}$). 01, 03

## I
- **Indirect prompt injection:** malicious instructions hidden in content the model processes (web page, email, document, tool result). 09
- **In-context learning:** adapting to a task from examples in the prompt, with no weight update. 01
- **Inference:** running a trained model to produce output; weights do not change. 00
- **Instruction tuning:** supervised fine-tuning on (instruction, response) pairs. 01, 05

## K
- **Key (K):** vector a token exposes for others to match against. 03
- **Knowledge cutoff:** the date after which the training data contains nothing. 05
- **KV cache:** stored keys and values of earlier tokens, reused so each new token is cheap to compute. 03

## L
- **Layer normalization / RMSNorm:** rescaling vectors to keep numbers in a stable range. 03
- **Least privilege:** giving a task only the tools and credentials it needs. 08, 09
- **"Lethal trifecta":** private data + untrusted content + external communication in one agent: a data-leak risk. 09
- **LLM-as-judge:** using a model to grade other model outputs; biased, so calibrate with humans. 05, 09
- **Learning rate:** step size of the weight updates; usually warmed up then decayed. 05
- **Logits:** raw, unnormalized scores, one per vocabulary token. 04
- **Logprobs:** log-probabilities of tokens, returned by some APIs. 04
- **LoRA:** fine-tuning via small low-rank matrices $BA$ added to frozen weights. 05
- **Loss:** a number measuring how wrong the model is; training minimizes it. 05

## M
- **Mamba:** an SSM architecture whose parameters depend on the input (selective SSM). 07
- **Masked language modeling:** pretraining objective for encoders: fill in hidden words. 03
- **Min-p:** sampling filter keeping tokens above a fraction of the top token's probability. 04
- **Memory-bandwidth-bound:** see *compute-bound*. 06
- **Mixed precision:** computing in 16-bit while keeping some values in 32-bit. 05
- **MLP (feed-forward):** per-token expand-activate-shrink network in each block. 03
- **MoE (Mixture of Experts):** a router sends each token to a few expert MLPs: compute follows active parameters, memory follows total. 01, 03, 07
- **Multi-head attention:** several attention heads in parallel, outputs concatenated. 03
- **Multimodal:** accepting inputs beyond text (images, audio) by encoding them into vectors the LLM attends over. 07

## N
- **Next-token prediction:** the core objective of decoder-only LLMs. 00, 05
- **Nucleus sampling:** see *top-p*. 04

## O
- **Optimizer:** the algorithm that updates weights from gradients (e.g., AdamW). 05
- **Over-training:** training a model on far more tokens than compute-optimal, to make a small model strong and cheap to serve. 05

## P
- **PagedAttention:** storing the KV cache in small fixed-size blocks to avoid fragmentation and fit more requests (vLLM). 06
- **Parameters (weights):** the learned numbers of the model. 00
- **Perplexity:** $e^{\text{loss}}$; roughly how many tokens the model is torn between. 05
- **Positional encoding:** information telling the model each token's position (sinusoidal, learned, RoPE, ALiBi). 02
- **Prompt injection:** text in the data acting as instructions to the model; defend with permissions, not wording. 09
- **Position interpolation:** scaling positions down so a longer text fits the trained range (RoPE context extension); NTK-aware scaling and YaRN are related methods. 07
- **PPO:** reinforcement-learning algorithm used in classic RLHF. 05
- **Prefill:** processing the whole prompt in parallel before generating. 03
- **Pretraining:** the large self-supervised first stage on raw text. 05
- **Prefix caching:** reusing the KV cache for a shared prompt prefix across requests; sold by providers as prompt caching. 06
- **Prompt chaining:** splitting a task into several LLM calls, each feeding the next. 08
- **Prompt caching:** reusing the KV cache of an unchanged prompt prefix to save cost and latency. 03, 06

## Q
- **QLoRA:** LoRA on top of a base model stored in 4-bit precision. 05
- **Quantization:** storing weights in fewer bits (e.g., int8, int4) plus scale factors, to save memory and speed up decoding. Per-group scales reduce error. 05, 06
- **Query (Q):** vector a token uses to search for relevant tokens. 03

## R
- **RAG:** retrieving relevant documents and adding them to the prompt, so answers are grounded in your data. 05, 08
- **Reranking:** reordering or filtering retrieved passages with a stronger model before prompting. 08
- **Re-prefill:** running prefill again for a new or shortened prompt, which rebuilds the KV cache from scratch (only an unchanged prefix can be reused). 03, 06
- **Reasoning model:** model trained (often with RL) to produce long intermediate reasoning before answering. 01, 05
- **Router (gating network):** small layer in an MoE block that scores experts for each token. 07
- **Regression gate:** a CI check that fails when eval scores drop below a baseline. 09
- **Residual connection:** adding a layer's output to its input. 03
- **Reward model:** model trained on human comparisons to score answers. 05
- **RLHF / RLAIF / RLVR:** reinforcement learning with human feedback / AI feedback / automatically verifiable rewards. 05
- **RNN / LSTM:** recurrent networks that read tokens one at a time, applying the same cell to the current token and the previous hidden state to produce a new hidden state. 01
- **RoPE:** positional method that rotates query/key vectors by position. 02

## S
- **Sampling:** drawing the next token randomly from the probability distribution. 04
- **Scaling laws:** loss falls predictably as parameters, data and compute grow. 01, 05
- **Self-attention:** attention where queries, keys and values all come from the same sequence. 03
- **Self-supervised learning:** learning from labels derived from the data itself (the next token). 05
- **Seq2seq:** setup with an encoder RNN that reads the source into a context vector and a decoder RNN that writes the target; attention later let the decoder look at all encoder states. 01
- **SFT:** supervised fine-tuning. 05
- **Sliding-window attention:** each token attends only to the last $w$ tokens; reach grows with depth. 07
- **Softmax:** turns scores into positive probabilities summing to 1: $\mathrm{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$. 03, 04
- **Special tokens:** vocabulary entries marking structure (end of text, roles, tool calls). 02
- **Speculative decoding:** a draft model proposes several tokens and the target verifies them in one pass; output distribution is unchanged. 06
- **State-space model (SSM):** sequence layer with a fixed-size state updated per token; recurrent for generation, convolution for training. 07
- **Stop sequence:** a string that ends generation when produced. 04
- **Sycophancy:** tendency to agree with the user, partly an artifact of preference tuning. 05

## T
- **System prompt:** standing instructions given to the model before the conversation. 08
- **Teacher forcing:** training with the true previous tokens as input, so every position yields a loss in one pass. 05
- **Temperature:** divides logits before softmax; low = focused, high = random. 04
- **Token / tokenizer:** a sub-word piece with an integer ID / the component that splits text into them. 02
- **Throughput:** total tokens per second across all users of a server. 06
- **Top-k / top-p:** sampling filters keeping the $k$ best tokens / the smallest set with cumulative probability $\ge p$. 04
- **Transformer:** architecture built from attention and MLP blocks (2017). 01, 03

## U–Z
- **Unembedding (output projection):** final matrix turning a vector into one logit per vocabulary token. 03
- **Unrolling:** drawing a recurrent network as one copy of the cell per time step, to follow how information flows (the copies share the same weights). 01
- **Value (V):** the information a token contributes when attended to. 03
- **Vanishing gradient:** the training signal (and the influence of old inputs) shrinking step by step in a recurrent network, so long-range links are hard to learn. 01
- **Vector database / index:** store that finds the nearest embeddings to a query vector. 08
- **Vision encoder / projector:** the encoder that turns image patches into vectors / the small network mapping them to the LLM's $d_{\text{model}}$. 07
- **Vocabulary:** the set of all tokens a model knows (about 30k to 250k+). 02
- **word2vec:** 2013 method that learns word embeddings in a separate training step (skip-gram with negative sampling); LLMs instead learn embeddings jointly with the network. 01, 02
- **Workflow:** a fixed sequence of steps in your code, some of which call an LLM; contrast with an *agent*. 08
- **Retry with exponential backoff:** retrying failed calls after growing delays (plus jitter). 09
- **Weight decay:** a small penalty on large weights, part of AdamW. 05
- **ZeRO / FSDP:** techniques that shard optimizer state, gradients and weights across GPUs. 05
