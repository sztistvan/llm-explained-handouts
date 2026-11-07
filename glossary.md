# Glossary and Index

Terms and abbreviations used in the handouts, with a pointer to where each is explained. Handout numbers: [00](docs/00-orientation.md) Orientation · [01](docs/01-history.md) History · [02](docs/02-text-to-numbers.md) Tokens and embeddings · [03](docs/03-transformer.md) Transformer · [04](docs/04-generation.md) Generation · [05](docs/05-training.md) Training · [06](docs/06-inference.md) Inference and serving · [07](docs/07-architecture-variants.md) Architecture variants.

Entries marked *(planned)* will be covered in a later handout. This file is updated with every new handout.

## Abbreviations
| Abbr. | Stands for | See |
|---|---|---|
| ALiBi | Attention with Linear Biases (positional method) | 02 |
| AWQ | Activation-aware Weight Quantization | 06 |
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
| KV cache | Key-Value cache | 03 |
| LLM | Large Language Model | 00 |
| LoRA | Low-Rank Adaptation | 05 |
| LSTM | Long Short-Term Memory (recurrent network) | 01 |
| MHA | Multi-Head Attention | 03 |
| MLA | Multi-head Latent Attention (compressed KV cache) | 07 |
| MLP | Multi-Layer Perceptron (the feed-forward part of a block) | 03 |
| MoE | Mixture of Experts | 01, 03, 07 |
| MQA | Multi-Query Attention | 03 |
| NER | Named-Entity Recognition | 03 |
| PPO | Proximal Policy Optimization (RL algorithm) | 05 |
| QLoRA | Quantized LoRA | 05 |
| RAG | Retrieval-Augmented Generation | 05, 08 *(planned)* |
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
| `n` | number of tokens in the sequence |
| `d_model` | length of the token vectors through the model (embedding size, hidden size) |
| `h` | number of attention heads |
| `d_k`, `d_v` | length of key/query and value vectors per head, `d_model / h` |
| `d_ff` | width of the MLP hidden layer, about `4 × d_model` |
| `N`, `D` | parameters, training tokens (compute ≈ `6·N·D` FLOPs, handout 05) |
| `W_Q`, `W_K`, `W_V` | learned matrices that produce queries, keys, values |
| `T` | temperature (handout 04) |

---

## A
- **Activation function:** nonlinearity inside the MLP (e.g., GELU). Without it, stacked layers would collapse into one linear map. 03
- **AdamW:** the standard optimizer for LLMs: adaptive per-weight step sizes, momentum, plus weight decay. 05
- **Alignment:** shaping a model's behavior toward helpful, honest, safe responses (post-training). 05
- **Attention:** mechanism where each token takes a weighted average of other tokens' value vectors, with weights from query-key matches. 01, 03
- **Active parameters:** the parameters actually used for one token in an MoE model (a fraction of the total). 07
- **Attention sink:** the first few tokens kept permanently in a streaming setup alongside a recent window. 07
- **Attention weights:** the softmax output of the score table; each row sums to 1. 03
- **Autoregressive:** generating one token at a time, feeding each output back as input. 00, 04

## B
- **Backpropagation:** algorithm that computes how each weight should change to reduce the loss. 05
- **Base model:** a pretrained model that only continues text; not yet an assistant. 05
- **Batch size:** number of sequences (or tokens) processed per training step. 05
- **Batching (static / continuous):** serving many requests per model pass. Static waits for the whole batch; continuous swaps finished requests for waiting ones after every step. 06
- **Beam search:** decoding that keeps the best *k* partial sequences instead of one. 04
- **Benchmark:** a standard test set with automatic scoring (MMLU, HumanEval, ...). 05
- **BPE (Byte-Pair Encoding):** builds a sub-word vocabulary by repeatedly merging the most frequent adjacent pair. 02

## C
- **Causal mask:** blocks attention to later positions so token *i* only sees tokens ≤ *i*. 03
- **Chinchilla:** 2022 result: scale parameters and training tokens together (about 20 tokens per parameter). 01, 05
- **Compute-bound / memory-bandwidth-bound:** limited by arithmetic speed (prefill) / by how fast data can be read from memory (decode). 06
- **Constitutional AI:** a method where a model uses written principles to critique and generate preference labels. 05
- **Constrained decoding:** masking illegal tokens at each step to guarantee a grammar or JSON schema. 04
- **Context window:** the maximum number of tokens the model can read at once; its only working memory. 00, 03
- **Contamination:** benchmark questions leaking into training data, inflating scores. 05
- **Cosine similarity:** how closely two vectors point in the same direction (−1 to 1). 02
- **Cross-attention:** attention where queries come from the decoder and keys/values from the encoder. 03
- **Cross-entropy:** the loss `-log p(correct token)`, averaged over positions. 05

## D
- **Decode (phase):** generating tokens one at a time after the prompt; memory-bandwidth-bound. 03
- **Decoder-only:** Transformer using only causal attention; the dominant LLM family. 03
- **Decoding:** turning logits into the chosen next token (greedy, sampling, beam search...). 04
- **Draft model:** small, fast model that proposes tokens for the large target model to verify (speculative decoding). 06
- **DPO:** preference tuning with a direct loss on (chosen, rejected) pairs, no reward model. 05

## E
- **Embedding:** a learned vector representing a token (or a whole text, for embedding models). 02
- **Expert:** one of several parallel MLPs in an MoE layer; the router picks a few per token. 07
- **Embedding model:** model that maps a whole text to one vector, used for search and RAG. 02
- **Emergent abilities:** skills that seem to appear suddenly at larger scale; the sharpness is debated. 05
- **Encoder-decoder:** Transformer with a bidirectional encoder and a causal decoder linked by cross-attention (T5, Whisper). 03
- **Encoder-only:** Transformer with bidirectional attention, for understanding tasks (BERT). 03
- **EOS token (end of sequence):** special token the model emits to say it is finished. 04

## F
- **Fine-tuning:** further training of a pretrained model on a narrower dataset. 05
- **FlashAttention:** exact attention computed in on-chip tiles, avoiding the full `n × n` table in slow memory. 06
- **FLOPs:** count of arithmetic operations; training ≈ `6·N·D`. 05

## G
- **GELU:** a smooth activation function used in many Transformers. 03
- **Gradient / gradient descent:** the direction of steepest loss increase / repeatedly stepping weights the opposite way. 05
- **Gradient clipping:** capping very large gradients to avoid unstable training. 05
- **Greedy decoding:** always pick the most probable token. 04
- **GQA (grouped-query attention):** several query heads share one set of keys/values, shrinking the KV cache. 03

## H
- **Hallucination:** fluent but false or unsupported output; a result of optimizing for plausible text. 00, 09 *(planned)*
- **Head (attention head):** one independent attention computation on a slice of the vector. 03
- **Hybrid model:** architecture interleaving attention layers with SSM (and sometimes MoE) layers. 07
- **Hidden state:** a token's vector inside the network after at least one block (size `d_model`). 03

## I
- **In-context learning:** adapting to a task from examples in the prompt, with no weight update. 01
- **Inference:** running a trained model to produce output; weights do not change. 00
- **Instruction tuning:** supervised fine-tuning on (instruction, response) pairs. 01, 05

## K
- **Key (K):** vector a token exposes for others to match against. 03
- **Knowledge cutoff:** the date after which the training data contains nothing. 05
- **KV cache:** stored keys and values of earlier tokens, reused so each new token is cheap to compute. 03

## L
- **Layer normalization / RMSNorm:** rescaling vectors to keep numbers in a stable range. 03
- **Learning rate:** step size of the weight updates; usually warmed up then decayed. 05
- **Logits:** raw, unnormalized scores, one per vocabulary token. 04
- **Logprobs:** log-probabilities of tokens, returned by some APIs. 04
- **LoRA:** fine-tuning via small low-rank matrices `B·A` added to frozen weights. 05
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
- **Perplexity:** `exp(loss)`; roughly how many tokens the model is torn between. 05
- **Positional encoding:** information telling the model each token's position (sinusoidal, learned, RoPE, ALiBi). 02
- **Position interpolation:** scaling positions down so a longer text fits the trained range (RoPE context extension); NTK-aware scaling and YaRN are related methods. 07
- **PPO:** reinforcement-learning algorithm used in classic RLHF. 05
- **Prefill:** processing the whole prompt in parallel before generating. 03
- **Pretraining:** the large self-supervised first stage on raw text. 05
- **Prefix caching:** reusing the KV cache for a shared prompt prefix across requests; sold by providers as prompt caching. 06
- **Prompt caching:** reusing the KV cache of an unchanged prompt prefix to save cost and latency. 03, 06

## Q
- **QLoRA:** LoRA on top of a base model stored in 4-bit precision. 05
- **Quantization:** storing weights in fewer bits (e.g., int8, int4) plus scale factors, to save memory and speed up decoding. Per-group scales reduce error. 05, 06
- **Query (Q):** vector a token uses to search for relevant tokens. 03

## R
- **RAG:** retrieving relevant documents and adding them to the prompt. 05, 08 *(planned)*
- **Reasoning model:** model trained (often with RL) to produce long intermediate reasoning before answering. 01, 05
- **Router (gating network):** small layer in an MoE block that scores experts for each token. 07
- **Residual connection:** adding a layer's output to its input. 03
- **Reward model:** model trained on human comparisons to score answers. 05
- **RLHF / RLAIF / RLVR:** reinforcement learning with human feedback / AI feedback / automatically verifiable rewards. 05
- **RNN / LSTM:** recurrent networks that read tokens sequentially with a hidden state. 01
- **RoPE:** positional method that rotates query/key vectors by position. 02

## S
- **Sampling:** drawing the next token randomly from the probability distribution. 04
- **Scaling laws:** loss falls predictably as parameters, data and compute grow. 01, 05
- **Self-attention:** attention where queries, keys and values all come from the same sequence. 03
- **Self-supervised learning:** learning from labels derived from the data itself (the next token). 05
- **Seq2seq:** encoder-decoder RNN setup for translation-like tasks. 01
- **SFT:** supervised fine-tuning. 05
- **Sliding-window attention:** each token attends only to the last *w* tokens; reach grows with depth. 07
- **Softmax:** turns scores into positive probabilities summing to 1. 03, 04
- **Special tokens:** vocabulary entries marking structure (end of text, roles, tool calls). 02
- **Speculative decoding:** a draft model proposes several tokens and the target verifies them in one pass; output distribution is unchanged. 06
- **State-space model (SSM):** sequence layer with a fixed-size state updated per token; recurrent for generation, convolution for training. 07
- **Stop sequence:** a string that ends generation when produced. 04
- **Sycophancy:** tendency to agree with the user, partly an artifact of preference tuning. 05

## T
- **Teacher forcing:** training with the true previous tokens as input, so every position yields a loss in one pass. 05
- **Temperature:** divides logits before softmax; low = focused, high = random. 04
- **Token / tokenizer:** a sub-word piece with an integer ID / the component that splits text into them. 02
- **Throughput:** total tokens per second across all users of a server. 06
- **Top-k / top-p:** sampling filters keeping the *k* best tokens / the smallest set with cumulative probability ≥ *p*. 04
- **Transformer:** architecture built from attention and MLP blocks (2017). 01, 03

## U–Z
- **Unembedding (output projection):** final matrix turning a vector into one logit per vocabulary token. 03
- **Value (V):** the information a token contributes when attended to. 03
- **Vision encoder / projector:** the encoder that turns image patches into vectors / the small network mapping them to the LLM's `d_model`. 07
- **Vocabulary:** the set of all tokens a model knows (about 30k to 250k+). 02
- **Weight decay:** a small penalty on large weights, part of AdamW. 05
- **ZeRO / FSDP:** techniques that shard optimizer state, gradients and weights across GPUs. 05
