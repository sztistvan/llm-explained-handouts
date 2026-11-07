# LLMs Explained: Handouts for Developers

Short, accurate, intuition-first handouts on how Large Language Models work, written for software developers. See [SCOPE.md](SCOPE.md) for goals and conventions.

## Reading paths
- **Fast track (the mental model):** 00 → 03 → 04 → 08 → 09
- **Full track:** 00 → 10 in order
- **"I only build apps":** 00 → 04 → 08 → 09

## Modules
| # | Handout | Status |
|---|---|---|
| 00 | [Orientation](docs/00-orientation.md) | draft |
| 01 | [History](docs/01-history.md) | draft |
| 02 | [Text to numbers: tokens and embeddings](docs/02-text-to-numbers.md) | draft |
| 03 | [The Transformer and model taxonomy](docs/03-transformer.md) | draft |
| 04 | [Generation and sampling](docs/04-generation.md) | draft |
| 05 | Training | planned |
| 06 | Inference and serving | planned |
| 07 | Architecture variants | planned |
| 08 | Building with LLMs | planned |
| 09 | Pitfalls and best practices | planned |
| 10 | Further reading | planned |

## Running the code examples
Requires Python 3.10+ (developed on 3.13).

```bash
python -m venv .venv
```
Activate it (`source .venv/bin/activate` on macOS/Linux, `.venv\Scripts\activate` on Windows), then:
```bash
pip install -r requirements-dev.txt
pytest
python examples/python/toy_bpe.py
```
The tiktoken example downloads its encoding file on first use (network needed); its test skips when offline.

Diagrams are written in [Mermaid](https://mermaid.js.org/). View them on GitHub, or in an editor with a Mermaid preview (VS Code: "Markdown Preview Mermaid Support").
