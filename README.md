# LLMs Explained: Handouts for Developers

Short, accurate, intuition-first handouts on how Large Language Models work, written for software developers. See [SCOPE.md](SCOPE.md) for goals and conventions.

**Unknown term or abbreviation?** See the [Glossary and Index](glossary.md).

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
| 05 | [Training](docs/05-training.md) | draft |
| 06 | [Inference and serving](docs/06-inference.md) | draft |
| 07 | [Architecture variants](docs/07-architecture-variants.md) | draft |
| 08 | [Building with LLMs](docs/08-building-with-llms.md) | draft |
| 09 | [Pitfalls and best practices](docs/09-pitfalls-and-practices.md) | draft |
| 10 | [Further reading](docs/10-further-reading.md) | draft |

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
JavaScript examples (Node 20+, no dependencies):
```bash
npm test
node examples/js/tool_loop.js
```
The tiktoken example downloads its encoding file on first use (network needed); its test skips when offline.

External links in the handouts can be re-verified at any time (needs network):
```bash
python tools/check_links.py
```

Diagrams are written in [Mermaid](https://mermaid.js.org/). View them on GitHub, or in an editor with a Mermaid preview (VS Code: "Markdown Preview Mermaid Support").
