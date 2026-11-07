"""Count tokens with tiktoken (handout 02). Needs network on first run
to download the encoding file."""
import tiktoken

if __name__ == "__main__":
    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode("Tokenization is fun!")
    print(ids, [enc.decode([i]) for i in ids])
    print(len(ids), "tokens")
