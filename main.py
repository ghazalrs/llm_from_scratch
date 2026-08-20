import re
from tokenizer import SimpleTokenizer
import tiktoken

if __name__ == "__main__":
    # Tokenize text
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    # Create a vocabulary
    all_tokens = sorted(list(set(preprocessed)))

    # Add two special tokens
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    vocab = {token: integer for integer, token in enumerate(all_tokens)}

    # Use SimpleTokenizer to instantiate new tokenizer object
    tokenizer = SimpleTokenizer(vocab)

    text = """"It's the last he painted, you know,"
               Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))

    # Instantiate the BPE tokenizer from tiktoken
    text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )
    bpe_tokenizer = tiktoken.get_encoding("gpt2")
    integers = bpe_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)
    print(bpe_tokenizer.decode(integers))