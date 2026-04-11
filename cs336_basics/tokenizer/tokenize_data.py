import numpy as np
from cs336_basics.tokenizer.tokenizer import BPETokenizer

def tokenize_data(input_path: str, output_path: str, vocab_path: str, merges_path: str):
    bpe = BPETokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    with open(input_path, "r") as f:
        text = f.read()
    ids = bpe.encode(text)
    np.array(ids, dtype=np.uint16).tofile(output_path)


if __name__ == "__main__":
    tokenize_data("data/raw_data/TinyStoriesV2-GPT4-train.txt", "data/tokenized_data/TinyStoriesV2-GPT4-train.bin", "data/TinyStoriesV2-GPT4-vocab.json", "data/TinyStoriesV2-GPT4-merges.txt")
    tokenize_data("data/raw_data/TinyStoriesV2-GPT4-valid.txt", "data/tokenized_data/TinyStoriesV2-GPT4-valid.bin", "data/TinyStoriesV2-GPT4-vocab.json", "data/TinyStoriesV2-GPT4-merges.txt")
