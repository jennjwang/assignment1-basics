import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from cs336_basics.tokenizer.tokenizer import BPETokenizer

_bpe = None

def _init_worker(vocab_path, merges_path):
    global _bpe
    _bpe = BPETokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

def _encode_chunk(lines):
    ids = []
    for line in lines:
        ids.extend(_bpe.encode(line))
    return ids

def tokenize_data(input_path: str, output_path: str, vocab_path: str, merges_path: str):
    with open(input_path, "r") as f:
        lines = f.readlines()

    num_workers = 10
    chunk_size = 10000
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    with Pool(num_workers, initializer=_init_worker, initargs=(vocab_path, merges_path)) as pool:
        results = list(tqdm(
            pool.imap(_encode_chunk, chunks),
            total=len(chunks), desc=f"Encoding {input_path}"
        ))

    ids = [id for chunk_ids in results for id in chunk_ids]
    np.array(ids, dtype=np.uint16).tofile(output_path)


if __name__ == "__main__":
    # tokenize_data("data/raw_data/TinyStoriesV2-GPT4-train.txt", "data/tokenized_data/TinyStoriesV2-GPT4-train.bin", "data/TinyStories_vocab.json", "data/TinyStories_merges.txt")
    # tokenize_data("data/raw_data/TinyStoriesV2-GPT4-valid.txt", "data/tokenized_data/TinyStoriesV2-GPT4-valid.bin", "data/TinyStories_vocab.json", "data/TinyStories_merges.txt")
    tokenize_data("data/raw_data/owt_valid.txt", "data/tokenized_data/owt_valid.bin", "data/owt_vocab.json", "data/owt_merges.txt")
    tokenize_data("data/raw_data/owt_train.txt", "data/tokenized_data/owt_train.bin", "data/owt_vocab.json", "data/owt_merges.txt")