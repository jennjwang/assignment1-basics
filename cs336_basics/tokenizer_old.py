from cProfile import Profile
from collections import defaultdict
import json
import mmap
import regex as re
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool, cpu_count

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _split_and_pretokenize(args):
    chunk_bytes, special_tokens = args
    text = chunk_bytes.decode("utf-8", errors="ignore")
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    parts = [part for part in re.split(pattern, text) if part]
    return [tok for part in parts for tok in re.findall(PAT, part)]

class BPETokenizer:

    def __init__(self,
                 input_path = None,
                 vocab_size = None,
                 special_tokens = None,
                 vocab = None,
                 merges = None):

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.vocab_size = vocab_size
        self.input_path = input_path
        self.merges = []
        self.special_tokens = special_tokens or []
        for tok in self.special_tokens:
            self.vocab[len(self.vocab)] = tok.encode("utf-8")

    def _pretokenize_chunks(self):

        with open(self.input_path, "rb") as f:
            num_processes = cpu_count()
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            boundaries = find_chunk_boundaries(mm, num_processes, b"<|endoftext|>")
            tasks = [bytes(mm[start:end]) for start, end in zip(boundaries[:-1], boundaries[1:])]

        # split on special tokens
        with Pool(num_processes) as p:
            results = p.map(_split_and_pretokenize, [(chunk, self.special_tokens) for chunk in tasks])

        return results

    def _merge(self, tokens, pair, merged_id, freq, indices, is_deleted, prv, nxt):
        pair_indices = list(indices[pair]) # snapshot; set mutates during iteration

        for chunk_idx, i in pair_indices:
            j = nxt[chunk_idx][i]  # derive j from nxt

            # skip if either position already deleted
            if is_deleted[chunk_idx][i] or is_deleted[chunk_idx][j]:
                continue

            # locate neighbors via prev/next arrays
            prev_i = prv[chunk_idx][i]
            next_j = nxt[chunk_idx][j]
            seq = tokens[chunk_idx]
            seq_len = len(seq)

            prev = seq[prev_i] if prev_i >= 0 else None
            next = seq[next_j] if next_j < seq_len else None

            # update token and splice j out of the prev/next chain
            seq[i] = merged_id
            is_deleted[chunk_idx][j] = True
            nxt[chunk_idx][i] = next_j
            if next_j < seq_len:
                prv[chunk_idx][next_j] = i

            if prev is not None:
                prev_pair = (prev, pair[0])
                new_prev = (prev, merged_id)
                freq[new_prev] += 1
                freq[prev_pair] -= 1
                indices[new_prev].add((chunk_idx, prev_i))
                indices[prev_pair].discard((chunk_idx, prev_i))

            if next is not None:
                next_pair = (pair[1], next)
                new_next = (merged_id, next)
                freq[new_next] += 1
                indices[new_next].add((chunk_idx, i))
                freq[next_pair] -= 1
                indices[next_pair].discard((chunk_idx, j))

        # remove old
        del freq[pair]
        del indices[pair]


    def train(self):

        # integer ID lookup: initially 0-255 for single bytes + special tokens
        id_to_bytes = dict(self.vocab)
        next_id = len(id_to_bytes)

        # pretokenize into integer ID sequences (bytes iter yields ints 0-255 directly)
        tokens = []
        for chunk in self._pretokenize_chunks():
            for pretoken in chunk:
                tokens.append(list(pretoken.encode("utf-8")))

        # build frequency and indices over integer ID pairs
        freq = defaultdict(int)
        indices = defaultdict(set)

        for chunk_idx, token in enumerate(tokens):
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                freq[pair] += 1
                indices[pair].add((chunk_idx, i))

        # build prev/next arrays for O(1) neighbor lookup
        prv = [list(range(-1, len(seq) - 1)) for seq in tokens]
        nxt = [list(range(1, len(seq) + 1)) for seq in tokens]

        # merge most frequent valid pairs
        is_deleted = [[False] * len(seq) for seq in tokens]
        merge_pairs = []
        num_merges = self.vocab_size - len(self.vocab)
        with tqdm(total=num_merges, desc="BPE merges") as pbar:
            while len(merge_pairs) < num_merges:
                # tie-break by bytes value to match reference
                pair = max(freq.items(), key=lambda item: (item[1],
                           id_to_bytes[item[0][0]], id_to_bytes[item[0][1]]))[0]
                merged_id = next_id
                next_id += 1
                id_to_bytes[merged_id] = id_to_bytes[pair[0]] + id_to_bytes[pair[1]]
                self._merge(tokens, pair, merged_id, freq, indices, is_deleted, prv, nxt)
                merge_pairs.append(pair)
                pbar.update(1)

        # build vocab and merges from integer IDs
        self.vocab = id_to_bytes
        self.merges = [(id_to_bytes[a], id_to_bytes[b]) for a, b in merge_pairs]

def save(bpe, vocab_path="vocab.json", merges_path="merges.txt"):
    # vocab: {token_id: string with escaped non-utf8 bytes}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({k: v.decode("utf-8", errors="replace") for k, v in bpe.vocab.items()}, f, indent=2, ensure_ascii=False)

    # merges: one pair per line as readable strings
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in bpe.merges:
            f.write(f"{a.decode('utf-8', errors='replace')} {b.decode('utf-8', errors='replace')}\n")

if __name__ == "__main__":

    with Profile() as profile:
        special_tokens = ['<|endoftext|>']
        input_path = 'data/TinyStoriesV2-GPT4-valid.txt'
        vocab_size = 10000

        bpe = BPETokenizer(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
        bpe.train()
        save(bpe, vocab_path="data/vocab.json", merges_path="data/merges.txt")
        import pstats

    stats = pstats.Stats(profile)
    stats.strip_dirs().sort_stats("cumulative").print_stats(20)
    stats.dump_stats("data/profile.out")
