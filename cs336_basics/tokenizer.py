from cProfile import Profile
from collections import Counter, defaultdict
import json
import mmap
import numpy as np
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
    pretokens = [tok for part in parts for tok in re.findall(PAT, part)]
    return Counter(pretokens)

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

    def _pretokenize_chunks(self) -> Counter:
        with open(self.input_path, "rb") as f:
            num_processes = cpu_count()
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            boundaries = find_chunk_boundaries(mm, num_processes, b"<|endoftext|>")
            tasks = [bytes(mm[start:end]) for start, end in zip(boundaries[:-1], boundaries[1:])]

        with Pool(num_processes) as p:
            results = list(tqdm(
                p.imap(_split_and_pretokenize, [(chunk, self.special_tokens) for chunk in tasks]),
                total=len(tasks), desc="Pretokenizing"
            ))

        merged = Counter()
        for chunk_counts in results:
            merged.update(chunk_counts)
        return merged

    def _merge(self, tokens, pair, merged_id, pair_freq, indices, is_merged, prv, nxt, counts):
        N = len(tokens)
        for i in list(indices[pair]):
            j = nxt[i] # the position of the right token in the pair
            # skip if either position already deleted
            if i < 0 or j >= N or is_merged[i] or is_merged[j]:
                continue

            # locate neighbors via prev/next arrays
            prev_i = prv[i]
            next_j = nxt[j]
            token_count = counts[i]
            prev = tokens[prev_i] if prev_i >= 0 else None
            next = tokens[next_j] if next_j < N else None

            # update tokens with new merged id
            tokens[i] = merged_id
            is_merged[j] = True
            nxt[i] = next_j # skips over the merged id
            if next_j < N:
                prv[next_j] = i# update doubly linked list

            # update the frequency and indices of previous pairs
            if prev is not None:
                    prev_pair = (prev, pair[0])
                    new_prev = (prev, merged_id)
                    pair_freq[new_prev] += token_count
                    pair_freq[prev_pair] -= token_count
                    indices[new_prev].add(prev_i)
                    indices[prev_pair].discard(prev_i)
            
            # update the frequency and indices of next pairs
            if next is not None:
                next_pair = (pair[1], next)
                new_next = (merged_id, next)
                pair_freq[new_next] += token_count
                pair_freq[next_pair] -= token_count
                indices[new_next].add(i)
                indices[next_pair].discard(j)

        # remove old
        del pair_freq[pair]
        del indices[pair]


    def train(self):
        # integer ID lookup
        id_to_bytes = dict(self.vocab)
        next_id = len(id_to_bytes)

        # get tokens and counts (need to flatten)
        pretoken_counts = self._pretokenize_chunks()
        raw_str = list(pretoken_counts.keys()) 
        raw_counts = list(pretoken_counts.values()) # count corresponding to token at i
        raw_tokens = [list(s.encode("utf-8")) for s in raw_str] # byte sequence corresponding to token at i

        # offsets to track start and end of the tokens
        lengths = np.array([len(seq) for seq in raw_tokens])
        offsets = np.zeros(len(raw_tokens) + 1, dtype=np.int64) # the actual position of the token in the sequence
        offsets[1:] = np.cumsum(lengths)
        N = offsets[-1]

        # build prev/next arrays for neighbor lookup
        tokens = np.empty(N, dtype=np.int32) # flatten tokens
        prv = np.empty(N, dtype=np.int64) # holds the index to the previous token for token at i
        nxt = np.empty(N, dtype=np.int64) # holds the index to the next token for token at i
        counts = np.empty(N, dtype=np.int64)
        is_merged = np.zeros(N, dtype=bool)

        for i, (seq, count) in enumerate(zip(raw_tokens, raw_counts)):
            start, end = offsets[i], offsets[i+1]
            tokens[start: end] = raw_tokens[i]
            prv[start:end] = np.arange(start-1, end-1)
            nxt[start:end] = np.arange(start+1, end+1)
            prv[start] = -1
            nxt[end - 1] = N
            counts[start: end] = count

        # build freq and indices for each pair
        pair_freq = defaultdict(int)
        pair_indices = defaultdict(set)
        for pos in range(N):
            if nxt[pos] < N:
                pair = (tokens[pos], tokens[nxt[pos]])
                pair_freq[pair] += counts[pos]
                pair_indices[pair].add(pos)

        # merge most frequent pairs
        merged_pairs = []
        num_merges = self.vocab_size - len(self.vocab)
        with tqdm(total=num_merges, desc="BPE merges") as pbar:
            while len(merged_pairs) < num_merges:
                pair = max(pair_freq.items(), key=lambda item: (item[1],
                           id_to_bytes[item[0][0]], id_to_bytes[item[0][1]]))[0] # lexicographically greater
            
                merged_id = next_id
                next_id += 1
                id_to_bytes[merged_id] = id_to_bytes[pair[0]] + id_to_bytes[pair[1]]
                self._merge(tokens, pair, merged_id, pair_freq, pair_indices, is_merged, prv, nxt, counts)
                merged_pairs.append(pair)
                pbar.update(1)

        # build vocab and merges from integer IDs
        self.vocab = id_to_bytes
        self.merges = [(id_to_bytes[a], id_to_bytes[b]) for a, b in merged_pairs]

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
        input_path = 'data/owt_valid.txt'
        vocab_size = 32000

        bpe = BPETokenizer(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
        bpe.train()
        save(bpe, vocab_path="data/vocab.json", merges_path="data/merges.txt")
        import pstats

    stats = pstats.Stats(profile)
    stats.strip_dirs().sort_stats("cumulative").print_stats(20)
    stats.dump_stats("data/profile.out")
