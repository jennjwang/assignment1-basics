from cProfile import Profile
from collections import Counter, defaultdict
import heapq
import json
import mmap
import numpy as np
import regex as re
import time
from tqdm import tqdm
import tracemalloc

from cs336_basics.tokenizer.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool, cpu_count

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _split_and_pretokenize(args):
    file_path, start, end, special_tokens = args
    with open(file_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        parts = [part for part in re.split(pattern, text) if part]
    else:
        parts = [text]
    return Counter(tok for part in parts for tok in re.findall(PAT, part))

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

    def _pretokenize_chunks(self) -> dict[str, int]:
        """Returns a dict mapping each unique pretoken to its total frequency."""
        num_processes = cpu_count()
        num_chunks = num_processes * 4
        with open(self.input_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            boundaries = find_chunk_boundaries(mm, num_chunks, b"<|endoftext|>")
            mm.close()

        tasks = [(self.input_path, start, end, self.special_tokens)
                 for start, end in zip(boundaries[:-1], boundaries[1:])]

        with Pool(num_processes) as p:
            results = list(tqdm(
                p.imap_unordered(_split_and_pretokenize, tasks),
                total=len(tasks), desc="Pretokenizing"
            ))

        merged: Counter = Counter()
        for chunk_counts in results:
            merged.update(chunk_counts)
        return merged

    def _merge(self, flat_tokens, pair, merged_id, freq, indices,
               flat_is_deleted, flat_prv, flat_nxt, N, flat_weight,
               heap, _hc):

        pos_arr = np.array(indices[pair], dtype=np.int64)

        if len(pos_arr) > 0:
            j_arr = flat_nxt[pos_arr]

            # filter: not deleted, within bounds, tokens still match (lazy deletion check)
            valid = (~flat_is_deleted[pos_arr] &
                     (j_arr < N) &
                     (flat_tokens[pos_arr] == pair[0]) &
                     (flat_tokens[j_arr] == pair[1]))

            i_pos = pos_arr[valid]
            j_pos = j_arr[valid]

            if len(i_pos) > 1:
                # resolve adjacency conflicts: sort, then greedily skip i[k] == j[k-1]
                sort_idx = np.argsort(i_pos)
                i_pos = i_pos[sort_idx]
                j_pos = j_pos[sort_idx]
                if (i_pos[1:] == j_pos[:-1]).any():
                    keep = np.ones(len(i_pos), dtype=bool)
                    for k in range(1, len(i_pos)):
                        if i_pos[k] == j_pos[k - 1] and keep[k - 1]:
                            keep[k] = False
                    i_pos = i_pos[keep]
                    j_pos = j_pos[keep]

            if len(i_pos) > 0:
                prev_pos = flat_prv[i_pos]
                next_pos = flat_nxt[j_pos]

                has_prev = prev_pos >= 0
                has_next = next_pos < N

                safe_prev = np.where(has_prev, prev_pos, 0)
                safe_next = np.where(has_next, next_pos, 0)
                prev_tokens = np.where(has_prev, flat_tokens[safe_prev], -1)
                next_tokens = np.where(has_next, flat_tokens[safe_next], -1)

                # batch update tokens, deleted, nxt, prv
                flat_tokens[i_pos] = merged_id
                flat_is_deleted[j_pos] = True
                flat_nxt[i_pos] = next_pos
                if has_next.any():
                    flat_prv[next_pos[has_next]] = i_pos[has_next]

                # batch freq + indices update for left neighbors
                if has_prev.any():
                    p_tok = prev_tokens[has_prev]
                    p_pos = prev_pos[has_prev]
                    unique_pt, inv = np.unique(p_tok, return_inverse=True)
                    for k, pt in enumerate(unique_pt):
                        pt = int(pt)
                        mask = inv == k
                        cnt = int(flat_weight[p_pos[mask]].sum())
                        old_p = (pt, pair[0])
                        freq[old_p] -= cnt
                        if freq[old_p] <= 0:
                            del freq[old_p]
                            if old_p in indices:
                                del indices[old_p]
                        else:
                            _hc[0] += 1
                            heapq.heappush(heap, (-freq[old_p], _hc[0], old_p))
                        new_p = (pt, merged_id)
                        freq[new_p] += cnt
                        _hc[0] += 1
                        heapq.heappush(heap, (-freq[new_p], _hc[0], new_p))
                        indices[new_p].extend(p_pos[mask].tolist())

                # batch freq + indices update for right neighbors
                if has_next.any():
                    n_tok = next_tokens[has_next]
                    i_next = i_pos[has_next]
                    unique_nt, inv = np.unique(n_tok, return_inverse=True)
                    for k, nt in enumerate(unique_nt):
                        nt = int(nt)
                        mask = inv == k
                        cnt = int(flat_weight[i_next[mask]].sum())
                        old_n = (pair[1], nt)
                        freq[old_n] -= cnt
                        if freq[old_n] <= 0:
                            del freq[old_n]
                            if old_n in indices:
                                del indices[old_n]
                        else:
                            _hc[0] += 1
                            heapq.heappush(heap, (-freq[old_n], _hc[0], old_n))
                        new_n = (merged_id, nt)
                        freq[new_n] += cnt
                        _hc[0] += 1
                        heapq.heappush(heap, (-freq[new_n], _hc[0], new_n))
                        indices[new_n].extend(i_next[mask].tolist())

        del freq[pair]
        del indices[pair]


    def train(self):

        # integer ID lookup: initially 0-255 for single bytes + special tokens
        id_to_bytes = dict(self.vocab)
        next_id = len(id_to_bytes)

        # pretokenize: deduplicated dict of pretoken -> frequency
        t0 = time.perf_counter()
        pretoken_counts = self._pretokenize_chunks()
        print(f"Pretokenization: {time.perf_counter() - t0:.3f}s  ({len(pretoken_counts)} unique pretokens)")
        if not pretoken_counts:
            return

        # unique sequences and their weights (avoids storing duplicates in flat arrays)
        raw_token_strs = list(pretoken_counts.keys())
        raw_token_weights = [pretoken_counts[s] for s in raw_token_strs]
        raw_tokens = [list(s.encode("utf-8")) for s in raw_token_strs]

        # build flat numpy arrays (N+1 so sentinel index N is safe)
        lengths = [len(s) for s in raw_tokens]
        offsets = np.zeros(len(raw_tokens) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        N = int(offsets[-1])

        flat_tokens = np.empty(N + 1, dtype=np.int32)
        flat_tokens[N] = -1  # sentinel
        flat_prv = np.empty(N, dtype=np.int64)
        flat_nxt = np.empty(N, dtype=np.int64)
        flat_weight = np.empty(N, dtype=np.int64)

        for ci, (seq, w) in enumerate(zip(raw_tokens, raw_token_weights)):
            s, e = int(offsets[ci]), int(offsets[ci + 1])
            flat_tokens[s:e] = seq
            flat_prv[s:e] = np.arange(s - 1, e - 1)
            flat_nxt[s:e] = np.arange(s + 1, e + 1)
            flat_prv[s] = -1   # no prev at sequence start
            flat_nxt[e - 1] = N  # sentinel: no next at sequence end
            flat_weight[s:e] = w

        flat_is_deleted = np.zeros(N, dtype=bool)

        # build freq and indices using numpy grouping
        all_pos = np.arange(N, dtype=np.int64)
        valid_pos = all_pos[flat_nxt[all_pos] < N]
        left_toks = flat_tokens[valid_pos].astype(np.int64)
        right_toks = flat_tokens[flat_nxt[valid_pos]].astype(np.int64)

        # encode each pair as a single int for fast grouping
        enc_factor = np.int64(next_id + 1)
        pair_enc = left_toks * enc_factor + right_toks
        sort_idx = np.argsort(pair_enc, kind='stable')
        sorted_enc = pair_enc[sort_idx]
        sorted_pos = valid_pos[sort_idx]
        sorted_left = left_toks[sort_idx]
        sorted_right = right_toks[sort_idx]

        bounds = np.concatenate([[0], np.where(np.diff(sorted_enc) != 0)[0] + 1, [len(sorted_enc)]])

        freq = defaultdict(int)
        indices = defaultdict(list)
        for k in range(len(bounds) - 1):
            gs, ge = int(bounds[k]), int(bounds[k + 1])
            a, b = int(sorted_left[gs]), int(sorted_right[gs])
            freq[(a, b)] = int(flat_weight[sorted_pos[gs:ge]].sum())
            indices[(a, b)] = sorted_pos[gs:ge].tolist()

        # build max-heap with lazy deletion; entries: (-freq, counter, pair)
        _hc = [0]
        heap = []
        for pair_key, f in freq.items():
            if f > 0:
                _hc[0] += 1
                heapq.heappush(heap, (-f, _hc[0], pair_key))

        # merge loop
        merge_pairs = []
        num_merges = self.vocab_size - len(self.vocab)
        with tqdm(total=num_merges, desc="BPE merges") as pbar:
            while len(merge_pairs) < num_merges:
                while heap:
                    neg_f, _, pair = heapq.heappop(heap)
                    if pair in freq and freq[pair] == -neg_f:
                        break
                else:
                    break
                merged_id = next_id
                next_id += 1
                id_to_bytes[merged_id] = id_to_bytes[pair[0]] + id_to_bytes[pair[1]]
                self._merge(flat_tokens, pair, merged_id, freq, indices,
                            flat_is_deleted, flat_prv, flat_nxt, N, flat_weight,
                            heap, _hc)
                merge_pairs.append(pair)
                pbar.update(1)

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

    tracemalloc.start()
    with Profile() as profile:
        special_tokens = ['<|endoftext|>']
        # input_path = 'data/TinyStoriesV2-GPT4-train.txt'
        # vocab_size = 10000

        input_path = 'data/owt_valid.txt'
        vocab_size = 32000

        bpe = BPETokenizer(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
        bpe.train()
        save(bpe, vocab_path="data/vocab.json", merges_path="data/merges.txt")
        import pstats

    stats = pstats.Stats(profile)
    stats.strip_dirs().sort_stats("cumulative").print_stats(20)
    stats.dump_stats("data/profile.out")    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"tracemalloc current: {current / 1e6:.1f} MB, peak: {peak / 1e6:.1f} MB")
    
    vocab_path = "data/vocab.json"
    merges_path = "data/merges.txt"

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
        longest = max(vocab.items(), key=lambda kv: len(kv[1]))
        # or all ties:
        max_len = max(len(v) for v in vocab.values())
        longest_all = [(k, v) for k, v in vocab.items() if len(v) == max_len]
        print(longest)
        print(max_len)
        print(longest_all)
