from cProfile import Profile
from collections import Counter, defaultdict
import json
import mmap
import numpy as np
import regex as re
from tqdm import tqdm
import heapq
import array
from cs336_basics.tokenizer.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool, cpu_count


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _split_and_pretokenize(args):
    chunk_bytes, special_tokens = args
    text = chunk_bytes.decode("utf-8", errors="ignore")
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    parts = [part for part in re.split(pattern, text) if part]
    pretokens = [tok for part in parts for tok in re.findall(PAT, part)]
    return Counter(pretokens)

class _ReverseByte:
    def __init__(self, b): self.b = b
    def __lt__(self, o): return self.b > o.b
    def __le__(self, o): return self.b >= o.b
    def __gt__(self, o): return self.b < o.b
    def __ge__(self, o): return self.b <= o.b
    def __eq__(self, o): return self.b == o.b

class BPETokenizer:

    def __init__(self,
                 input_path = None,
                 vocab_size = None,
                 special_tokens = None,
                 vocab = None,
                 merges = None):
     
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.input_path = input_path
        self.merges = merges or []
        self.special_tokens = special_tokens or []
        if vocab is None:
            self.vocab = {i: bytes([i]) for i in range(256)}
            for tok in self.special_tokens:
                self.vocab[len(self.vocab)] = tok.encode("utf-8")
        self._bytes_to_id = {v: k for k, v in self.vocab.items()}
        self._encode_cache = {}
        self._merge_order = {}
        for i, (a, b) in enumerate(self.merges):
            pair = (self._bytes_to_id[a], self._bytes_to_id[b])
            if pair not in self._merge_order:
                self._merge_order[pair] = i
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
            vocab = {int(k): v.encode("latin-1") for k, v in vocab.items()}
        with open(merges_filepath, "r") as f:
            merges = f.readlines()
            final_merges = []
            for line in merges:
                line = line.strip('\n')
                if line.strip():
                    if line[0] == ' ':
                        rest = line[1:]
                        parts = rest.split(' ', 1)
                        a, b = ' ' + parts[0], parts[1] if len(parts) > 1 else '\n'
                    else:
                        a, b = line.split(' ', 1)
                    final_merges.append((a.encode('latin-1'), b.encode('latin-1')))
        return cls(vocab=vocab, merges=final_merges, special_tokens=special_tokens)

    def encode(self, text):
        res = []
        sorted_toks = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(tok) for tok in sorted_toks) + ")"
        parts = re.split(pattern, text)  if self.special_tokens else [text]
        for part in parts:
            if part in self.special_tokens:
                res.append(self._bytes_to_id[part.encode('utf-8')])
                continue
            for pretoken in re.findall(PAT, part):
                if pretoken in self._encode_cache:
                    res.extend(self._encode_cache[pretoken])
                    continue
                token_ids = [self._bytes_to_id[bytes([b])] for b in pretoken.encode('utf-8')]
                while len(token_ids) > 1:
                    priority_merge = None
                    best_rank = float('inf')
                    for i in range(len(token_ids) - 1):
                        rank = self._merge_order.get((token_ids[i], token_ids[i+1]))
                        if rank is not None and rank < best_rank:
                            priority_merge = i
                            best_rank = rank
                    if priority_merge is None:
                        break
                    pair = (token_ids[priority_merge], token_ids[priority_merge+1])
                    merged_id = self._bytes_to_id[self.vocab[pair[0]] + self.vocab[pair[1]]]
                    token_ids[priority_merge:priority_merge+2] = [merged_id]
                # for pair in self.merges:
                #     merged_id = self.bytes_to_id[pair[0] + pair[1]]
                #     i = 0
                #     while i < len(token_ids) - 1:
                #         if (self.vocab[token_ids[i]], self.vocab[token_ids[i+1]]) == pair:
                #             token_ids[i:i+2] = [merged_id]
                #         else:
                #             i += 1
                self._encode_cache[pretoken] = token_ids
                res.extend(token_ids)
        return res

    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids):
        token_bytes = [self.vocab[i] for i in ids]
        text = b"".join(token_bytes).decode("utf-8", errors="replace")
        return text
    
    def _pretokenize_chunks(self) -> Counter:
        with open(self.input_path, "rb") as f:
            num_processes = cpu_count()
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            boundaries = find_chunk_boundaries(mm, num_processes, b"<|endoftext|>")
            tasks = [bytes(mm[start:end]) for start, end in zip(boundaries[:-1], boundaries[1:])]

        with Pool(num_processes) as p:
            results = list(tqdm(
                p.imap(_split_and_pretokenize, [(chunk, self.special_tokens) for chunk in tasks]),
                total=len(tasks),
                desc="Pretokenizing"
            ))

        merged = Counter()
        for chunk_counts in results:
            merged.update(chunk_counts)
        return merged

    def _merge(self, full_tokens, pair, merged_id, pair_freq, pair_indices, prev_idx, next_idx, counts, is_merged):
        total_len = len(full_tokens)
        changed = set()
        while pair_indices[pair]:
            i = pair_indices[pair].pop()
            j = next_idx[i]
            if i < 0 or j >= total_len or is_merged[i] or is_merged[j]:
                continue
            
            prev_i = prev_idx[i]
            next_j = next_idx[j]
            pair_count = counts[i]
            prev_token = full_tokens[prev_i] if prev_i >= 0 else None
            next_token = full_tokens[next_j] if next_j < total_len else None

            full_tokens[i] = merged_id
            is_merged[j] = True
            next_idx[i] = next_j
            if next_j < total_len:
                prev_idx[next_j] = i
            prev_idx[j] = prev_i

            if prev_token is not None:
                prev_pair = (prev_token, pair[0])
                new_prev = (prev_token, merged_id)
                pair_freq[new_prev] += pair_count
                pair_freq[prev_pair] -= pair_count  
                pair_indices[new_prev].add(prev_i)
                pair_indices[prev_pair].discard(prev_i)
                changed.add(new_prev)
                changed.add(prev_pair)
            
            if next_token is not None:
                next_pair = (pair[1], next_token)
                new_next = (merged_id, next_token)
                pair_freq[new_next] += pair_count
                pair_freq[next_pair] -= pair_count
                pair_indices[new_next].add(i)
                pair_indices[next_pair].discard(j)
                changed.add(new_next)
                changed.add(next_pair)

        del pair_indices[pair]
        del pair_freq[pair]
        return changed

    def train(self):
        # integer ID lookup
        id_to_bytes = dict(self.vocab)
        next_id = len(id_to_bytes)
        cached_bytes = {}

        pretoken_counts = self._pretokenize_chunks()
        pretoken_str = list(pretoken_counts.keys()) 
        pretoken_counts = list(pretoken_counts.values()) # count corresponding to token at i
        pretokens_seq = [list(s.encode("utf-8")) for s in pretoken_str] # byte sequence corresponding to token at i

        pretoken_len = [len(seq) for seq in pretokens_seq]
        pretoken_idx = np.zeros(len(pretokens_seq) + 1, dtype=int)
        pretoken_idx[1:] = np.cumsum(pretoken_len)
        total_len = pretoken_idx[-1]

        full_tokens = array.array('i', [0] * total_len)
        next_idx = array.array('i', list(range(1, total_len + 1)))
        prev_idx = array.array('i', list(range(-1, total_len - 1)))
        full_counts = array.array('i', [0] * total_len)
        is_merged = array.array('b', [False] * total_len)

        for i in range(len(pretokens_seq)):
            pretoken_start, pretoken_end = int(pretoken_idx[i]), int(pretoken_idx[i+1])
            full_tokens[pretoken_start:pretoken_end] = array.array('i', pretokens_seq[i])
            full_counts[pretoken_start:pretoken_end] = array.array('i', [pretoken_counts[i]] * (pretoken_end - pretoken_start))
            prev_idx[pretoken_start:pretoken_end] = array.array('i', list(range(pretoken_start-1, pretoken_end-1)))
            next_idx[pretoken_start:pretoken_end] = array.array('i', list(range(pretoken_start+1, pretoken_end+1)))
            prev_idx[pretoken_start] = -1 # marks the start of seq, so there are no prev tokens
            next_idx[pretoken_end-1] = total_len

        pair_freq = defaultdict(int)
        pair_indices = defaultdict(set)
        for i in range(total_len):
            if next_idx[i] < total_len:
                pair = (full_tokens[i], full_tokens[next_idx[i]])
                pair_freq[pair] += full_counts[i]
                pair_indices[pair].add(i)
        
        heap_freq = []
        for pair, freq in pair_freq.items():
            pair_0 = id_to_bytes[pair[0]]
            pair_1 = id_to_bytes[pair[1]]
            cached_bytes[pair[0]] = _ReverseByte(pair_0)
            cached_bytes[pair[1]] = _ReverseByte(pair_1)
            heap_freq.append((-freq, cached_bytes[pair[0]], cached_bytes[pair[1]], pair))
        heapq.heapify(heap_freq)
        
        merged_pairs = []
        num_merges = self.vocab_size - len(self.vocab)

        with tqdm(total=num_merges, desc="BPE merges") as pbar:
            while len(merged_pairs) < num_merges:
                best_pair = None
                while heap_freq:
                    n, _, _, pair = heapq.heappop(heap_freq)
                    current = pair_freq.get(pair, 0)
                    if current == -n and current > 0:
                        best_pair = pair
                        break
                if best_pair is None:
                    break
                    # pair = max(pair_freq.items(), key=lambda item: (item[1],
                    #            id_to_bytes[item[0][0]], id_to_bytes[item[0][1]]))[0] # lexicographically greater
                merged_id = next_id
                next_id += 1
                id_to_bytes[merged_id] = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]
                changed = self._merge(full_tokens, best_pair, merged_id, pair_freq, pair_indices, prev_idx, next_idx, full_counts, is_merged)

                for changed_pair in changed:
                    freq = pair_freq.get(changed_pair, 0)
                    if freq > 0:
                        changed_pair_0 = id_to_bytes[changed_pair[0]]
                        changed_pair_1 = id_to_bytes[changed_pair[1]]
                        if changed_pair[0] not in cached_bytes:
                            cached_bytes[changed_pair[0]] = _ReverseByte(changed_pair_0)
                        if changed_pair[1] not in cached_bytes:
                            cached_bytes[changed_pair[1]] = _ReverseByte(changed_pair_1)
                        heapq.heappush(heap_freq, (-freq, cached_bytes[changed_pair[0]], cached_bytes[changed_pair[1]], changed_pair))

                merged_pairs.append(best_pair)
                pbar.update(1)

        self.vocab = id_to_bytes
        self.merges = [(id_to_bytes[a], id_to_bytes[b]) for a, b in merged_pairs]

def save(bpe, vocab_path="vocab.json", merges_path="merges.txt"):
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({k: v.decode("latin-1") for k, v in bpe.vocab.items()}, f, indent=2, ensure_ascii=False)
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in bpe.merges:
            f.write(f"{a.decode('latin-1')} {b.decode('latin-1')}\n")

if __name__ == "__main__":

    with Profile() as profile:
        special_tokens = ['<|endoftext|>']
        input_path = 'data/raw_data/owt_train.txt'
        vocab_size = 32000

        bpe = BPETokenizer(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
        bpe.train()
        # save(bpe, vocab_path="data/vocab.json", merges_path="data/merges.txt")
    
    import pstats

    stats = pstats.Stats(profile)
    stats.strip_dirs().sort_stats("cumulative").print_stats(20)
    stats.dump_stats("data/profile.out")

'''
4619906 function calls (4619188 primitive calls) in 23.278 seconds

   Ordered by: cumulative time
   List reduced from 503 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       76    0.002    0.000   20.927    0.275 pool.py:500(_wait_for_updates)
  262/258    0.002    0.000   19.789    0.077 connection.py:390(_recv)
  518/514    2.395    0.005   19.786    0.038 {built-in method posix.read}
      155    0.008    0.000   19.713    0.127 connection.py:1122(wait)
      6/5    0.028    0.005   15.391    3.078 threading.py:637(wait)
      2/1    1.131    0.565   15.357   15.357 tokenizer.py:191(train)
      2/1    2.265    1.133   13.009   13.009 tokenizer.py:126(_pretokenize_chunks)
      129    0.003    0.000   12.553    0.097 connection.py:246(recv)
      129    0.000    0.000   12.424    0.096 util.py:208(__call__)
        1    0.000    0.000   12.422   12.422 pool.py:738(__exit__)
        1    0.000    0.000   12.422   12.422 pool.py:654(terminate)
        1    0.001    0.001   12.422   12.422 pool.py:680(_terminate_pool)
        1    0.001    0.001   12.210   12.210 pool.py:671(_help_stuff_finish)
       57    0.078    0.001   12.209    0.214 {method 'acquire' of '_multiprocessing.SemLock' objects}
  131/129    0.001    0.000   12.181    0.094 connection.py:429(_recv_bytes)
      3/1    0.001    0.000   12.130   12.130 threading.py:1001(run)
        1    0.000    0.000   12.130   12.130 pool.py:573(_handle_results)
        1    0.001    0.001   12.128   12.128 pool.py:527(_handle_tasks)
      261    5.537    0.021    5.537    0.021 {method 'dump' of '_pickle.Pickler' objects}
        1    0.000    0.000    5.310    5.310 pool.py:305(_repopulate_pool)

'''