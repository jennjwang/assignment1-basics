import os
import json

import json

with open("data/TinyStories_vocab.json") as f:
    v1 = set(json.load(f).values())
with open("data/owt_vocab.json") as f:
    v2 = set(json.load(f).values())

shared = v1 & v2
only_tiny = v1 - v2
only_owt = v2 - v1

print(f"Shared: {len(shared)}, Only TinyStories: {len(only_tiny)}, Only OWT: {len(only_owt)}")

# Longest tokens in each
print("Longest TinyStories:", sorted(only_tiny, key=len, reverse=True)[:10])
print("Longest OWT:", sorted(only_owt, key=len, reverse=True)[:10])

print("Longest TinyStories:", sorted(v1, key=len, reverse=True)[:10])
print("Longest OWT:", sorted(v2, key=len, reverse=True)[:10])


import random

print("\nSample tokens only in TinyStories:")
print(random.sample(list(only_tiny), min(15, len(only_tiny))))

print("\nSample tokens only in OWT:")
print(random.sample(list(only_owt), min(15, len(only_owt))))


# vocab_path = "data/vocab.json"
# merges_path = "data/merges.txt"

# with open(vocab_path, "r") as f:
#     vocab = json.load(f)
#     longest = max(vocab.items(), key=lambda kv: len(kv[1]))
#     # or all ties:
#     max_len = max(len(v) for v in vocab.values())
#     longest_all = [(k, v) for k, v in vocab.items() if len(v) == max_len]
#     print(longest)
#     print(max_len)
#     print(longest_all)

# with open(merges_path, "r") as f:
#     merges = f.readlines()

# print(merges)