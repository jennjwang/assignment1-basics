'''
(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained
TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively),
encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio
(bytes/token)?
Deliverable: A one-to-two sentence response.
(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer?
Compare the compression ratio and/or qualitatively describe what happens.
Deliverable: A one-to-two sentence response.
(c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to
tokenize the Pile dataset (825GB of text)?
Deliverable: A one-to-two sentence response.
(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and
development datasets into a sequence of integer token IDs. We’ll use this later to train our
language model. We recommend serializing the token IDs as a NumPy array of datatype
uint16. Why is uint16 an appropriate choice?
'''

import numpy as np
from tokenizer import BPETokenizer
from tokenize_data import tokenize_data
import time

tiny_tokenizer = BPETokenizer.from_files("data/TinyStories_vocab.json", "data/TinyStories_merges.txt")
owt_tokenizer = BPETokenizer.from_files("data/owt_vocab.json", "data/owt_merges.txt")

tiny_train = open("data/raw_data/TinyStoriesV2-GPT4-valid.txt").read().split("<|endoftext|>")
owt_train = open("data/raw_data/owt_valid.txt").read().split("<|endoftext|>")

sample_tiny = tiny_train[:10]
sample_tiny_text = "".join(sample_tiny)
sample_owt = owt_train[:10]
sample_owt_text = "".join(sample_owt)

# sample_tiny_ids = owt_tokenizer.encode(sample_tiny_text)
tiny_start_time = time.time()
sample_tiny_ids = tiny_tokenizer.encode(sample_tiny_text)
total_time = time.time() - tiny_start_time
# print(f"TinyStories tokenization time: {total_time} seconds")
print(f"TinyStories throughput: {len(sample_tiny_text.encode('utf-8')) / total_time} bytes/second")


owt_start_time = time.time()
sample_owt_ids = owt_tokenizer.encode(sample_owt_text)
total_time = time.time() - owt_start_time
# print(f"OpenWebText tokenization time: {total_time} seconds")
print(f"OpenWebText throughput: {len(sample_owt_text.encode('utf-8')) / total_time} bytes/second")

# print(f"TinyStories compression ratio: {len(sample_tiny_text.encode('utf-8')) / len(sample_tiny_ids)}")
# print(f"OpenWebText compression ratio: {len(sample_owt_text.encode('utf-8')) / len(sample_owt_ids)}")

