2.1
a. it returns '\x00' which is a null byte.
b. its string representation is '\x00' whereas its printed representation looks like an empty string.
c. when this character is printed, it is treated as a null character. It's invisible, but it's not an empty string because it still has a length of 1.

2.2
a.

2.5
a. It took 38 seconds and 304.4 MB in memory. The longest token in the vocabulary is a tie among ' accomplishment', ' disappointment', and ' responsibility'. [TODO: do these words make sense?]
b. The part that took the longest was pretokenization (19s)
a. It took 182 seconds and 1073 MB in memory. The longest token in the vocabulary is '-------------------------'
b. The OWT tokenizer has a much larger and more diverse vocab. TinyStories' tokenizer tends to skew toward children's vocabulary (e.g., "ladybug", "pancakes"), while OWT has unique tokens that reflect real-world content (e.g., "violent", "war", "Filipino") as well as a number of characters that don't make up words (e.g., "" [TODO: add examples]). The longest tokens in OWT are repeated character like '-------------------------' which are most likely some formatting separators. TinyStories' longest tokens are instead common English words like "responsibility."

3.5
a.
parameters:

embedding: vocab_size \* d_model = 50,257 \* 1,600
transformer blocks: num_layers \* transformer_block = 48 \* ...

- qkv_weights = 3 \* d_model \* d_model = 3 \* 1,600 \* 1,600
- output_weights = d_model \* d_model = 1,600 \* 1,600
- RMSNorm = 2 \* d_model = 2 \* 1,600
- ffn = 3 \* d_ff \* d_model = 3 \* 1,600 \* 4,288

norm = d_model = 1,600
linear = d_model \* vocab_size = 50,257 \* 1,600

matrix_multiplies:
