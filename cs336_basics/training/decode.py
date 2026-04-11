from cs336_basics.transformer.attention import softmax
import torch
"""
    Deliverable: Implement a function to decode from your language model. We recommend that
you support the following features:
• Generate completions for a user-provided prompt (i.e., take in some 𝑥1…𝑡 and sample a
completion until you hit an <|endoftext|> token).
• Allow the user to control the maximum number of generated tokens.
• Given a desired temperature value, apply softmax temperature scaling to the predicted next-
token distributions before sampling.
• Top-𝑝 sampling ([A. Holtzman et al., 2020] also referred to as nucleus sampling), given a user-
specified threshold value.
"""

def decode(model, tokenizer, inputs, max_tokens, temperature=None, threshold=None):
    for i in range(max_tokens):
        logits = model(inputs)[:, -1, :]

        # temperature
        if temperature is not None:
            logits = logits / temperature
        
        probs = softmax(logits)

        # top-p sampling
        if threshold is not None:
            sorted_probs = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs.values, dim=-1)
            probs_indices = sorted_probs.indices[cum_probs - sorted_probs.values <= threshold]
            mask = torch.zeros_like(probs)
            mask[..., probs_indices] = 1
            probs = probs * mask
            probs = probs / probs.sum(dim=-1, keepdim=True)

        token = torch.multinomial(probs, num_samples=1)

        if tokenizer.decode(token.flatten().tolist()) == "<|endoftext|>":
            break

        inputs = torch.cat([inputs, token], dim=1)
    return inputs
