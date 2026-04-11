# Deliverable: Write a script that runs a training loop to train your model on user-provided input.
# In particular, we recommend that your training script allow for (at least) the following:
# • Ability to configure and control the various model and optimizer hyperparameters.
# • Memory-efficient loading of large training and validation datasets with np.memmap.
# • Serializing checkpoints to a user-provided path.
# 36
# • Periodically logging training and validation performance (e.g., to console and/or an external
# service like Weights and Biases).

import argparse
import json
import time
import numpy as np
import wandb
from cs336_basics.training.data_loading import get_batch
from cs336_basics.training.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.training.adamW import AdamW
from cs336_basics.training.lr_schedule import learning_rate_schedule
from cs336_basics.transformer.transformer import Transformer
from cs336_basics.training.cross_entropy_loss import cross_entropy_loss
from cs336_basics.training.grad_clip import gradient_clipping
from einops import rearrange
import torch

wandb_secret = modal.Secret.from_name("wandb")

@app.function(secrets=[modal.Secret.from_name("wandb")])  

parser = argparse.ArgumentParser(description="training loop")
parser.add_argument("--config", type=str, default="params.json")
args = parser.parse_args()

with open(args.config, "r") as f:
    params = json.load(f)

batch_size = params["batch_size"]
context_length = params["context_length"]
vocab_size = params["vocab_size"]
device = params["device"]
train_dataset = np.memmap(params["train_path"], dtype=np.uint16, mode="r")
val_dataset = np.memmap(params["val_path"], dtype=np.uint16, mode="r")

model = Transformer(vocab_size=vocab_size, 
                    context_length=context_length, 
                    d_model=params["d_model"], 
                    num_layers=params["num_layers"], 
                    num_heads=params["num_heads"], 
                    d_ff=params["d_ff"], 
                    rope_theta=params["rope_theta"])

model.to(device)

max_learning_rate = params["max_learning_rate"]
min_learning_rate = params["min_learning_rate"]
warmup_iters = params["warmup_iters"]
cosine_cycle_iters = params["cosine_cycle_iters"]
max_l2_norm = params["max_l2_norm"]
save_interval = params["save_interval"]
log_interval = params["log_interval"]
num_iters = params["num_iters"]
learning_rate = learning_rate_schedule(0, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)

optimizer = AdamW(model.parameters(), 
                lr=learning_rate, 
                weight_decay=params["weight_decay"], 
                betas=params["betas"], 
                eps=params["eps"])

wandb.init(project="cs336_basics", config=params)

model.train()
start_time = time.time()

for iteration in range(num_iters):
    learning_rate = learning_rate_schedule(
        iteration, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
    optimizer.param_groups[0]['lr'] = learning_rate

    (inputs, labels) = get_batch(train_dataset, batch_size, context_length, device)

    logits = model(inputs)
    logits = rearrange(logits, "batch_size context_len vocab -> (batch_size context_len) vocab")
    labels = rearrange(labels, "batch_size context_len -> (batch_size context_len)")

    loss = cross_entropy_loss(logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    gradient_clipping(model.parameters(), max_l2_norm)
    optimizer.step()

    if iteration % save_interval == 0:
        save_checkpoint(model, optimizer, iteration, params["checkpoint_path"])

    if iteration % log_interval == 0:
        with torch.no_grad():
            model.eval()
            (inputs, labels) = get_batch(val_dataset, batch_size, context_length, device)

            logits = model(inputs)
            logits = rearrange(logits, "batch_size context_len vocab -> (batch_size context_len) vocab")
            labels = rearrange(labels, "batch_size context_len -> (batch_size context_len)")

            val_loss = cross_entropy_loss(logits, labels)
            print(f"Iteration {iteration}, Loss: {loss.item()}, Learning Rate: {learning_rate}, Validation Loss: {val_loss.item()}")
            wandb.log({"train_loss": loss.item(), "val_loss": val_loss.item(), "lr": learning_rate, "time": time.time() - start_time}, step=iteration)
        model.train()

