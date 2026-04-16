from cs336_basics.training.train_loop import train_loop
from cs336_basics.modal_utils import app
import torch
import json
import time
from cs336_basics.experiments.lr_tuning import run_lr_tuning

def run_batch_size_tuning(params: dict):

    batch_size_value = 512
    best_batch_size = None
    best_val_loss = float('inf')

    try:
        while True:
            params_copy = params.copy()
            params_copy["num_iters"] = 327680000 // batch_size_value // params_copy["context_length"]
            params_copy["batch_size"] = batch_size_value
            params_copy["experiment_name"] = f"batch_size_tuning_{batch_size_value}_{time.time()}"
            params_copy['cosine_cycle_iters'] = params_copy["num_iters"] - 500
            # val_loss = train_loop.remote(params_copy)

            lr, val_loss = run_lr_tuning(params_copy)
            params_copy["max_learning_rate"] = lr
            params_copy["min_learning_rate"] = lr * 0.1
            params_copy['learning_rate'] = lr

            print(f"Batch Size: {batch_size_value}, LR: {lr}, Val Loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_batch_size = batch_size_value
            
            if batch_size_value < 64:
                batch_size_value *= 4
            else:
                batch_size_value *= 2

    except torch.cuda.OutOfMemoryError as e:
        print(f"Best Batch Size: {best_batch_size}, Val Loss: {best_val_loss}")
        return

@app.local_entrypoint()
def main_batch_size_tuning():
    with open("cs336_basics/training/params.json") as f:
        params = json.load(f)
    run_batch_size_tuning(params)
