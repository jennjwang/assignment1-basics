from cs336_basics.training.train_loop import train_loop
from cs336_basics.modal_utils import app, build_image
from tqdm import tqdm
import json
import time

def run_lr_tuning(params: dict):

    lr_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    ranking = {}

    # parallelize training loop into 3 batches
    batches = [lr_values[i:i+3] for i in range(0, len(lr_values), 3)]

    for batch in batches:

        parallel_lrs = {}
        for lr in tqdm(batch):
            params_copy = params.copy()
            params_copy["max_learning_rate"] = lr
            params_copy["min_learning_rate"] = lr * 0.1
            params_copy["experiment_name"] = f"lr_tuning_{lr}_{time.time()}"
            parallel_lrs[lr] = train_loop.spawn(params_copy)
        
        for lr in tqdm(batch):
            ranking[lr] = parallel_lrs[lr].get()

    best_lr = min(ranking, key=ranking.get)
    num_rounds = 2

    ranking = {}
    for i in range(num_rounds):

        lr_values = [best_lr * 0.5, best_lr * 2]
        parallel_lrs = {}

        for lr in tqdm(lr_values):
            params_copy = params.copy()
            params_copy["max_learning_rate"] = lr
            params_copy["min_learning_rate"] = lr * 0.1
            params_copy["experiment_name"] = f"lr_tuning_{lr}_{time.time()}"
            parallel_lrs[lr] = train_loop.spawn(params_copy)

        for lr in tqdm(lr_values):
            ranking[lr] = parallel_lrs[lr].get()

        best_lr = min(ranking, key=ranking.get)
        print(f"Round {i+1}, Best LR: {best_lr}, Val Loss: {ranking[best_lr]}")


@app.local_entrypoint()
def main_lr_tuning():
    with open("cs336_basics/training/params.json") as f:
        params = json.load(f)
    run_lr_tuning(params)
