from cs336_basics.training.train_loop import train_loop
from cs336_basics.modal_utils import app, build_image
from tqdm import tqdm
import json
import time
import math

def run_lr_tuning(params: dict):

    rounds = 4
    lr_low = 1e-5
    lr_high = 1e-2

    while rounds > 0:

        m1 = lr_low ** (2/3) * lr_high ** (1/3)
        m2 = lr_low ** (1/3) * lr_high ** (2/3)
        lrs = [m1, m2]
        ranking = {}
        parallel_lrs = {}

        for lr in lrs:
            params_copy = params.copy()
            params_copy["max_learning_rate"] = lr
            params_copy["min_learning_rate"] = lr * 0.1
            params_copy["experiment_name"] = f"{params['experiment_name']}_lr_{lr}_{time.time()}"
            parallel_lrs[lr] = train_loop.spawn(params_copy)
        
        for lr in lrs:
            ranking[lr] = parallel_lrs[lr].get()
            print(f"LR: {lr}, Val Loss: {ranking[lr]}")

        if ranking[m1] < ranking[m2]:
            lr_high = m2
        elif ranking[m2] < ranking[m1]:
            lr_low = m1
        else:
            return m1, ranking[m1]
        
        rounds -= 1

        print(f"Round {5 - rounds}, LR Low: {lr_low}, LR High: {lr_high}")


    
    best_lr = min(ranking, key=ranking.get)

    print(f"Best LR: {best_lr}, Val Loss: {ranking[best_lr]}")

    return best_lr, ranking[best_lr]

    # lr_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    
    # # for diverging
    # # lr_values = [1e1, 1e2, 1e3]
    # ranking = {}

    # # parallelize training loop into 3 batches
    # batches = [lr_values[i:i+3] for i in range(0, len(lr_values), 3)]

    # for batch in batches:

    #     parallel_lrs = {}
    #     for lr in tqdm(batch):
    #         params_copy = params.copy()
    #         params_copy["max_learning_rate"] = lr
    #         params_copy["min_learning_rate"] = lr * 0.1
    #         params_copy["experiment_name"] = f"{params['experiment_name']}_lr_{lr}_{time.time()}"
    #         parallel_lrs[lr] = train_loop.spawn(params_copy)
        
    #     for lr in tqdm(batch):
    #         ranking[lr] = parallel_lrs[lr].get()

    # best_lr = min(ranking, key=ranking.get)
    # while best_lr <= lr_values[0] or best_lr >= lr_values[-1]:
    #     if best_lr <= lr_values[0]:
    #         extend_lr_values = [lr_values[0] * 1/9, lr_values[0] * 1/3]
    #         lr_values.insert(0, extend_lr_values[1])
    #         lr_values.insert(0, extend_lr_values[0])
    #     else:
    #         extend_lr_values = [lr_values[-1] * 3, lr_values[-1] * 9]
    #         lr_values.append(extend_lr_values[0])
    #         lr_values.append(extend_lr_values[1])
        
    #     for lr in tqdm(extend_lr_values):
    #         params_copy = params.copy()
    #         params_copy["max_learning_rate"] = lr
    #         params_copy["min_learning_rate"] = lr * 0.1
    #         params_copy["experiment_name"] = f"{params['experiment_name']}_lr_{lr}_{time.time()}"
    #         parallel_lrs[lr] = train_loop.spawn(params_copy)
        
    #     for lr in tqdm(extend_lr_values):
    #         ranking[lr] = parallel_lrs[lr].get()

    #     best_lr = min(ranking, key=ranking.get)
    
    # print(f"Best LR: {best_lr}, Val Loss: {ranking[best_lr]}")

    # return best_lr, ranking[best_lr]

@app.local_entrypoint()
def main_lr_tuning():
    with open("cs336_basics/training/params.json") as f:
        params = json.load(f)
    run_lr_tuning(params)
