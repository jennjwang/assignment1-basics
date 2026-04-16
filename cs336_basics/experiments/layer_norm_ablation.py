from cs336_basics.modal_utils import app
import json
from cs336_basics.training.train_loop import train_loop
from cs336_basics.experiments.lr_tuning import run_lr_tuning
from pathlib import Path

'''
Remove all of the RMSNorms from your Transformer and train. What happens at the previous
optimal learning rate? Can you get stability by using a lower learning rate?
'''

@app.local_entrypoint()
def layer_norm_ablation():
    with open("cs336_basics/training/params.json") as f:
        params = json.load(f)
    params["experiment_name"] = f"layer_norm_ablation"
    # train_loop.remote(params)

    run_lr_tuning(params)

# if __name__ == "__main__":
#     main()