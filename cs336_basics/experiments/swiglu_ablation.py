from cs336_basics.modal_utils import app
import json
from cs336_basics.training.train_loop import train_loop
from cs336_basics.experiments.lr_tuning import run_lr_tuning
from pathlib import Path

'''
A learning curve comparing the performance of SwiGLU and SiLU feed-forward
networks, with approximately matched parameter counts.
'''

@app.local_entrypoint()
def swiglu_ablation():
    with open("cs336_basics/training/params.json") as f:
        params = json.load(f)
    params["experiment_name"] = f"swiglu_ablation"
    params['d_model'] = 448 # divisible by 32
    params['d_ff'] = 4 * params['d_model']
    train_loop.remote(params)

# if __name__ == "__main__":
#     main()