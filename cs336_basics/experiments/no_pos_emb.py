from cs336_basics.modal_utils import app
import json
from cs336_basics.training.train_loop import train_loop
from pathlib import Path

'''
Modify your Transformer implementation with RoPE to remove the position embedding
information entirely, and see what happens
'''

@app.local_entrypoint()
def no_pos_emb():
    with open("cs336_basics/training/params.json") as f:
        params = json.load(f)
    params["experiment_name"] = f"no_pos_emb"
    params["rope_theta"] = None
    train_loop.remote(params)


# if __name__ == "__main__":
#     main()