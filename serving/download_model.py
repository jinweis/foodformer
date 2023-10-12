import os
from pathlib import Path

import wandb

if not os.environ.get("WANDB_API_KEY"):
    raise ValueError(
        "You must set the WANDB_API_KEY environment variable " "to download the model."
    )

wandb_team = "jinwei-k-sun"
wandb_project = "Foodformer"
wandb_model = "vit:v0"
wandb_model_path = f"{wandb_team}/{wandb_project}/{wandb_model}"

wandb.init()

current_folder = Path(__file__).parent
print(f"Folder: {current_folder}")
path = wandb.use_artiface(wandb_model_path).download()
print(f"Model downloaded to: {path}")
