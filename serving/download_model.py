# import os
# from pathlib import Path

# import wandb

# if not os.environ.get("WANDB_API_KEY"):
#     raise ValueError(
#         "You must set the WANDB_API_KEY environment variable " "to download the mod
# el."
#     )

# wandb_team = "jinwei-k-sun"
# wandb_project = "foodformer"
# wandb_model = "VisionTransformer-base"
# wandb_model_path = f"{wandb_team}/{wandb_project}/{wandb_model}"

# wandb.init()

# current_folder = Path(__file__).parent
# print(f"Folder: {current_folder}")
# path = wandb.use_artifact(wandb_model_path).download(root=current_folder)
# print(f"Model downloaded to: {path}")
import os

import wandb

# from pathlib import Path


if not os.environ.get("WANDB_API_KEY"):
    raise ValueError(
        "You must set the WANDB_API_KEY environment variable " "to download the model."
    )

entity = "jinwei-k-sun"
project = "foodformer"
run_name = "VisionTransformer-base"

# Initialize the wandb API
api = wandb.Api()

# Get all runs from the project
runs = api.runs(f"{entity}/{project}")

# Find the run ID for the specified run name
for run in runs:
    if run.name == run_name:
        run_id = run.id
        break
else:
    raise ValueError(f"Run with name {run_name} not found in project {project}")

# Now you can use the run ID to get the specific run and download the file
file_name = "best_model.ckpt"
run = api.run(f"{entity}/{project}/{run_id}")
run.file(file_name).download(replace=True)


print("Model downloaded")
