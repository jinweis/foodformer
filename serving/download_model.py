import os

import wandb

# from pathlib import Path


if not os.environ.get("WANDB_API_KEY"):
    raise ValueError(
        "You must set the WANDB_API_KEY environment variable " "to download the model."
    )

wandb_team = "jinwei-k-sun"
wandb_project = "Foodformer"
wandb_model = "vit:v0"
wandb_model_path = f"{wandb_team}/{wandb_project}/{wandb_model}"

# Initialize the wandb API
api = wandb.Api()

# Get all runs from the project
runs = api.runs(f"{wandb_team}/{wandb_project}")

# Find the run ID for the specified run name
for run in wandb_model:
    if run.name == wandb_model:
        run_id = run.id
        break
else:
    raise ValueError(
        f"Run with name {wandb_model} not found in project {wandb_project}"
    )

run = api.run(f"{wandb_team}/{wandb_project}/{run_id}")
run.file(wandb_model).download(replace=True)

print("Model downloaded")
# current_folder = Path(__file__).parent
# print(f"Folder: {current_folder}")
# path = wandb.use_artifact(wandb_model_path).download()
# print(f"Model downloaded to: {path}")


# entity = "wei_academic"
# project = "Foodformer_res"
# run_name = "VisionTransformer-5epochs"

# # Initialize the wandb API
# api = wandb.Api()

# # Get all runs from the project
# runs = api.runs(f"{entity}/{project}")

# # Find the run ID for the specified run name
# for run in runs:
#     if run.name == run_name:
#         run_id = run.id
#         break
# else:
#     raise ValueError(f"Run with name {run_name} not found in project {project}")

# # Now you can use the run ID to get the specific run and download the file
# file_name = "best_model.ckpt"
# run = api.run(f"{entity}/{project}/{run_id}")
# run.file(file_name).download(replace=True)


# print(f"Model downloaded")
