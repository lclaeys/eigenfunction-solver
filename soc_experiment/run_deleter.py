import wandb

# Set your WandB project and group name
PROJECT_NAME = "neural-eigenfunction-learner"
ENTITY = "louis-claeys-eth-z-org"  # Or your WandB team name
GROUP_NAME = "OU-Stable-Quadratic-Hard"  # The group to delete

# Initialize API
api = wandb.Api()

# Get all runs in the project
runs = api.runs(f"{ENTITY}/{PROJECT_NAME}")
print(runs)
# Delete runs in the specified group
for run in runs:
    if run.group == GROUP_NAME:
        print(f"Deleting run: {run.name} ({run.id})")
        run.delete()