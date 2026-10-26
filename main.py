# main.py
# Runs the full pipeline: data preparation, training and the REST API.
import subprocess

# 1. Data preparation
print("Preparing data...")
subprocess.run(["python", "prepare_data.py"], check=True)

# 2. Model training
print("Training the agent...")
subprocess.run(["python", "train.py"], check=True)

# 3. REST API
print("Starting the REST API (CTRL+C to stop)...")
try:
    subprocess.run(["uvicorn", "api:app", "--reload"])
except KeyboardInterrupt:
    print("\nAPI stopped manually.")
