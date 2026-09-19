import subprocess
import os
import time

# 1. Priprema podataka
print("Pokrećem pripremu podataka...")
subprocess.run(["python", "prepare_data.py"], check=True)

# 2. Treniranje modela
print("Pokrećem treniranje agenta...")
subprocess.run(["python", "train.py"], check=True)

# 3. Pokretanje REST API-ja
print("Pokrećem REST API (CTRL+C za izlaz)...")
try:
    subprocess.run(["uvicorn", "api:app", "--reload"])
except KeyboardInterrupt:
    print("\nAPI ručno zaustavljen.")
