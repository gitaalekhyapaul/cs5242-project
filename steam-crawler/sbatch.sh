#!/bin/bash
#SBATCH --job-name=gap_steam
#SBATCH --output=~/logs/%j_out.log
#SBATCH --error=~/errors/%j_error.log
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gapaul@comp.nus.edu.sg

# Use the shared cluster environment that already has the required Python toolchain.
source ~/env/bin/activate

# Run from the repository folder so notebook paths resolve consistently regardless of submit location.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Jupyter uses this host and token if you later choose to uncomment the interactive lab command.
PORT=8888
HOSTNAME=$(hostname -s)
# Generate a fresh token for each job submission.
TOKEN=$(openssl rand -base64 64 | tr -dc 'A-Za-z0-9' | head -c 50; echo)

# Interactive Jupyter remains available as an opt-in path if you need to reverse tunnel into the node.
# echo "Use token: $TOKEN, Jupyter will run on: $HOSTNAME:$PORT"
# jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT --ServerApp.token=$TOKEN --ServerApp.allow_origin='*'

# Default batch path: execute the crawler notebook headlessly and persist the executed notebook artifact.
jupyter nbconvert \
  --to notebook \
  --execute notebooks/steam_crawler.ipynb \
  --output steam_crawler.executed.ipynb
