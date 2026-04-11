#!/bin/bash
#SBATCH --job-name=gap_steam_long
#SBATCH --output=/home/g/gapaul/logs/%j_out.log
#SBATCH --error=/home/g/gapaul/errors/%j_error.log
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gapaul@comp.nus.edu.sg

# Config
SOC_USERNAME="gapaul"
SOC_USERNAME_PREFIX="g"
PROJECT_NAME="cs5242-project"

# Use the shared cluster environment that already has the required Python toolchain.
echo "Activating environment: /home/${SOC_USERNAME_PREFIX}/${SOC_USERNAME}/env/bin/activate"
source /home/${SOC_USERNAME_PREFIX}/${SOC_USERNAME}/env/bin/activate
echo "Environment activated"
echo "Submitting from directory: $(pwd) [Is same as ${SLURM_SUBMIT_DIR}?]"
cd "/home/${SOC_USERNAME_PREFIX}/${SOC_USERNAME}/${PROJECT_NAME}"
echo "Current directory: $(pwd)"
# Interactive Jupyter remains available as an opt-in path if you need to reverse tunnel into the node.
# echo "Use token: $TOKEN, Jupyter will run on: $HOSTNAME:$PORT"
# jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT --ServerApp.token=$TOKEN --ServerApp.allow_origin='*'

# Default batch path: execute the crawler notebook headlessly and persist the executed notebook artifact.
    # jupyter nbconvert \
    # --to notebook \
    # --execute notebooks/steam_crawler.ipynb \
    # --ExecutePreprocessor.timeout=-1 \
    # --output steam_crawler.executed.ipynb

# Run the crawler notebook headlessly
python steam-crawler/run_notebook.py --run-mode full --stage all