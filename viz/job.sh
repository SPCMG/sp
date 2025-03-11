#!/bin/bash
#SBATCH --job-name=jupyter-gpu     # Job name
#SBATCH --partition=gpu            # Partition name
#SBATCH --gres=gpu:2               # Request one GPU
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem=16G                  # Memory per node
#SBATCH --time=02:00:00            # Time limit hrs:min:sec
#SBATCH --output=jupyter_gpu_%j.out  # Standard output and error log
#SBATCH --error=jupyter_gpu_%j.err

echo "fidis $HOSTNAME"

# Initialize Conda
eval "$(conda shell.bash hook)"

# Activate the Conda environment
conda activate motionclip

# Run the Jupyter Notebook and save output to a new file
jupyter nbconvert --to notebook --execute viz_cluster.ipynb --output viz_show.ipynb --ExecutePreprocessor.timeout=-1

echo "Jupyter Notebook execution finished"
