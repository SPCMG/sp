#!/bin/bash
#SBATCH --nodes=1                # Use 1 node unless you need distributed computing across nodes
#SBATCH --ntasks=4               # Adjust tasks (parallel jobs) per node as needed
#SBATCH --cpus-per-task=4        # CPUs per task, adjust if needed
#SBATCH --mem=100G               # Increase memory if needed
#SBATCH --gres=gpu:2             # Use GPU if supported by the code
#SBATCH --time=72:00:00          # Set the job to 1 hour max
#SBATCH --partition=gpu          # h100, l40s
#SBATCH --account=cs433
#SBATCH --output=./logs/train_%j.log
#SBATCH --error=./logs/train_%j.err
#SBATCH --mail-type=END,FAIL,DONE
#SBATCH --mail-user=jianan.xu@epfl.ch

echo "fidis $HOSTNAME"

# Initialize Conda
eval "$(conda shell.bash hook)"

# Activate the Conda environment
conda activate motionclip

# Run the Python script
python -m train