#!/bin/bash

# learning_rates=(1e-6 5e-6 1e-5 5e-5)
# weight_decays=(0.01 0.05)

learning_rates=(1e-6 5e-6 1e-5 5e-5)
weight_decays=(0.05)

for lr in "${learning_rates[@]}"; do
  for wd in "${weight_decays[@]}"; do
    sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1                # Use 1 node unless you need distributed computing across nodes
#SBATCH --ntasks=4               # Adjust tasks (parallel jobs) per node as needed
#SBATCH --cpus-per-task=4        # CPUs per task, adjust if needed
#SBATCH --mem=100G               # Increase memory if needed
#SBATCH --gres=gpu:2             # Use GPU if supported by the code
#SBATCH --time=24:00:00          # Set the job to 1 hour max
#SBATCH --partition=gpu          # h100, l40s
#SBATCH --account=cs433
#SBATCH --output=./logs/train_lr${lr}_wd${wd}_%j.log
#SBATCH --error=./logs/train_lr${lr}_wd${wd}_%j.err
#SBATCH --mail-type=END,FAIL,DONE 
--mail-user=jianan.xu@epfl.ch
#SBATCH --job-name=train_lr${lr}_wd${wd}

echo "fidis \$HOSTNAME"
eval "\$(conda shell.bash hook)"
conda activate motionclip
python -m train --learning_rate ${lr} --weight_decay ${wd}
EOT
  done
done