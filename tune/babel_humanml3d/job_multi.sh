#!/bin/bash

learning_rates=(1e-6 5e-6 1e-5 5e-5)
weight_decays=(0.01 0.05)

for lr in "${learning_rates[@]}"; do
  for wd in "${weight_decays[@]}"; do
    sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1 --ntasks=4 --cpus-per-task=4 --mem=100G --gres=gpu:2
#SBATCH --time=72:00:00 --partition=gpu
#SBATCH --output=./logs/train_lr${lr}_wd${wd}_%j.log
#SBATCH --error=./logs/train_lr${lr}_wd${wd}_%j.err
#SBATCH --mail-type=END,FAIL,DONE --mail-user=jianan.xu@epfl.ch
#SBATCH --job-name=train_lr${lr}_wd${wd}

echo "fidis \$HOSTNAME"
eval "\$(conda shell.bash hook)"
conda activate motionclip
python -m train --learning_rate ${lr} --weight_decay ${wd}
EOT
  done
done