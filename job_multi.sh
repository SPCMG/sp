#!/bin/bash

# Define parameters
learning_rates=(1e-4)
# text_encoders=("clip" "laclip" "motionlaclip")
text_encoders=("motionlaclipplus")
motion_encoders=("mamba")
# text_encoders=("clip")
# motion_encoders=("transformer" "mamba")

# Loop through all combinations of text_encoder, motion_encoder, and learning_rate
for lr in "${learning_rates[@]}"; do
  for te in "${text_encoders[@]}"; do
    for me in "${motion_encoders[@]}"; do
      sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1 --ntasks=4 --cpus-per-task=4 --mem=100G --gres=gpu:2
#SBATCH --time=72:00:00 --partition=gpu
#SBATCH --output=./logs/train_te${te}_me${me}_lr${lr}_%j.log
#SBATCH --error=./logs/train_te${te}_me${me}_lr${lr}_%j.err
#SBATCH --mail-type=END,FAIL,DONE --mail-user=jianan.xu@epfl.ch
#SBATCH --job-name=train_te${te}_me${me}_lr${lr}

echo "fidis \$HOSTNAME"
eval "\$(conda shell.bash hook)"
conda activate motionclip
python -m train --text_encoder ${te} --motion_encoder ${me} --learning_rate ${lr}
EOT
    done
  done
done