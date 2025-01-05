#!/bin/bash

# List of triples (text_encoder, motion_encoder, checkpoint directory)
model_combinations=(
    "motionlaclipplus mamba /scratch/izar/jiaxu/ckpts/motion_encoder/2ndtry_Aligner_ME_mamba_TE_motionlaclipplus_LR_5e-05_EP_100_10srqike"
    "motionlaclipplus transformer /scratch/izar/jiaxu/ckpts/motion_encoder/2ndtry_Aligner_ME_transformer_TE_motionlaclipplus_LR_5e-05_EP_100_ujyb2pvs"
    "motionlaclip mamba /scratch/izar/jiaxu/ckpts/motion_encoder/2ndtry_Aligner_ME_mamba_TE_motionlaclip_LR_5e-05_EP_100_ln120xcg"
    "motionlaclip transformer /scratch/izar/jiaxu/ckpts/motion_encoder/2ndtry_Aligner_ME_transformer_TE_motionlaclip_LR_5e-05_EP_100_ietogvw7"
    "laclip mamba /scratch/izar/jiaxu/ckpts/motion_encoder/2ndtry_Aligner_ME_mamba_TE_laclip_LR_5e-05_EP_100_t0v63n7d"
    "laclip transformer /scratch/izar/jiaxu/ckpts/motion_encoder/2ndtry_Aligner_ME_transformer_TE_laclip_LR_5e-05_EP_100_9nl7n4l6"
    "clip mamba /scratch/izar/jiaxu/ckpts/motion_encoder/2ndtry_Aligner_ME_mamba_TE_clip_LR_5e-05_EP_100_3xbidc8m"
    "clip transformer /scratch/izar/jiaxu/ckpts/motion_encoder/2ndtry_Aligner_ME_transformer_TE_clip_LR_5e-05_EP_100_exnzawg1"
)

output_dir="./test_logs_new"           # Output directory for logs
mkdir -p "$output_dir"

# Iterate over each model combination
for combo in "${model_combinations[@]}"; do
  # Parse the triple
  read -r text_encoder motion_encoder ckpt_dir <<< "$combo"

  # Debug: Print current combination
  echo "Processing: text_encoder=${text_encoder}, motion_encoder=${motion_encoder}, ckpt_dir=${ckpt_dir}"

  # Check if the directory exists
  if [ ! -d "$ckpt_dir" ]; then
    echo "Directory ${ckpt_dir} does not exist. Skipping..."
    continue
  fi

  # List files in the directory for debugging
  echo "Files in ${ckpt_dir}:"
  ls "$ckpt_dir"

  # Find the checkpoint file with the largest epoch number
  ckpt_file=$(ls "${ckpt_dir}"/best_model_epoch_*.pth 2>/dev/null | sort -V | tail -n 1)
  if [ -z "$ckpt_file" ]; then
    echo "No checkpoint file found in ${ckpt_dir}. Skipping..."
    continue
  fi

  # Debug: Print the checkpoint file path
  echo "Found checkpoint: ${ckpt_file}"

  # Extract the largest epoch number from the checkpoint file name
  epoch=$(basename "$ckpt_file" | grep -oP "epoch_\K[0-9]+")
  if [ -z "$epoch" ]; then
    echo "Failed to extract epoch number from ${ckpt_file}. Skipping..."
    continue
  fi

  # Debug: Print the epoch number
  echo "Using epoch: ${epoch}"

  # Submit the job
  sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1 --ntasks=4 --cpus-per-task=4 --mem=100G --gres=gpu:1
#SBATCH --time=72:00:00 --partition=gpu
#SBATCH --output=${output_dir}/test_te${text_encoder}_me${motion_encoder}_ep${epoch}_%j.log
#SBATCH --error=${output_dir}/test_te${text_encoder}_me${motion_encoder}_ep${epoch}_%j.err
#SBATCH --mail-type=END,FAIL,DONE --mail-user=jianan.xu@epfl.ch
#SBATCH --job-name=test_te${text_encoder}_me${motion_encoder}_ep${epoch}

echo "Running on \$HOSTNAME"
eval "\$(conda shell.bash hook)"
conda activate motionclip

# Run the test.py script
python test.py --text_encoder "${text_encoder}" --motion_encoder "${motion_encoder}" --ckpt "${ckpt_file}"
EOT

  # Debug: Confirm job submission
  echo "Job submitted for text_encoder=${text_encoder}, motion_encoder=${motion_encoder}, epoch=${epoch}"
done





