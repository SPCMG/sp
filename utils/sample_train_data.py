import random

# File paths
input_file = '/home/jiaxu/projects/sp-mdm/dataset/HumanML3D/train.txt'  
output_file = '/home/jiaxu/projects/sp-mdm/dataset/HumanML3D/train_temp.txt'
num_train_data = 11680

# Read all lines from the file
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Randomly sample
sampled_lines = random.sample(lines, num_train_data)

# Write the sampled lines to the output file
with open(output_file, 'w', encoding='utf-8') as f_out:
    f_out.writelines(sampled_lines)

print(f"Sampled {len(sampled_lines)} lines and saved them to {output_file}.")