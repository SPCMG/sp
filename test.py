import torch
from torch.utils.data import DataLoader
from data.dataset import MotionTripleDataset, collate_fn
from eval.metrics import euclidean_distance_matrix, r_precision, chronological_accuracy_rate
import numpy as np
from omegaconf import OmegaConf
import argparse
import os
from models.model import MotionTextModel
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import clip

def evaluate_batch(motions, pos_texts, neg_texts, model, device):
    """
    Evaluate metrics for a single batch.
    """
    motion_embeds = model.encode_motion(motions)  # [batch_size, D]
    pos_embeds = model.encode_text(pos_texts)     # [batch_size, D]

    # Compute Euclidean distance matrix
    distance_matrix = euclidean_distance_matrix(motion_embeds, pos_embeds)

    # R-Precision for the batch
    r_prec_top1 = r_precision(distance_matrix, k=1)
    r_prec_top3 = r_precision(distance_matrix, k=3)
    r_prec_top5 = r_precision(distance_matrix, k=5)

    # --- CAR ---
    valid_motions = []
    valid_pos_texts = []
    valid_neg_texts = []

    for m, pos, neg_list in zip(motion_embeds, pos_embeds, neg_texts):
        if neg_list:  # Only include if there is a negative caption
            neg = neg_list[0]  # Extract the negative caption
            neg_tokens = clip.tokenize([neg]).to(device)
            neg_embeds = model.text_encoder.encode_text(neg_tokens).float()
            neg_embeds = F.normalize(neg_embeds, p=2, dim=1)
            valid_motions.append(m)
            valid_pos_texts.append(pos)
            valid_neg_texts.append(neg_embeds.squeeze(0))  # Remove batch dimension

    if len(valid_motions) > 0:
        car = chronological_accuracy_rate(valid_motions, valid_pos_texts, valid_neg_texts)
    else:
        car = 0  # No valid samples for CAR

    return r_prec_top1, r_prec_top3, r_prec_top5, car

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model across all batches and calculate metrics with mean and std.
    """
    batch_r_prec_top1 = []
    batch_r_prec_top3 = []
    batch_r_prec_top5 = []
    batch_car = []

    for i, (motions, pos_texts, neg_texts) in enumerate(dataloader):
        motions = pad_sequence(motions, batch_first=True).to(device)
        r_prec_top1, r_prec_top3, r_prec_top5, car = evaluate_batch(motions, pos_texts, neg_texts, model, device)

        print(f"Batch: {i}, R-Precision Top1: {r_prec_top1:.4f}, Top3: {r_prec_top3:.4f}, Top5: {r_prec_top5:.4f}, CAR: {car:.4f}")
        batch_r_prec_top1.append(r_prec_top1)
        batch_r_prec_top3.append(r_prec_top3)
        batch_r_prec_top5.append(r_prec_top5)
        batch_car.append(car)

    # Compute mean and std for each metric
    metrics = {
        "R-Precision Top1": (np.mean(batch_r_prec_top1), np.std(batch_r_prec_top1)),
        "R-Precision Top3": (np.mean(batch_r_prec_top3), np.std(batch_r_prec_top3)),
        "R-Precision Top5": (np.mean(batch_r_prec_top5), np.std(batch_r_prec_top5)),
        "Chronological Accuracy Rate": (np.mean(batch_car), np.std(batch_car)),
    }

    return metrics

def main():
    config = OmegaConf.load("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_encoder", type=str, default=config.model.motion_encoder, help="Type of motion encoder (e.g., transformer, mamba)")
    parser.add_argument("--text_encoder", type=str, default=config.model.text_encoder, help="Type of text encoder (e.g., clip, laclip, motionlaclip, motionlaclip+)")
    parser.add_argument('--ckpt', type=str, default=config.test.ckpt, help='Path to best model checkpoint')
    args = parser.parse_args()

    # Override config values with user inputs
    config.model.motion_encoder = args.motion_encoder
    config.model.text_encoder = args.text_encoder
    config.test.ckpt = args.ckpt
    
    # Load test dataset
    test_dataset = MotionTripleDataset(
        motion_dir=config.data.motion_dir,
        text_dir=config.data.text_dir,
        negative_text_dir=config.data.negative_text_dir,
        split_file=os.path.join(config.data.data_root, "test.txt"),
        min_motion_len=config.data.min_motion_length,
        max_motion_len=config.data.max_motion_length
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        collate_fn=collate_fn
    )

    # Load model
    model = MotionTextModel(config).to(device)
    checkpoint = torch.load(config.test.ckpt, map_location=device)
    model.load_state_dict(checkpoint)

    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    for metric, (mean, std) in metrics.items():
        print(f"{metric}: Mean = {mean:.4f}, Std = {std:.4f}")


if __name__ == "__main__":
    main()