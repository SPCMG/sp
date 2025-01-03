import os
import torch
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
import torch.nn.functional as F
from metrics import compute_car, compute_r_precision, process_batch, build_similarity_matrix

from data.dataset import MotionDataset, collate_fn
from models.model import MotionTextModel


def evaluate_model(model, dataloader, device="cuda"):
    """
    Evaluate the model on CAR and R-Precision.
    """
    model.eval()

    motion_embs = []
    all_pos_caps_list = []
    pos_avg_sims = []
    neg_avg_sims = []

    with torch.no_grad():
        for motions, pos_caps_list, neg_caps_list in dataloader:
            # Process motions
            motions = rnn.pad_sequence(motions, batch_first=True).to(device)
            motion_emb = model.motion_encoder(motions)
            motion_emb = F.normalize(motion_emb, p=2, dim=1)
            motion_embs.append(motion_emb)

            # Store positive captions for building the similarity matrix
            all_pos_caps_list.extend(pos_caps_list)

            # Compute positive and negative similarities
            pos_sims, neg_sims = process_batch(model, motion_emb, pos_caps_list, neg_caps_list, device)
            pos_avg_sims.extend(pos_sims)
            neg_avg_sims.extend(neg_sims)

    # Concatenate motion embeddings
    motion_embs = torch.cat(motion_embs, dim=0)

    # Convert similarities to tensors
    pos_avg_sims = torch.tensor(pos_avg_sims, device=device)
    neg_avg_sims = torch.tensor([s if s is not None else float('-inf') for s in neg_avg_sims], device=device)

    # Build similarity matrix for R-Precision
    motion_cap_sim_matrix = build_similarity_matrix(model, motion_embs, all_pos_caps_list, device)

    # Compute CAR
    car_score = compute_car(pos_avg_sims, neg_avg_sims)

    # Compute R-Precision
    r_precision_scores = compute_r_precision(motion_cap_sim_matrix, k_values=[1, 3, 5])

    return car_score, r_precision_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best model checkpoint')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    test_dataset = MotionDataset(
        motion_dir=config.data.motion_dir,
        text_dir=config.data.text_dir,
        negative_text_dir=config.data.negative_text_dir,
        split_file=os.path.join(config.data.data_root, "test.txt"),
        min_motion_len=config.data.min_motion_length,
        max_motion_len=config.data.max_motion_length
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn
    )

    # Load model
    model = MotionTextModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)

    # Evaluate model
    car_score, r_precision_scores = evaluate_model(model, test_loader, device)

    print(f"CAR: {car_score:.4f}")
    for k, v in r_precision_scores.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()