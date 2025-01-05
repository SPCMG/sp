import os
import torch
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
import torch.nn.functional as F

from data.dataset_test import MotionTripleDataset, collate_fn
from models.model import MotionTextModel
from eval.metrics import (
    compute_car_vectorized,
    compute_r_precision,
    process_batch,
    encode_and_average_captions,
    build_similarity_matrix_fast  
)

def evaluate_model(model, dataloader, device="cuda"):
    model.eval()

    motion_embs = []
    all_pos_caps_list = []
    pos_avg_sims = []
    neg_avg_sims = []

    with torch.no_grad():
        for motions, pos_caps_list, neg_caps_list in dataloader:
            # 1) Encode the motions once
            motions = rnn.pad_sequence(motions, batch_first=True).to(device)
            batch_motion_emb = model.motion_encoder(motions)
            batch_motion_emb = F.normalize(batch_motion_emb, p=2, dim=1)

            motion_embs.append(batch_motion_emb)

            # 2) Keep track of all positive & negative captions
            all_pos_caps_list.extend(pos_caps_list)

            # 3) Compute pos/neg average similarities for CAR
            #    (But let's do a minor batching speedup inside process_batch)
            batch_pos_sims, batch_neg_sims = process_batch(
                model, batch_motion_emb, pos_caps_list, neg_caps_list, device
            )
            pos_avg_sims.extend(batch_pos_sims)
            neg_avg_sims.extend(batch_neg_sims)

    # Concatenate motion embeddings [N, D]
    motion_embs = torch.cat(motion_embs, dim=0)

    # Convert average pos/neg sim arrays to tensors
    pos_avg_sims = torch.tensor(pos_avg_sims, device=device)
    neg_avg_sims = torch.tensor(
        [s if s is not None else float('-inf') for s in neg_avg_sims], device=device
    )

    # --- NEW: build NxN sim matrix in one go ---
    # 1) Already have motion_embs for all N samples
    # 2) Encode & average the *positive* captions for each sample in one pass:
    #      text_embs: [N, D]
    text_embs = encode_and_average_captions(model, all_pos_caps_list, device)

    # If your motion_embs are already normalized, also ensure text_embs is normalized
    motion_embs = F.normalize(motion_embs, p=2, dim=1)

    # 3) Single matrix multiplication -> NxN
    motion_cap_sim_matrix = build_similarity_matrix_fast(motion_embs, text_embs)

    # Metrics
    car_score = compute_car_vectorized(pos_avg_sims, neg_avg_sims)
    r_precision_scores = compute_r_precision(motion_cap_sim_matrix, k_values=[1, 3, 5])

    return car_score, r_precision_scores

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
    car_score, r_precision_scores = evaluate_model(model, test_loader, device)

    print(f"For {config.model.text_encoder}-{config.model.motion_encoder} model:")
    print(f"CAR: {car_score:.4f}")
    for k, v in r_precision_scores.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()