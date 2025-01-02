import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.nn.utils import rnn, clip_grad_norm_

from data.dataset import MotionTripleDataset, collate_fn
from models.model import MotionTextModel
from utils.logging import setup_wandb_and_logging
import wandb
import argparse

def train_one_epoch(model, dataloader, optimizer, device):
    """
    Train for one epoch using the three-term contrastive loss.
    Returns the average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for motions, pos_texts, neg_texts in dataloader:
        if len(motions) == 0:
            continue
        num_batches += 1

        # Convert motion list to padded tensor
        motions = rnn.pad_sequence(motions, batch_first=True).to(device)

        # Positive pair loss
        labels_pos = torch.zeros(len(motions), device=device)
        loss_pos = model(pos_texts, motions, labels_pos)

        # Random negative loss
        shift = 1
        idx_shifted = (torch.arange(len(motions), device=device) + shift) % len(motions)
        random_motions = motions[idx_shifted]
        labels_neg = torch.ones(len(motions), device=device)
        loss_neg_random = model(pos_texts, random_motions, labels_neg)

        # Custom negative loss
        valid_pairs = [(m, n, l) for m, n, l in zip(motions, neg_texts, labels_neg) if n != "(no negative lines)"]

        if valid_pairs:
            valid_motions, valid_neg_texts, valid_labels = zip(*valid_pairs)
            valid_motions = torch.stack(valid_motions).to(device)
            valid_labels = torch.tensor(valid_labels, device=device)
            loss_neg_custom = model(list(valid_neg_texts), valid_motions, valid_labels)
        else:
            loss_neg_custom = 0.0

        # Compute total loss
        loss_total = loss_pos + loss_neg_random + loss_neg_custom

        # Backprop and optimize
        optimizer.zero_grad()
        loss_total.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_total.item()

    return total_loss / num_batches if num_batches > 0 else 0.0

def validate_one_epoch(model, dataloader, device):
    """
    Evaluate the model (no gradient updates).
    Returns the average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for motions, pos_texts, neg_texts in dataloader:
            if len(motions) == 0:
                continue
            num_batches += 1

            # Convert motion list to padded tensor
            motions = rnn.pad_sequence(motions, batch_first=True).to(device)

            # Positive pair loss
            labels_pos = torch.zeros(len(motions), device=device)
            loss_pos = model(pos_texts, motions, labels_pos)

            # Random negative loss
            shift = 1
            idx_shifted = (torch.arange(len(motions), device=device) + shift) % len(motions)
            random_motions = motions[idx_shifted]
            labels_neg = torch.ones(len(motions), device=device)
            loss_neg_random = model(pos_texts, random_motions, labels_neg)
        
            # Custom negative loss
            valid_pairs = [(m, n, l) for m, n, l in zip(motions, neg_texts, labels_neg) if n != "(no negative lines)"]

            if valid_pairs:
                valid_motions, valid_neg_texts, valid_labels = zip(*valid_pairs)
                valid_motions = torch.stack(valid_motions).to(device)
                valid_labels = torch.tensor(valid_labels, device=device)
                loss_neg_custom = model(list(valid_neg_texts), valid_motions, valid_labels)
            else:
                loss_neg_custom = 0.0

            # Compute total loss
            loss_total = loss_pos + loss_neg_random + loss_neg_custom
            total_loss += loss_total.item()

    return total_loss / num_batches if num_batches > 0 else 0.0

def main():
    """
    Main training loop with user inputs for motion_encoder, text_encoder, and learning_rate.
    """
    # 1) Load config
    config = OmegaConf.load("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Parse user inputs with config defaults
    parser = argparse.ArgumentParser(description="Train Text-Motion Matching Model")
    parser.add_argument("--motion_encoder", type=str, default=config.model.motion_encoder,
                        help="Type of motion encoder (e.g., transformer, mamba)")
    parser.add_argument("--text_encoder", type=str, default=config.model.text_encoder,
                        help="Type of text encoder (e.g., clip, laclip, motionlaclip)")
    parser.add_argument("--learning_rate", type=float, default=config.train.learning_rate,
                        help="Learning rate for the optimizer")
    args = parser.parse_args()

    # Override config values with user inputs
    config.model.motion_encoder = args.motion_encoder
    config.model.text_encoder = args.text_encoder
    config.train.learning_rate = args.learning_rate

    # Initialize wandb
    run_name = setup_wandb_and_logging(config)

    # 3) Create Datasets
    train_dataset = MotionTripleDataset(
        motion_dir=config.data.motion_dir,
        text_dir=config.data.text_dir,
        negative_text_dir=config.data.negative_text_dir,
        split_file=os.path.join(config.data.data_root, "train.txt"),
        min_motion_len=config.data.min_motion_length,
        max_motion_len=config.data.max_motion_length,
        random_seed=config.data.random_seed
    )

    val_dataset = MotionTripleDataset(
        motion_dir=config.data.motion_dir,
        text_dir=config.data.text_dir,
        negative_text_dir=config.data.negative_text_dir,
        split_file=os.path.join(config.data.data_root, "val.txt"),
        min_motion_len=config.data.min_motion_length,
        max_motion_len=config.data.max_motion_length,
        random_seed=config.data.random_seed
    )

    # 4) DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )

    # 5) Initialize Model
    model = MotionTextModel(config).to(device)

    # 6) Optimizer
    optimizer = optim.AdamW(model.motion_encoder.parameters(), lr=config.train.learning_rate)

    # 7) Create a run-specific checkpoint directory
    checkpoint_dir = os.path.join(config.checkpoint.save_path, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 8) Training Loop
    best_val_loss = float("inf")
    for epoch in range(config.train.num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        # Validate
        val_loss = validate_one_epoch(model, val_loader, device)

        # Print progress
        print(f"[Epoch {epoch+1}/{config.train.num_epochs}] "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Log to wandb
        wandb.log({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  [*] Saved best model to {save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
