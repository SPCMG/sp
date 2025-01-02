import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import torch.nn.functional as F

from data.dataset import MotionTripleDataset, collate_fn
from models.model import MotionTextModel

def train_one_epoch(model, dataloader, optimizer, device):
    """
    Train for one epoch using the three-term contrastive loss.
    Returns the average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    print("Training...")

    for motions, pos_texts, neg_texts in dataloader:
        if len(motions) == 0:
            continue
        num_batches += 1

        # Convert motion list to padded tensor
        motions = torch.nn.utils.rnn.pad_sequence(motions, batch_first=True).to(device)

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
            valid_labels = torch.tensor(valid_labels, device=device)
            loss_neg_custom = model(list(valid_neg_texts), list(valid_motions), valid_labels)
        else:
            loss_neg_custom = 0.0

        # Compute total loss
        loss_total = loss_pos + loss_neg_random + loss_neg_custom

        # Backprop and optimize
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        total_loss += loss_total.item()

    return total_loss / num_batches if num_batches > 0 else 0.0

def val_one_epoch(model, dataloader, device):
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
            motions = torch.nn.utils.rnn.pad_sequence(motions, batch_first=True).to(device)

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
                valid_labels = torch.tensor(valid_labels, device=device)
                loss_neg_custom = model(list(valid_neg_texts), list(valid_motions), valid_labels)
            else:
                loss_neg_custom = 0.0

            # Compute total loss
            loss_total = loss_pos + loss_neg_random + loss_neg_custom
            total_loss += loss_total.item()

    return total_loss / num_batches if num_batches > 0 else 0.0

def main():
    """
    Main training loop:
      - Loads config.yaml
      - Creates dataset & dataloader
      - Builds model
      - Runs train/val epochs
      - Logs losses
    """
    # 1) Load config
    config = OmegaConf.load("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Create Datasets
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

    # 3) DataLoaders
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

    # 4) Initialize Model
    model = MotionTextModel(config).to(device)

    # 5) Optimizer (only motion_encoder is trainable if text encoder is frozen)
    optimizer = optim.Adam(model.motion_encoder.parameters(), lr=config.train.learning_rate)

    # 6) Training Loop
    best_val_loss = float("inf")
    for epoch in range(config.train.num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        # Validate
        val_loss = val_one_epoch(model, val_loader, device)

        # Print progress
        print(f"[Epoch {epoch+1}/{config.train.num_epochs}] "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # (Optional) Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     save_path = os.path.join(config.checkpoint.save_dir, "best_model.pth")
        #     torch.save(model.state_dict(), save_path)
        #     print(f"  [*] Saved best model to {save_path}")

if __name__ == "__main__":
    main()
