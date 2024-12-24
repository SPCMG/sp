import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import clip
from omegaconf import OmegaConf

from configs.config import load_config
from data.dataset import HumanML3D, collate_fn
from models.model import MotionClipModel


def train_one_epoch(model, dataloader, optimizer, device="cuda"):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Move motion data to device
        x = batch["x"].to(device)          # [bs, max_len, 263]
        mask = batch["mask"].to(device)    # [bs, max_len]
        lengths = batch["lengths"].to(device)  # [bs]

        # Tokenize text using CLIP
        text_list = batch["clip_text"]      # list of strings
        text_tokenized = clip.tokenize(text_list).to(device)  # [bs, token_length]

        # Construct the dict for motion data 
        motion_batch = {
            "x": x,          # [bs, max_len, 263]
            "mask": mask,    # [bs, max_len]
            "lengths": lengths,  # [bs]
            # "y" is a dummy tensor; adjust if needed
            "y": torch.zeros_like(lengths)
        }

        # Forward pass
        outputs = model(motion_batch, text_tokenized)  # {"loss", "motion_emb", "text_emb"}
        loss = outputs["loss"]

        # Backward + step
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


def validate_one_epoch(model, dataloader, device="cuda"):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move motion data to device
            x = batch["x"].to(device)          # [bs, max_len, 263]
            mask = batch["mask"].to(device)    # [bs, max_len]
            lengths = batch["lengths"].to(device)  # [bs]

            # Tokenize text using CLIP
            text_list = batch["clip_text"]      # list of strings
            text_tokenized = clip.tokenize(text_list).to(device)  # [bs, token_length]

            # Construct the dict for motion data 
            motion_batch = {
                "x": x,          # [bs, max_len, 263]
                "mask": mask,    # [bs, max_len]
                "lengths": lengths,  # [bs]
                # "y" is a dummy tensor; adjust if needed
                "y": torch.zeros_like(lengths)
            }

            # Forward pass
            outputs = model(motion_batch, text_tokenized)  # {"loss", "motion_emb", "text_emb"}
            loss = outputs["loss"]

            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # 1) Load YAML config
    # ------------------------------------------------------------------
    cfg = load_config("config.yaml")  # Ensure the path is correct

    # ------------------------------------------------------------------
    # 2) Instantiate the datasets using config fields
    # ------------------------------------------------------------------
    train_dataset = HumanML3D(
        opt=cfg.data,
        split="train"
    )

    val_dataset = HumanML3D(
        opt=cfg.data,
        split="val"
    )

    # ------------------------------------------------------------------
    # 3) Build DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.train.shuffle,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,  # Optionally, use a different batch size
        shuffle=False,                    # Validation typically doesn't need shuffling
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn
    )

    # ------------------------------------------------------------------
    # 4) Initialize the MotionClipModel
    # ------------------------------------------------------------------
    model = MotionClipModel(
        njoints=cfg.data.n_joints,
        n_feats=cfg.data.n_feats,
        num_frames=cfg.data.max_motion_length,
        latent_dim=cfg.model.latent_dim,
        clip_model_name=cfg.model.clip_model_name,
        device=device
    ).to(device)

    # ------------------------------------------------------------------
    # 5) Setup Optimizer
    # ------------------------------------------------------------------
    # We typically train only the motion encoder’s parameters,
    # because the text encoder is frozen.
    optimizer = optim.Adam(model.motion_encoder.parameters(), lr=cfg.train.learning_rate)

    # ------------------------------------------------------------------
    # 6) Setup Checkpoint Directory
    # ------------------------------------------------------------------
    checkpoint_dir = cfg.checkpoint.save_path
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 7) Training Loop
    # ------------------------------------------------------------------
    num_epochs = cfg.train.num_epochs
    save_every = cfg.checkpoint.save_every

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        # Training phase
        train_loss = train_one_epoch(model, train_loader, optimizer, device=device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation phase
        val_loss = validate_one_epoch(model, val_loader, device=device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save checkpoint every 'save_every' epochs
        if epoch % save_every == 0 or epoch == num_epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"motionclipmodel_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        print("-" * 50)


if __name__ == "__main__":
    main()
