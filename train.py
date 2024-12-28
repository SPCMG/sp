import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import clip
from omegaconf import OmegaConf
import wandb
from utils.setup_wandb import setup_wandb_and_logging

from configs.config import load_config
from data.dataset import HumanML3D, custom_collate_fn
from models.model import MotionClipModel


def train_one_epoch(model, dataloader, optimizer, device="cuda"):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # 1. Prepare motion data
        x = batch["x"].to(device)          # [bs, max_len, 263]
        mask = batch["mask"].to(device)    # [bs, max_len]
        lengths = batch["lengths"].to(device)
        
        motion_batch = {
            "x": x,
            "mask": mask,
            "lengths": lengths
        }

        # 2. Encode motion embeddings
        motion_embs = model.encode_motion(motion_batch)  # [bs, latent_dim]

        # 3. Initialize lists for text embeddings and motion IDs
        text_embs_list = []
        shuffled_text_embs_list = []
        motion_ids_tensor = []
        bs = x.size(0)

        # 4. Process each sample in the batch
        for i in range(bs):
            captions = batch["captions"][i]  # Original captions for the motion
            shuffled_event_texts = batch["shuffled_event_texts"][i]  # Shuffled event texts for this motion

            # Encode original captions
            text_embs = model.encode_texts_for_one_item(captions)  # [num_captions, latent_dim]
            text_embs_list.append(text_embs)

            # Encode shuffled event texts
            # if shuffled_event_texts:
            #     shuffled_text_embs = model.encode_texts_for_one_item(shuffled_event_texts)
            #     shuffled_text_embs_list.append(shuffled_text_embs)

            # Assign a numeric motion ID
            motion_ids_tensor.append(hash(batch["motion_id"][i]) % 100000)

        # Convert motion_ids_tensor to a tensor on the device
        motion_ids_tensor = torch.tensor(motion_ids_tensor, dtype=torch.long, device=device)

        # 5. Compute loss
        loss = model.compute_motion_text_alignment_loss(
            motion_embs, text_embs_list, shuffled_text_embs_list, motion_ids_tensor
        )

        # 6. Backward pass and optimization step
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / max(num_batches, 1)
    return average_loss


def validate_one_epoch(model, dataloader, device="cuda"):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 1. Prepare motion data
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)

            motion_batch = {
                "x": x,
                "mask": mask,
                "lengths": lengths
            }

            # 2. Encode motion embeddings
            motion_embs = model.encode_motion(motion_batch)

            # 3. Initialize lists for text embeddings and motion IDs
            text_embs_list = []
            shuffled_text_embs_list = []
            motion_ids_tensor = []
            bs = x.size(0)

            # 4. Process each sample in the batch
            for i in range(bs):
                captions = batch["captions"][i]
                shuffled_event_texts = batch["shuffled_event_texts"][i]

                text_embs = model.encode_texts_for_one_item(captions)
                text_embs_list.append(text_embs)

                if any(text != "" for text in shuffled_event_texts):
                    shuffled_text_embs = model.encode_texts_for_one_item([text for text in shuffled_event_texts if text != ""])
                    shuffled_text_embs_list.append(shuffled_text_embs)

                motion_ids_tensor.append(hash(batch["motion_id"][i]) % 100000)

            motion_ids_tensor = torch.tensor(motion_ids_tensor, dtype=torch.long, device=device)

            # 5. Compute loss
            loss = model.compute_motion_text_alignment_loss(
                motion_embs, text_embs_list, shuffled_text_embs_list, motion_ids_tensor
            )

            total_loss += loss.item()
            num_batches += 1

    average_loss = total_loss / max(num_batches, 1)
    return average_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config("configs/config.yaml")

    # 1) Setup wandb
    run_name = setup_wandb_and_logging(cfg)  # returns the run name

    # 2) Prepare dataset / dataloader
    train_dataset = HumanML3D(opt=cfg.data, split="train")
    val_dataset = HumanML3D(opt=cfg.data, split="val")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.train.shuffle,
        num_workers=cfg.train.num_workers,
        collate_fn=custom_collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=custom_collate_fn,
        drop_last=True
    )

    # 3) Initialize model
    model = MotionClipModel(
        n_feats=cfg.data.n_feats,
        num_frames=cfg.data.max_motion_length,
        latent_dim=cfg.model.latent_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        ff_size=cfg.model.ff_size,
        dropout=cfg.model.dropout,
        clip_model_name=cfg.model.clip_model_name,
        activation=cfg.model.activation,
        temperature=cfg.loss.temperature,
        margin=cfg.loss.margin,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.motion_encoder.parameters(), lr=cfg.train.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    # 4) Create a run-specific checkpoint directory
    #    For example: ./checkpoints/<run_name>
    checkpoint_dir = os.path.join(cfg.checkpoint.save_path, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 5) Train loop
    for epoch in range(1, cfg.train.num_epochs + 1):
        print(f"Epoch {epoch}/{cfg.train.num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device=device)
        val_loss = validate_one_epoch(model, val_loader, device=device)

        # Step the scheduler with validation loss
        scheduler.step(val_loss)

        # Get current LR from the optimizer
        current_lr = optimizer.param_groups[0]['lr']

        # 6) Print to console
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # 7) Log to wandb
        wandb.log({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Learning Rate": current_lr
        })

        # 8) Save checkpoint
        if epoch % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"motionclipmodel_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}\n{'-'*50}")

    # 9) Mark the run as finished
    wandb.finish()


if __name__ == "__main__":
    main()
