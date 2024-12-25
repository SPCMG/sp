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
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Get batch size for this batch
        current_batch_size = batch["x"].size(0)
        print(f"\nBatch {batch_idx} size: {current_batch_size}")
        # Print number of captions
        print(f"Number of captions per sample: {[len(caps) for caps in batch['captions']]}")

        optimizer.zero_grad()

        # 1) Prepare motion data
        x = batch["x"].to(device)          # [bs, max_len, 263]
        mask = batch["mask"].to(device)    # [bs, max_len]
        lengths = batch["lengths"].to(device)
        # motion_ids: list of length [bs] (string or int)
        motion_ids_list = batch["motion_id"]  # e.g. [ '00001', '00002', ... ]

        # 2) Build motion_batch dict
        motion_batch = {
            "x": x,
            "mask": mask,
            "lengths": lengths,
            "y": torch.zeros_like(lengths).to(device)
        }

        # 3) Encode motion once per item => motion_embs
        motion_embs = model.encode_motion(motion_batch)  # shape [bs, latent_dim]

        # 4) For text, we have a *list of captions* for each item
        #    We'll build text_embs_list: a list of Tensors [num_captions_i, latent_dim]
        text_embs_list = []
        motion_ids_tensor = []
        bs = x.size(0)

        for i in range(bs):
            # all captions for sample i
            captions_i = batch["captions"][i]
            text_emb_i = model.encode_texts_for_one_item(captions_i)  # shape [num_captions_i, 512]

            # store
            text_embs_list.append(text_emb_i)
            
            # motion_id for sample i => parse to an int or keep as string. 
            # for the new loss, we want a numeric device-friendly ID
            # if it's a string, let's do a quick hash or parse to int:
            motion_id_i = hash(motion_ids_list[i]) % 100000  # or int(motion_ids_list[i]) if they are numeric
            motion_ids_tensor.append(motion_id_i)

        # convert motion_ids_tensor => Tensor on device
        motion_ids_tensor = torch.tensor(motion_ids_tensor, dtype=torch.long, device=device)

        # 5) Compute multi-caption loss
        #    motion_embs => [bs, latent_dim]
        #    text_embs_list => list of [num_captions_i, latent_dim]
        #    motion_ids_tensor => [bs]
        loss = model.compute_multicap_loss(motion_embs, text_embs_list, motion_ids_tensor)

        # 6) Backward + step
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        print(f"Batch {batch_idx} loss: {loss.item():.4f}")

    average_loss = total_loss / max(num_batches, 1)
    return average_loss


def validate_one_epoch(model, dataloader, device="cuda"):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)
            motion_ids_list = batch["motion_id"]

            motion_batch = {
                "x": x,
                "mask": mask,
                "lengths": lengths,
                "y": torch.zeros_like(lengths).to(device)
            }
            motion_embs = model.encode_motion(motion_batch)

            text_embs_list = []
            motion_ids_tensor = []
            bs = x.size(0)

            for i in range(bs):
                captions_i = batch["captions"][i]
                text_emb_i = model.encode_texts_for_one_item(captions_i)
                text_embs_list.append(text_emb_i)

                motion_id_i = hash(motion_ids_list[i]) % 100000
                motion_ids_tensor.append(motion_id_i)

            motion_ids_tensor = torch.tensor(motion_ids_tensor, dtype=torch.long, device=device)

            loss = model.compute_multicap_loss(motion_embs, text_embs_list, motion_ids_tensor)
            total_loss += loss.item()
            num_batches += 1

    average_loss = total_loss / max(num_batches, 1)
    return average_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config("configs/config.yaml")

    train_dataset = HumanML3D(opt=cfg.data, split="train")
    print(f"Total training samples: {len(train_dataset)}")
    val_dataset = HumanML3D(opt=cfg.data, split="val")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.train.shuffle,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )

    model = MotionClipModel(
        n_feats=cfg.data.n_feats,
        num_frames=cfg.data.max_motion_length,
        latent_dim=cfg.model.latent_dim,
        clip_model_name=cfg.model.clip_model_name,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.motion_encoder.parameters(), lr=cfg.train.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    checkpoint_dir = cfg.checkpoint.save_path
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, cfg.train.num_epochs + 1):
        print(f"Epoch {epoch}/{cfg.train.num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device=device)
        print(f"Train Loss: {train_loss:.4f}")

        val_loss = validate_one_epoch(model, val_loader, device=device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Step the scheduler with validation loss
        scheduler.step(val_loss)
        # Optional: Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"motionclipmodel_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}\n{'-'*50}")


if __name__ == "__main__":
    main()
