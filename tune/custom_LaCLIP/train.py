import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    get_linear_schedule_with_warmup,
    CLIPTokenizer  
)
import wandb
import os
import argparse

from dataset import AnchorDataset, anchor_collate_fn
from log_utils import setup_wandb_and_logging
from model import ClipTextEncoder 
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F


def triplet_loss(anchor_emb, pos_embs, neg_embs, orig_anchor_emb, margin=0.3):
    """
    anchor_emb: [1, dim]        (the anchor embedding)
    pos_embs:   [P, dim]        (multiple positive embeddings)
    neg_embs:   [N, dim]        (multiple negative embeddings)
    margin:     float           margin for triplet loss

    We'll do a 'hardest-positive' and 'hardest-negative' approach:
        - hardest positive = one that is farthest from the anchor
        - hardest negative = one that is closest to the anchor

    Using 1 - cos_sim as a distance measure:
        dist(a, b) = 1 - (a·b).
    """

    # 1) Normalize so that dot-product is the cosine similarity
    anchor_emb = F.normalize(anchor_emb, p=2, dim=-1)    # [1, dim]
    pos_embs   = F.normalize(pos_embs,   p=2, dim=-1)    # [P, dim]
    neg_embs   = F.normalize(neg_embs,   p=2, dim=-1)    # [N, dim]

    # 2) Compute similarities with anchor
    pos_sim = torch.matmul(anchor_emb, pos_embs.T).squeeze(0)  
    neg_sim = torch.matmul(anchor_emb, neg_embs.T).squeeze(0)

    # 3) Hardest positive = the one that has the smallest cos sim => largest distance
    #    => we take min(pos_sim) to find the "most distant" positive
    pos_loss = 1.0 - pos_sim.min()  # hardest positive distance

    # 4) Hardest negative = the one that has the largest cos sim => smallest distance
    #    => we take max(neg_sim) to find the "closest" negative
    neg_loss = margin + neg_sim.max()  # hardest negative distance

    # 5) Consistency loss: anchor_emb should be close to the original anchor embedding
    consistency_loss = F.mse_loss(anchor_emb, orig_anchor_emb)

    return 0.5 * pos_loss + 0.5 * consistency_loss
    # return 0.5 * pos_loss + 0.5 * neg_loss + 0.5 * consistency_loss

def main():
    # 1) Load config
    config = OmegaConf.load("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_decay", type=float, default=config.train.weight_decay, help="Weight decay")
    parser.add_argument("--learning_rate", type=float, default=config.train.learning_rate, help="Learning rate")
    args = parser.parse_args()

    # Update config with user-provided values
    config.train.weight_decay = args.weight_decay
    config.train.learning_rate = args.learning_rate

    # 2) Setup wandb logging
    run_name = setup_wandb_and_logging(config)
    save_dir = os.path.join(config.checkpoint.save_path, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # 3) Create dataset & dataloader
    train_dataset = AnchorDataset(
        json_path=config.data.json_path,
        pretrained_name=config.model.pretrained_name,  
        max_length=config.data.max_length,             
        random_seed=config.data.random_seed,
        num_negatives=config.train.num_negatives
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=anchor_collate_fn
    )

    # 4) Initialize the CLIP text encoder
    model = ClipTextEncoder(
        hf_pretrained_name=config.model.pretrained_name,
        pretrained_ckpt_path=config.model.pretrained_ckpt_path,
        dropout=config.model.dropout
    ).to(device)

    # 5) Use a CLIP tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(config.model.pretrained_name)

    # 6) Optimizer & scheduler
    optimizer = AdamW(model.clip_model.text_model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, 
    #     num_warmup_steps=int(0.2 * len(train_loader) * config.train.num_epochs),  
    #     num_training_steps=len(train_loader) * config.train.num_epochs
    # )

    # [ADDED] STEP A: Precompute original anchor embeddings
    # -----------------------------------------------------
    print("[*] Precomputing original anchor embeddings...")
    model.eval()  # Turn off dropout, etc.

    anchor_original_embs = {}
    with torch.no_grad():
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            # anchor item from dataset has "id" = i, so:
            anchor_id = item["anchor"]["id"]  # or just i
            input_ids = item["anchor"]["input_ids"].unsqueeze(0).to(device)
            attn_mask = item["anchor"]["attention_mask"].unsqueeze(0).to(device)
            emb = model(input_ids, attn_mask)  # [1, dim]
            anchor_original_embs[anchor_id] = emb.cpu()  # store on CPU

    model.train()  # Re-enable training mode
    # -----------------------------------------------------
    
    best_loss = float("inf")

    # 7) Training loop
    for epoch in range(config.train.num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_data in train_loader:
            loss_per_batch = []

            for i in range(len(batch_data["anchor"])):
                anchor_data = batch_data["anchor"][i]
                pos_data = batch_data["pos_batch"][i]
                neg_data = batch_data["neg_batch"][i]

                # [ADDED] retrieve anchor_id so we can do consistency
                anchor_id = anchor_data["id"]

                # Forward pass for anchor
                anchor_emb = model(
                    anchor_data["input_ids"].unsqueeze(0).to(device),
                    anchor_data["attention_mask"].unsqueeze(0).to(device)
                )

                # Forward pass for positives (stack them into a single batch)
                pos_input_ids = torch.stack([x["input_ids"] for x in pos_data]).to(device)
                pos_attn_mask = torch.stack([x["attention_mask"] for x in pos_data]).to(device)
                pos_embs = model(pos_input_ids, pos_attn_mask)

                # Forward pass for negatives
                if len(neg_data) > 0:
                    neg_input_ids = torch.stack([x["input_ids"] for x in neg_data]).to(device)
                    neg_attn_mask = torch.stack([x["attention_mask"] for x in neg_data]).to(device)
                    neg_embs = model(neg_input_ids, neg_attn_mask)
                else:
                    neg_embs = torch.zeros_like(pos_embs)

                # Compute triplet-style loss
                orig_anchor_emb = anchor_original_embs[anchor_id].to(device)  # shape [1, dim]
                batch_loss = triplet_loss(
                    anchor_emb, 
                    pos_embs, 
                    neg_embs, 
                    orig_anchor_emb,
                    margin=config.train.margin
                )
                loss_per_batch.append(batch_loss)

            total_loss = torch.stack(loss_per_batch).mean()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(total_loss)

            epoch_loss += total_loss.item()
            wandb.log({"batch_loss": total_loss.item()})

        epoch_loss /= len(train_loader)
        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "epoch_loss": epoch_loss})

        # 8) Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"[*] Saving best model at {save_dir}")
            model_state_dir = os.path.join(save_dir, "model")
            os.makedirs(model_state_dir, exist_ok=True)

            # Save model's state_dict
            torch.save(model.state_dict(), os.path.join(model_state_dir, "pytorch_model.bin"))
            # Optionally, save tokenizer
            tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

    print("[*] Training complete.")
    wandb.finish()

if __name__ == "__main__":
    main()
