import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, DistilBertTokenizer
import wandb
import os
import argparse

from dataset import AnchorDataset, anchor_collate_fn
from log_utils import setup_wandb_and_logging
from transformers import DistilBertModel
from omegaconf import OmegaConf

def loss(anchor_emb, pos_embs, neg_embs, margin=0.3):
    pos_loss = 1 - torch.nn.functional.cosine_similarity(pos_embs, anchor_emb, dim=1).mean()
    neg_loss = torch.nn.functional.triplet_margin_loss(anchor_emb, pos_embs.mean(dim=0, keepdim=True), 
                                                       neg_embs.mean(dim=0, keepdim=True), margin=margin)
    return pos_loss + neg_loss

def main():
    config = OmegaConf.load("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = setup_wandb_and_logging(config)

    save_dir = os.path.join(config.checkpoint.save_path, run_name)
    os.makedirs(save_dir, exist_ok=True)

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

    model = DistilBertModel.from_pretrained(config.model.pretrained_name).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained(config.model.pretrained_name)

    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * len(train_loader) * config.train.num_epochs), num_training_steps=len(train_loader) * config.train.num_epochs)

    best_loss = float("inf")

    for epoch in range(config.train.num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            loss_per_batch = []

            for i in range(len(batch["anchor"])):
                anchor_data = batch["anchor"][i]
                pos_data = batch["pos_batch"][i]
                neg_data = batch["neg_batch"][i]

                anchor_emb = model(anchor_data["input_ids"].unsqueeze(0).to(device), anchor_data["attention_mask"].unsqueeze(0).to(device)).last_hidden_state[:, 0, :]
                pos_embs = model(torch.stack([x["input_ids"] for x in pos_data]).to(device), torch.stack([x["attention_mask"] for x in pos_data]).to(device)).last_hidden_state[:, 0, :]
                neg_embs = model(torch.stack([x["input_ids"] for x in neg_data]).to(device), torch.stack([x["attention_mask"] for x in neg_data]).to(device)).last_hidden_state[:, 0, :]

                loss_per_batch.append(loss(anchor_emb, pos_embs, neg_embs, margin=config.train.margin))

            total_loss = torch.stack(loss_per_batch).mean()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            wandb.log({"batch_loss": total_loss.item()})

        epoch_loss /= len(train_loader)
        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}")

        wandb.log({"epoch": epoch + 1, "epoch_loss": epoch_loss})

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"[*] Saving best model at {save_dir}")
            model.save_pretrained(save_dir)  
            tokenizer.save_pretrained(save_dir)

    print("[*] Training complete.")

    wandb.finish()

if __name__ == "__main__":
    main()
