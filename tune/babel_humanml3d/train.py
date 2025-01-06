import torch 
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from os.path import join as pjoin
from omegaconf import OmegaConf
from dataset import HumanML3DDataset, collate_fn
from model import ClipTextEncoder
from loss import HumanML3DLoss
from tokenizer import SimpleTokenizer
from log_utils import setup_wandb_and_logging
import wandb
import os
import argparse

def train_one_epoch(model, loss_fn, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    # NEW: Now we have 4 items in the batch
    for pos_sames, neg_sames, pos_others, neg_others in dataloader:
        optimizer.zero_grad()
        batch_loss = 0.0

        for i in range(len(pos_sames)):
            # 1) Positives (same motion)
            p_ids, p_masks = pos_sames[i]
            if len(p_ids) > 0:
                p_ids = torch.stack(p_ids, dim=0).to(device)    # shape (P, seq_len)
                p_masks = torch.stack(p_masks, dim=0).to(device)
                pos_same_emb = model(p_ids, p_masks)            # shape (P, emb_dim)
            else:
                pos_same_emb = torch.zeros((0, model.clip_model.config.text_config.hidden_size)).to(device)

            # 2) Negatives (same motion)
            ns_ids, ns_masks = neg_sames[i]
            if len(ns_ids) > 0:
                ns_ids = torch.stack(ns_ids, dim=0).to(device) 
                ns_masks = torch.stack(ns_masks, dim=0).to(device)
                neg_same_emb = model(ns_ids, ns_masks)
            else:
                neg_same_emb = torch.zeros((0, model.clip_model.config.text_config.hidden_size)).to(device)

            # 3) Positives (other motion)
            po_ids, po_masks = pos_others[i]
            if len(po_ids) > 0:
                po_ids = torch.stack(po_ids, dim=0).to(device)
                po_masks = torch.stack(po_masks, dim=0).to(device)
                pos_other_emb = model(po_ids, po_masks)
            else:
                pos_other_emb = torch.zeros((0, model.clip_model.config.text_config.hidden_size)).to(device)

            # 4) Negatives (other motion)
            no_ids, no_masks = neg_others[i]
            if len(no_ids) > 0:
                no_ids = torch.stack(no_ids, dim=0).to(device)
                no_masks = torch.stack(no_masks, dim=0).to(device)
                neg_other_emb = model(no_ids, no_masks)
            else:
                neg_other_emb = torch.zeros((0, model.clip_model.config.text_config.hidden_size)).to(device)

            # --- Pass all embeddings to your updated loss ---
            loss_val, partials = loss_fn(
                pos_same_emb,
                neg_same_emb,
                pos_other_emb,
                neg_other_emb
            )
            batch_loss += loss_val

        # Average over number of samples in this batch
        if len(pos_sames) > 0:
            batch_loss = batch_loss / len(pos_sames)

        # Backprop
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    return total_loss / len(dataloader)

def validate_one_epoch(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        # NEW: Now we have 4 items in the batch
        for pos_sames, neg_sames, pos_others, neg_others in dataloader:
            batch_loss = 0.0

            for i in range(len(pos_sames)):
                # 1) Positives (same motion)
                p_ids, p_masks = pos_sames[i]
                if len(p_ids) > 0:
                    p_ids = torch.stack(p_ids, dim=0).to(device)    # shape (P, seq_len)
                    p_masks = torch.stack(p_masks, dim=0).to(device)
                    pos_same_emb = model(p_ids, p_masks)            # shape (P, emb_dim)
                else:
                    pos_same_emb = torch.zeros((0, model.clip_model.config.text_config.hidden_size)).to(device)

                # 2) Negatives (same motion)
                ns_ids, ns_masks = neg_sames[i]
                if len(ns_ids) > 0:
                    ns_ids = torch.stack(ns_ids, dim=0).to(device) 
                    ns_masks = torch.stack(ns_masks, dim=0).to(device)
                    neg_same_emb = model(ns_ids, ns_masks)
                else:
                    neg_same_emb = torch.zeros((0, model.clip_model.config.text_config.hidden_size)).to(device)

                # 3) Positives (other motion)
                po_ids, po_masks = pos_others[i]
                if len(po_ids) > 0:
                    po_ids = torch.stack(po_ids, dim=0).to(device)
                    po_masks = torch.stack(po_masks, dim=0).to(device)
                    pos_other_emb = model(po_ids, po_masks)
                else:
                    pos_other_emb = torch.zeros((0, model.clip_model.config.text_config.hidden_size)).to(device)

                # 4) Negatives (other motion)
                no_ids, no_masks = neg_others[i]
                if len(no_ids) > 0:
                    no_ids = torch.stack(no_ids, dim=0).to(device)
                    no_masks = torch.stack(no_masks, dim=0).to(device)
                    neg_other_emb = model(no_ids, no_masks)
                else:
                    neg_other_emb = torch.zeros((0, model.clip_model.config.text_config.hidden_size)).to(device)

                # --- Pass all embeddings to your updated loss ---
                loss_val, partials = loss_fn(
                    pos_same_emb,
                    neg_same_emb,
                    pos_other_emb,
                    neg_other_emb
                )
                batch_loss += loss_val

            # Average over number of samples in this batch
            if len(pos_sames) > 0:
                batch_loss = batch_loss / len(pos_sames)

            total_loss += batch_loss.item()

    return total_loss / len(dataloader)

def main():
    # 1) Load config
    config = OmegaConf.load("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_decay", type=float, default=config.train.weight_decay, help="Weight decay")
    parser.add_argument("--learning_rate", type=float, default=config.train.learning_rate, help="Learning rate")
    parser.add_argument("--scheduler_factor", type=float, default=config.train.scheduler_factor, help="Scheduler factor")
    args = parser.parse_args()

    # Update config with user-provided values
    config.train.weight_decay = args.weight_decay
    config.train.learning_rate = args.learning_rate
    config.train.scheduler_factor = args.scheduler_factor

    # Initialize wandb
    run_name = setup_wandb_and_logging(config)

    # 2) Initialize tokenizer
    tokenizer = SimpleTokenizer()

    # 3) Create TRAIN dataset
    train_dataset = HumanML3DDataset(
        text_dir=config.data.text_dir,
        negative_text_dir=config.data.negative_text_dir,
        motion_babel_json=config.data.motion_babel_json,
        babel_caption_json=config.data.babel_caption_json,
        split_file=pjoin(config.data.data_root, "train.txt"),
        tokenizer=tokenizer,
        max_text_length=config.data.max_text_length,
        num_other_negatives=config.data.num_other_negatives,
        num_other_positives=config.data.num_other_positives,
        random_seed=config.data.random_seed
    )

    # 4) Create VAL dataset
    val_dataset = HumanML3DDataset(
        text_dir=config.data.text_dir,
        negative_text_dir=config.data.negative_text_dir,
        motion_babel_json=config.data.motion_babel_json,
        babel_caption_json=config.data.babel_caption_json,
        split_file=pjoin(config.data.data_root, "val.txt"),
        tokenizer=tokenizer,
        max_text_length=config.data.max_text_length,
        num_other_negatives=config.data.num_other_negatives,
        num_other_positives=config.data.num_other_positives,
        random_seed=config.data.random_seed
    )

    # 5) DataLoaders
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

    # 6) Model, Loss, Optim
    model = ClipTextEncoder(config.model.pretrained_name, config.model.ckpt_path, config.model.dropout).to(device)
    loss_fn = loss_fn = HumanML3DLoss(margin=config.loss.margin, alpha1=config.loss.alpha1, alpha2=config.loss.alpha2, alpha3=config.loss.alpha3, alpha4=config.loss.alpha4)
    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.train.scheduler_factor, patience=config.train.scheduler_patience, verbose=True)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=1, eta_min=1e-7, last_epoch=-1)

    # 7) Create a run-specific checkpoint directory
    checkpoint_dir = os.path.join(config.checkpoint.save_path, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 8) Training Loop
    best_val_loss = float("inf")
    for epoch in range(config.train.num_epochs):
        # Train
        train_loss = train_one_epoch(model, loss_fn, train_loader, optimizer, device)
        # Validate
        val_loss = validate_one_epoch(model, loss_fn, val_loader, device)

        # Update the learning rate based on validation loss
        scheduler.step(val_loss)
        
        print(f"[Epoch {epoch+1}/{config.train.num_epochs}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to wandb
        wandb.log({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Learning Rate": optimizer.param_groups[0]['lr']
        })

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"finetunelaclip_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)
            print(f"  [*] Saved best model to {checkpoint_path}")

    wandb.finish()


if __name__ == "__main__":
    main()


