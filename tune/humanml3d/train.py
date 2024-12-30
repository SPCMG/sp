import torch 
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from os.path import join as pjoin
from omegaconf import OmegaConf
from dataset import HumanML3DDataset, collate_fn
from model import ClipTextEncoder
from loss import HumanML3DLoss
from tokenizer import SimpleTokenizer
from setup_wandb import setup_wandb_and_logging
import wandb
import os
import argparse

def train_one_epoch(model, loss_fn, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for anchors, pos_sames, neg_others, neg_sames in dataloader:
        optimizer.zero_grad()
        batch_loss = 0.0

        for i in range(len(anchors)):
            a_ids, a_mask = anchors[i]
            a_ids = a_ids.unsqueeze(0).to(device)
            a_mask = a_mask.unsqueeze(0).to(device)

            anchor_emb = model(a_ids, a_mask)  # (1, dim)

            # Pos same
            p_ids, p_masks = pos_sames[i]
            if len(p_ids) > 0:
                p_ids = torch.stack(p_ids, dim=0).to(device)
                p_masks = torch.stack(p_masks, dim=0).to(device)
                pos_emb = model(p_ids, p_masks)
            else:
                pos_emb = torch.zeros((0, anchor_emb.shape[-1])).to(device)

            # Neg other
            no_ids, no_masks = neg_others[i]
            if len(no_ids) > 0:
                no_ids = torch.stack(no_ids, dim=0).to(device)
                no_masks = torch.stack(no_masks, dim=0).to(device)
                neg_other_emb = model(no_ids, no_masks)
            else:
                neg_other_emb = torch.zeros((0, anchor_emb.shape[-1])).to(device)

            # Neg same
            ns_ids, ns_masks = neg_sames[i]
            if len(ns_ids) > 0:
                ns_ids = torch.stack(ns_ids, dim=0).to(device)
                ns_masks = torch.stack(ns_masks, dim=0).to(device)
                neg_same_emb = model(ns_ids, ns_masks)
            else:
                neg_same_emb = torch.zeros((0, anchor_emb.shape[-1])).to(device)

            loss_val, _ = loss_fn(anchor_emb, pos_emb, neg_other_emb, neg_same_emb)
            batch_loss += loss_val

        if len(anchors) > 0:
            batch_loss = batch_loss / len(anchors)

        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    return total_loss / len(dataloader)

def validate_one_epoch(model, loss_fn, dataloader, device):
    """
    Runs a forward pass on the validation set, computes average val loss.
    No gradients or optimizer steps here.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for anchors, pos_sames, neg_others, neg_sames in dataloader:
            batch_loss = 0.0

            for i in range(len(anchors)):
                a_ids, a_mask = anchors[i]
                a_ids = a_ids.unsqueeze(0).to(device)
                a_mask = a_mask.unsqueeze(0).to(device)

                anchor_emb = model(a_ids, a_mask)

                p_ids, p_masks = pos_sames[i]
                if len(p_ids) > 0:
                    p_ids = torch.stack(p_ids, dim=0).to(device)
                    p_masks = torch.stack(p_masks, dim=0).to(device)
                    pos_emb = model(p_ids, p_masks)
                else:
                    pos_emb = torch.zeros((0, anchor_emb.shape[-1])).to(device)

                no_ids, no_masks = neg_others[i]
                if len(no_ids) > 0:
                    no_ids = torch.stack(no_ids, dim=0).to(device)
                    no_masks = torch.stack(no_masks, dim=0).to(device)
                    neg_other_emb = model(no_ids, no_masks)
                else:
                    neg_other_emb = torch.zeros((0, anchor_emb.shape[-1])).to(device)

                ns_ids, ns_masks = neg_sames[i]
                if len(ns_ids) > 0:
                    ns_ids = torch.stack(ns_ids, dim=0).to(device)
                    ns_masks = torch.stack(ns_masks, dim=0).to(device)
                    neg_same_emb = model(ns_ids, ns_masks)
                else:
                    neg_same_emb = torch.zeros((0, anchor_emb.shape[-1])).to(device)

                loss_val, _ = loss_fn(anchor_emb, pos_emb, neg_other_emb, neg_same_emb)
                batch_loss += loss_val

            if len(anchors) > 0:
                batch_loss = batch_loss / len(anchors)

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
        split_file=pjoin(config.data.data_root, "train.txt"),
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        num_other_negatives=config.data.num_other_negatives,
        random_seed=config.data.random_seed
    )

    # 4) Create VAL dataset
    val_dataset = HumanML3DDataset(
        text_dir=config.data.text_dir,
        negative_text_dir=config.data.negative_text_dir,
        split_file=pjoin(config.data.data_root, "val.txt"),
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        num_other_negatives=config.data.num_other_negatives,
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
    model = ClipTextEncoder(config.model.pretrained_name, config.model.ckpt_path).to(device)
    loss_fn = HumanML3DLoss(config.loss.margin, config.loss.alpha, config.loss.beta)
    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.train.scheduler_factor, patience=config.train.scheduler_patience, verbose=True)

    # 7) Create a run-specific checkpoint directory
    checkpoint_dir = os.path.join(config.checkpoint.save_path, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

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
        if epoch % config.checkpoint.save_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)  
            checkpoint_path = os.path.join(checkpoint_dir, f"finetunelaclip_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}\n{'-'*50}")

    wandb.finish()


if __name__ == "__main__":
    main()


