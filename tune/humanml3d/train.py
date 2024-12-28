import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from dataset import HumanML3DDataset
from model import ClipTextEncoder
from loss import HumanML3DLoss  # same as previously shown

def collate_fn(batch):
    anchors = []
    pos_sames = []
    neg_others = []
    neg_sames = []

    for item in batch:
        if (item['anchor_tokens'] is None):
            # skip empty item
            continue

        a_ids = item['anchor_tokens']['input_ids'].squeeze(0)
        a_mask = item['anchor_tokens']['attention_mask'].squeeze(0)
        anchors.append((a_ids, a_mask))

        # Positives
        p_tokens = item['positives_same_motion']
        p_ids = [t['input_ids'].squeeze(0) for t in p_tokens]
        p_masks = [t['attention_mask'].squeeze(0) for t in p_tokens]
        pos_sames.append((p_ids, p_masks))

        # Negatives (other motion)
        no_tokens = item['negatives_other_motion']
        no_ids = [t['input_ids'].squeeze(0) for t in no_tokens]
        no_masks = [t['attention_mask'].squeeze(0) for t in no_tokens]
        neg_others.append((no_ids, no_masks))

        # Negatives (same motion swapped)
        ns_tokens = item['negatives_same_motion']
        ns_ids = [t['input_ids'].squeeze(0) for t in ns_tokens]
        ns_masks = [t['attention_mask'].squeeze(0) for t in ns_tokens]
        neg_sames.append((ns_ids, ns_masks))

    return anchors, pos_sames, neg_others, neg_sames

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_dir = "/path/to/HumanML3D/texts"
    neg_text_dir = "/path/to/HumanML3D/neg_texts"

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dataset = HumanML3DDataset(
        text_dir=text_dir,
        negative_text_dir=neg_text_dir,
        tokenizer=tokenizer,
        max_length=77,
        num_other_negatives=3,  # sample 3 lines from other motions
        random_seed=42
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,    # multiple workers for parallel I/O
        collate_fn=collate_fn
    )

    model = ClipTextEncoder("openai/clip-vit-base-patch32").to(device)
    loss_fn = HumanML3DLoss(margin=0.2, alpha=1.0, beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    epochs = 5
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, loss_fn, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "clip_text_encoder_finetuned.pt")

if __name__ == "__main__":
    main()
