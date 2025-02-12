import json
import random
import torch
from torch.utils.data import Dataset
from transformers import CLIPTokenizer


class AnchorDataset(Dataset):
    """
    Each item corresponds to one BABEL label as the anchor.
    The positives are all sentences belonging to that label.
    The negatives are three randomly selected different labels.
    """
    def __init__(self, json_path, pretrained_name="openai/clip-vit-base-patch32",
                 max_length=64, random_seed=42, num_negatives=3):
        super().__init__()
        random.seed(random_seed)
        self.num_negatives = num_negatives

        # Load JSON
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Store label names
        self.labels = sorted(list(self.data.keys()))  
        self.label_indices = list(range(len(self.labels)))

        # Initialize tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.label_indices)

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        label_name = self.labels[idx]
        pos_sents = self.data[label_name]

        # Tokenize anchor (BABEL label name)
        anchor = self._tokenize(label_name)
        anchor = {
            "input_ids": anchor["input_ids"].squeeze(0),
            "attention_mask": anchor["attention_mask"].squeeze(0)
        }

        # Tokenize positives (sentences from the same label)
        pos_batch = []
        for sent in pos_sents:
            tokenized = self._tokenize(sent)
            pos_batch.append({
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0)
            })

        # Choose 3 random different labels for negatives
        negative_labels = random.sample([l for l in self.labels if l != label_name], self.num_negatives)
        neg_batch = []
        for neg_label in negative_labels:
            tokenized = self._tokenize(neg_label)
            neg_batch.append({
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0)
            })

        return {
            "anchor": anchor,
            "pos_batch": pos_batch,
            "neg_batch": neg_batch
        }


def anchor_collate_fn(batch):
    anchors = []
    pos_batches = []
    neg_batches = []

    for item in batch:
        anchors.append(item["anchor"])
        pos_batches.append(item["pos_batch"])
        neg_batches.append(item["neg_batch"])

    return {
        "anchor": anchors,
        "pos_batch": pos_batches,
        "neg_batch": neg_batches
    }
