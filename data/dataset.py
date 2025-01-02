import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.text_process import read_lines

class MotionTripleDataset(Dataset):
    """
    Returns a triple for each motion:
      (motion_tensor, pos_caption, neg_caption)

    - 'pos_caption': a correct/matching caption (label=0)
    - 'neg_caption': a custom negative line (label=1), e.g. swapped actions
    """

    def __init__(
        self,
        motion_dir,
        text_dir,
        negative_text_dir,
        split_file,
        min_motion_len=40,
        max_motion_len=200,
        max_text_length=60,
        random_seed=42
    ):
        super().__init__()
        random.seed(random_seed)

        self.motion_dir = motion_dir
        self.text_dir = text_dir
        self.negative_text_dir = negative_text_dir
        self.min_motion_len = min_motion_len
        self.max_motion_len = max_motion_len
        self.max_text_length = max_text_length

        # Read the IDs from the split file
        with open(split_file, 'r') as f:
            all_ids = [line.strip() for line in f if line.strip()]

        self.samples = []
        skipped = 0
        for mid in all_ids:
            motion_path = os.path.join(motion_dir, f"{mid}.npy")
            if not os.path.isfile(motion_path):
                continue

            # Check motion length
            motion_data = np.load(motion_path)
            length = motion_data.shape[0]
            if not (self.min_motion_len <= length <= self.max_motion_len):
                skipped += 1
                continue

            # Read the positive lines
            pos_path = os.path.join(text_dir, f"{mid}.txt")
            pos_lines = read_lines(pos_path, self.max_text_length)
            if len(pos_lines) == 0:
                skipped += 1
                continue

            # Read the custom negative lines
            neg_path = os.path.join(negative_text_dir, f"{mid}.txt")
            neg_lines = read_lines(neg_path, self.max_text_length)
            if len(neg_lines) == 0:
                neg_lines = ["(no negative lines)"]

            # We choose exactly ONE positive line and ONE negative line
            # per motion to create a triple.
            pos_line = random.choice(pos_lines)
            neg_line = random.choice(neg_lines)

            self.samples.append({
                "motion_path": motion_path,
                "pos_text": pos_line,
                "neg_text": neg_line
            })

        print(f"[MotionTripleDataset] Loaded {len(self.samples)} samples total.")
        print(f"  Skipped {skipped} motions out of [min={self.min_motion_len}, max={self.max_motion_len}] range.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        motion_data = np.load(item["motion_path"])  # shape (T, D)
        motion_tensor = torch.from_numpy(motion_data).float()
        pos_text = item["pos_text"]
        neg_text = item["neg_text"]
        return motion_tensor, pos_text, neg_text


def collate_fn(batch):
    """
    Given a list of (motion_tensor, pos_text, neg_text),
    produce:
      motions -> List of length B, each a (T, D) tensor
      pos_texts -> list of length B (strings)
      neg_texts -> list of length B (strings)
    """
    motions = []
    pos_texts = []
    neg_texts = []

    for (motion_tensor, pos_str, neg_str) in batch:
        motions.append(motion_tensor)
        pos_texts.append(pos_str)
        neg_texts.append(neg_str)

    return motions, pos_texts, neg_texts
