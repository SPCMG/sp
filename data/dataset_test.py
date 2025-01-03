import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.text_process import read_lines

class MotionTripleDataset(Dataset):
    """
    Returns for each motion:
      (motion_tensor, positive_captions, negative_captions)

    - 'positive_captions': list of matching captions
    - 'negative_captions': list of shuffled/mismatched captions
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
        np.random.seed(random_seed)

        self.motion_dir = motion_dir
        self.text_dir = text_dir
        self.negative_text_dir = negative_text_dir
        self.min_motion_len = min_motion_len
        self.max_motion_len = max_motion_len
        self.max_text_length = max_text_length

        # Read IDs from the split file
        with open(split_file, 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]

        # Filter IDs by valid motion lengths
        self.samples = []
        for mid in self.ids:
            motion_path = os.path.join(motion_dir, f"{mid}.npy")
            if not os.path.isfile(motion_path):
                continue

            # Check motion length
            motion_data = np.load(motion_path)
            length = motion_data.shape[0]
            if not (self.min_motion_len <= length <= self.max_motion_len):
                continue

            # Read captions
            pos_path = os.path.join(text_dir, f"{mid}.txt")
            neg_path = os.path.join(negative_text_dir, f"{mid}.txt")
            pos_captions = read_lines(pos_path, self.max_text_length)
            neg_captions = read_lines(neg_path, self.max_text_length)

            if pos_captions:
                self.samples.append({
                    "motion_path": motion_path,
                    "positive_captions": pos_captions,
                    "negative_captions": neg_captions
                })

        print(f"[MotionDataset] Loaded {len(self.samples)} samples total.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        motion_data = np.load(item["motion_path"])
        motion_tensor = torch.from_numpy(motion_data).float()
        return motion_tensor, item["positive_captions"], item["negative_captions"]


def collate_fn(batch):
    """
    Given a list of (motion_tensor, positive_captions, negative_captions),
    produce:
      motions -> List of tensors
      positive_captions -> List of lists of strings
      negative_captions -> List of lists of strings
    """
    motions = []
    positive_captions = []
    negative_captions = []

    for motion_tensor, pos_caps, neg_caps in batch:
        motions.append(motion_tensor)
        positive_captions.append(pos_caps)
        negative_captions.append(neg_caps)

    return motions, positive_captions, negative_captions
