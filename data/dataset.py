import random
import numpy as np
from os.path import join as pjoin
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from itertools import permutations
import torch.nn as nn

class HumanML3D(Dataset):
    def __init__(self, opt, split="train"):
        """
        Args:
            opt: OmegaConf DictConfig with fields such as:
                - data_root
                - motion_dir
                - text_dir
                - min_motion_length
                - max_motion_length
                - n_joints
                - n_feats
            split: 'train', 'val', or 'test'.
        """
        self.opt = opt
        self.split = split

        # Load normalization parameters
        self.mean = np.load(pjoin(opt.data_root, "Mean.npy"))
        self.std = np.load(pjoin(opt.data_root, "Std.npy"))

        self.min_motion_length = opt.min_motion_length
        self.max_motion_length = opt.max_motion_length

        self.n_joints = opt.n_joints  # e.g., 22
        self.n_feats = opt.n_feats    # e.g., 263 (raw features)

        # Load data file list
        split_file = pjoin(opt.data_root, f"{split}.txt")
        with open(split_file, "r") as f:
            self.data_files = [line.strip() for line in f.readlines()]

        # Preload data into memory
        self.data_dict = self._load_data()

    def _load_data(self):
        data_dict = {}
        for name in tqdm(self.data_files, desc=f"Loading {self.split} data"):
            motion_path = pjoin(self.opt.motion_dir, f"{name}.npy")
            text_path = pjoin(self.opt.text_dir, f"{name}.txt")
            event_text_dir = pjoin(self.opt.event_text_dir)

            try:
                motion = np.load(motion_path)
                if len(motion) < self.min_motion_length or len(motion) > self.max_motion_length:
                    continue

                with open(text_path, "r") as f:
                    captions = [line.split("#")[0].strip() for line in f if line.strip()]

                shuffled_event_texts = []
                for i, cap in enumerate(captions):
                    event_path = pjoin(event_text_dir, f"{name}_{i}.json")
                    with open(event_path, "r") as fjson:
                        event_data = json.load(fjson)
                        events = event_data.get("events", [])
                        original_text = " ".join(events).lower()

                        # Generate all permutations of the events
                        all_permutations = list(permutations(events))
                        random.shuffle(all_permutations)

                        # Create shuffled texts excluding the original order
                        shuffled_texts = [
                            " ".join(perm).lower() for perm in all_permutations if " ".join(perm).lower() != original_text
                        ]

                        shuffled_event_texts.extend(shuffled_texts)

                # Take the first 5 shuffled texts or pad with empty strings if less than 5
                shuffled_event_texts = shuffled_event_texts[:5] + [""] * (5 - len(shuffled_event_texts))

                if len(captions) > 0:
                    data_dict[name] = {
                        "motion": motion,
                        "captions": captions,
                        "shuffled_event_texts": shuffled_event_texts,
                        "motion_id": name,
                    }
            except FileNotFoundError:
                print(f"File not found: {name}")
                continue

        return data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        # Retrieve data sample
        key = list(self.data_dict.keys())[idx]
        item = self.data_dict[key]

        # motion_raw: [T, D], D = 263
        # T: The number of frames in the motion sequence. Each frame represents a snapshot of the motion at a given point in time.
        # D: The total number of features for each frame, encompassing 4 root features, 21 * 9 = 189 joint positions/rotations, 22 * 3 = 66 local velocities, and 4 foot contacts
        motion_raw = item["motion"]
        captions = item["captions"]
        shuffled_event_texts = item["shuffled_event_texts"]
        motion_id = item["motion_id"]

        motion_raw = (motion_raw - self.mean) / self.std
        motion, mask, length = self._pad_or_truncate_motion(motion_raw)

        print(captions)
        print(shuffled_event_texts)

        sample = {
            "x": motion,
            "mask": mask,
            "lengths": length,
            "captions": captions,
            "shuffled_event_texts": shuffled_event_texts,
            "motion_id": motion_id
        }
        return sample

    def _pad_or_truncate_motion(self, motion_raw):
        """
        Args:
            motion_raw: [T, 263]
        Returns:
            motion_padded: [max_motion_length, 263]
            mask: [max_motion_length] boolean array
            length: int
        """
        T = motion_raw.shape[0]
        D = motion_raw.shape[1]
        max_len = self.max_motion_length

        # Truncate if necessary
        if T > max_len:
            motion_raw = motion_raw[:max_len]
            T = max_len

        # Pad if necessary
        if T < max_len:
            pad = np.zeros((max_len - T, D), dtype=motion_raw.dtype)
            motion_padded = np.concatenate([motion_raw, pad], axis=0)  # shape [max_len, 263]
        else:
            motion_padded = motion_raw  # shape [max_len, 263]

        # Create mask
        mask = np.zeros((max_len,), dtype=bool)
        mask[:T] = True

        return motion_padded, mask, T

# Default collate_fn
from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    """
    Custom collate function to handle list of dicts.
    Converts numpy arrays to torch tensors automatically.
    """
    return default_collate(batch)
