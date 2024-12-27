import random
import numpy as np
from os.path import join as pjoin
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from itertools import permutations
import os

class HumanML3D(Dataset):
    def __init__(self, opt, split="train"):
        """
        Args:
            opt: OmegaConf DictConfig with fields like data_root, motion_dir, etc.
            split: 'train', 'val', or 'test'.
        """
        self.opt = opt
        self.split = split

        # Load normalization parameters
        self.mean = np.load(pjoin(opt.data_root, "Mean.npy"))
        self.std = np.load(pjoin(opt.data_root, "Std.npy"))

        self.min_motion_length = opt.min_motion_length
        self.max_motion_length = opt.max_motion_length
        self.n_joints = opt.n_joints
        self.n_feats = opt.n_feats

        # Load file list
        split_file = pjoin(opt.data_root, f"{split}.txt")
        with open(split_file, "r") as f:
            self.data_files = [line.strip() for line in f.readlines()]

        # Instead of loading all motion/text data now,
        # just build a small dictionary that references paths or IDs.
        self.data_dict = self._build_index()

    def _build_index(self):
        data_index = {}
        for name in tqdm(self.data_files, desc=f"Indexing {self.split} data"):
            motion_path = pjoin(self.opt.motion_dir, f"{name}.npy")
            text_path = pjoin(self.opt.text_dir, f"{name}.txt")
            event_text_dir = self.opt.event_text_dir
            
            # 1) Check if motion file exists
            if not os.path.exists(motion_path):
                continue

            # 2) Quickly read the shape of the .npy file
            try:
                with np.load(motion_path, mmap_mode='r') as motion_temp:
                    T = motion_temp.shape[0]
            except:
                continue

            # 3) Filter out-of-range
            if T < self.min_motion_length or T > self.max_motion_length:
                continue

            # 4) If we get here, it's a valid motion => we store it in the index
            data_index[name] = {
                "motion_path": motion_path,
                "text_path": text_path,
                "event_text_dir": event_text_dir,
                "motion_id": name
            }

        return data_index

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        # 1) Retrieve paths/info for this item
        key = list(self.data_dict.keys())[idx]
        item_info = self.data_dict[key]

        motion_path = item_info["motion_path"]
        text_path = item_info["text_path"]
        event_text_dir = item_info["event_text_dir"]
        motion_id = item_info["motion_id"]

        # 2) Load the motion .npy (only for this sample!)
        motion_raw = np.load(motion_path)

        # 3) Load captions from the text file
        with open(text_path, "r") as f:
            captions = [line.split("#")[0].strip() for line in f if line.strip()]

        # 4) For each caption, load event JSON and build permutations
        shuffled_event_texts = []
        for i, cap in enumerate(captions):
            event_path = pjoin(event_text_dir, f"{motion_id}_{i}.json")
            try:
                with open(event_path, "r") as fjson:
                    event_data = json.load(fjson)
                    events = event_data.get("events", [])
                    original_text = " ".join(events).lower()

                    # Create permutations
                    all_permutations = list(permutations(events))
                    random.shuffle(all_permutations)

                    # Exclude the original order
                    shuffled_texts = [
                        " ".join(perm).lower() for perm in all_permutations
                        if " ".join(perm).lower() != original_text
                    ]
                    shuffled_event_texts.extend(shuffled_texts)
            except FileNotFoundError:
                pass

        # 5) Truncate/pad shuffled event texts
        shuffled_event_texts = shuffled_event_texts[:5]
        if len(shuffled_event_texts) < 5:
            shuffled_event_texts += [""] * (5 - len(shuffled_event_texts))

        # 6) Normalize motion => (motion_raw - mean) / std
        motion_raw = (motion_raw - self.mean) / self.std

        # 7) Pad or truncate motion
        motion, mask, length = self._pad_or_truncate_motion(motion_raw)

        # 8) Return sample
        sample = {
            "x": motion,  # [max_motion_length, 263]
            "mask": mask,  # [max_motion_length] bool
            "lengths": length,
            "captions": captions,
            "shuffled_event_texts": shuffled_event_texts,
            "motion_id": motion_id
        }
        return sample

    def _pad_or_truncate_motion(self, motion_raw):
        T = motion_raw.shape[0]
        D = motion_raw.shape[1]
        max_len = self.max_motion_length

        if T > max_len:
            motion_raw = motion_raw[:max_len]
            T = max_len

        if T < max_len:
            pad = np.zeros((max_len - T, D), dtype=motion_raw.dtype)
            motion_padded = np.concatenate([motion_raw, pad], axis=0)
        else:
            motion_padded = motion_raw

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
