import random
import numpy as np
from os.path import join as pjoin
from torch.utils.data import Dataset
from tqdm import tqdm

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
            try:
                motion = np.load(motion_path)  # shape [T, 263]
                # Filter out invalid motion lengths
                if len(motion) < self.min_motion_length or len(motion) > self.max_motion_length:
                    continue

                # Load captions
                with open(text_path, "r") as ftxt:
                    text_data = []
                    for line in ftxt:
                        line_split = line.strip().split("#")
                        if len(line_split) < 1:
                            continue
                        caption = line_split[0].strip()
                        if caption == "":
                            continue
                        text_data.append(caption)

                if len(text_data) == 0:
                    continue

                data_dict[name] = {
                    "motion": motion,          # shape [T, 263]
                    "captions": text_data,     # list of raw text strings
                    "motion_id": name          # or parse to int if you prefer
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
        captions = item["captions"]   # list of raw text strings
        motion_id = item["motion_id"] 
        print(f"Motion shape: {motion_raw.shape}, Captions: {captions}, Motion ID: {motion_id}")
        # caption = random.choice(captions)

        # Normalize motion data
        motion_raw = (motion_raw - self.mean) / self.std  # shape [T, 263]

        # Pad or truncate motion data
        motion, mask, length = self._pad_or_truncate_motion(motion_raw)

        # Construct sample dictionary
        sample = {
            "x": motion,               # np array [max_len, 263]
            "mask": mask,              # np boolean array [max_len]
            "lengths": length,         # int
            "captions": captions,      # list of raw strings
            "motion_id": motion_id     # unique identifier
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
