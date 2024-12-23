import random
import numpy as np
from os.path import join as pjoin
from torch.utils.data import Dataset
from tqdm import tqdm

class HumanML3D(Dataset):
    def __init__(self, opt, split="train"):
        """
        Args:
            opt: namespace or dict with fields such as:
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

        # e.g., for normalization
        self.mean = np.load(pjoin(opt.data_root, "Mean.npy"))
        self.std = np.load(pjoin(opt.data_root, "Std.npy"))

        self.min_motion_length = opt.min_motion_length
        self.max_motion_length = opt.max_motion_length

        # Suppose your motion is shaped [T, D], 
        # and you eventually want to reshape it to [n_joints, n_feats, T]
        self.n_joints = opt.n_joints  # e.g., 24
        self.n_feats = opt.n_feats    # e.g., 6 (xyz + xyz velocity?), or 3, etc.

        # For loading data
        split_file = pjoin(opt.data_root, f"{split}.txt")
        with open(split_file, "r") as f:
            self.data_files = [line.strip() for line in f.readlines()]

        # Preload data
        self.data_dict = self._load_data()

    def _load_data(self):
        data_dict = {}
        for name in tqdm(self.data_files, desc=f"Loading {self.split} data"):
            motion_path = pjoin(self.opt.motion_dir, f"{name}.npy")
            text_path = pjoin(self.opt.text_dir, f"{name}.txt")
            try:
                motion = np.load(motion_path)  # shape [T, D]
                # Filter out invalid motion lengths
                if len(motion) < self.min_motion_length or len(motion) > self.max_motion_length:
                    continue

                # We assume each .txt file has multiple lines, each with a caption
                # You said your file might have lines like "some caption # token1 token2 ..."
                # For CLIP, we only really need the raw caption text.
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
                    "motion": motion,    # shape [T, D]
                    "captions": text_data
                }
            except FileNotFoundError:
                continue

        return data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        # Pick an item
        key = list(self.data_dict.keys())[idx]
        item = self.data_dict[key]

        motion_raw = item["motion"]   # shape [T, D]
        captions = item["captions"]   # list of raw text strings
        caption = random.choice(captions)

        # Normalize motion
        motion_raw = (motion_raw - self.mean) / self.std  # shape [T, D]

        # Pad / slice motion
        motion, mask, length = self._pad_or_truncate_motion(motion_raw)

        # Return dictionary that matches MotionCLIP usage:
        #  - x         : [n_joints, n_feats, max_motion_length] (torch will handle shape)
        #  - mask      : [max_motion_length] boolean
        #  - lengths   : integer
        #  - clip_text : the raw caption string
        sample = {
            "x": motion,         # np array
            "mask": mask,        # np boolean array
            "lengths": length,   # int
            "clip_text": caption
        }
        return sample

    def _pad_or_truncate_motion(self, motion_raw):
        """
        motion_raw: [T, D]
        We'll:
         1) clip T to self.max_motion_length if T is too long
         2) zero-pad if T is too short
         3) reshape to [n_joints, n_feats, T_final]
         4) build a boolean mask of shape [T_final]
        """
        T = motion_raw.shape[0]
        D = motion_raw.shape[1]
        max_len = self.max_motion_length

        # If T > max_len, truncate
        if T > max_len:
            motion_raw = motion_raw[:max_len]
            T = max_len

        # Zero-padding if T < max_len
        if T < max_len:
            pad = np.zeros((max_len - T, D), dtype=motion_raw.dtype)
            motion_padded = np.concatenate([motion_raw, pad], axis=0)  # shape [max_len, D]
        else:
            motion_padded = motion_raw  # shape [max_len, D]

        # Now, mask shape = [max_len], True for real frames, False for padded
        mask = np.zeros((max_len,), dtype=bool)
        mask[:T] = True

        # Reshape from [max_len, D] to [n_joints, n_feats, max_len]
        # You must ensure that D == n_joints*n_feats
        assert D == self.n_joints * self.n_feats, (
            f"Dimension mismatch: D={D}, but n_joints*n_feats={self.n_joints*self.n_feats}."
        )
        motion_padded = motion_padded.reshape(max_len, self.n_joints, self.n_feats)
        motion_padded = motion_padded.transpose(1, 2, 0)  # => [n_joints, n_feats, max_len]

        return motion_padded, mask, T


################################
# Default collate_fn (optional)
################################
from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    # batch is a list of samples (each is a dict)
    # default_collate will convert all numpy arrays to torch tensors automatically
    return default_collate(batch)
