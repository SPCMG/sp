import os
import glob
import random
import torch
from torch.utils.data import Dataset

class HumanML3DDataset(Dataset):
    def __init__(
        self, 
        text_dir, 
        negative_text_dir, 
        tokenizer, 
        max_length=77,
        num_other_negatives=2,
        random_seed=42
    ):
        """
        Args:
            text_dir: Directory with positive captions (e.g., <motion_id>.txt).
            negative_text_dir: Directory with negative-swapped captions if exist (<motion_id>.txt).
            tokenizer: CLIP tokenizer (or any Hugging Face tokenizer).
            max_length: Max token length for the text encoder.
            num_other_negatives: how many "other motion" lines to sample per anchor.
            random_seed: fix seed for reproducible sampling (optional).
        """

        self.text_dir = text_dir
        self.negative_text_dir = negative_text_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_other_negatives = num_other_negatives

        random.seed(random_seed)

        # 1) Gather all <motion_id>.txt files in text_dir
        #    Each file corresponds to one motion
        motion_files = sorted(glob.glob(os.path.join(text_dir, "*.txt")))
        self.motion_id_list = []
        self.motion_id_to_path = {}
        for f in motion_files:
            motion_id = os.path.splitext(os.path.basename(f))[0]
            self.motion_id_list.append(motion_id)
            self.motion_id_to_path[motion_id] = f

        # 2) For negative-swapped lines
        #    We'll store a dict motion_id -> path to negative file if exists
        self.motion_id_to_neg_path = {}
        for mid in self.motion_id_list:
            neg_path = os.path.join(negative_text_dir, f"{mid}.txt")
            if os.path.isfile(neg_path):
                self.motion_id_to_neg_path[mid] = neg_path
            else:
                self.motion_id_to_neg_path[mid] = None

        # 3) The "samples" in our dataset: we simply treat each motion as an entry.
        #    We'll produce one anchor line *per motion* per __getitem__ call in a single epoch.
        #    If you want multiple anchors per file in one epoch, you can store them differently
        #    or iterate more times. But let's keep it simple.
        self.samples = self.motion_id_list

        print("[HumanML3DDataset] # Motions:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def _read_lines(self, path):
        """ Read lines from a file, stripping comments after '#' and empty lines. """
        if not os.path.isfile(path):
            return []
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('#', 1)
                caption = parts[0].strip()
                if caption:
                    lines.append(caption)
        return lines

    def __getitem__(self, idx):
        """ Return a single anchor from motion 'mid', plus positives from same file,
            plus negatives from same motion's swapped lines (if any),
            plus random lines from other motions.
        """
        mid = self.samples[idx]

        # Load all lines for the anchor motion from disk
        motion_file = self.motion_id_to_path[mid]
        all_lines = self._read_lines(motion_file)
        if len(all_lines) == 0:
            # If there's no line at all, return dummy
            return {
                'anchor_tokens': None,
                'positives_same_motion': [],
                'negatives_other_motion': [],
                'negatives_same_motion': []
            }

        # Randomly choose one line as anchor
        anchor_text = random.choice(all_lines)
        positives_same_motion = [l for l in all_lines if l != anchor_text]

        # Load negative-swapped lines if exist
        neg_path = self.motion_id_to_neg_path[mid]
        negatives_same_motion = self._read_lines(neg_path) if neg_path else []

        # Sample negative-other lines from other motions
        # We pick e.g. 'num_other_negatives' random lines from random other motions
        # For large datasets, we might just pick 1 random motion id for each negative, or multiple.
        negatives_other_motion = []
        other_motion_ids = [m for m in self.motion_id_list if m != mid]
        if len(other_motion_ids) > 0:
            for _ in range(self.num_other_negatives):
                rand_mid = random.choice(other_motion_ids)
                other_file = self.motion_id_to_path[rand_mid]
                other_lines = self._read_lines(other_file)
                if len(other_lines) > 0:
                    # pick exactly 1 random line from that other motion
                    negatives_other_motion.append(random.choice(other_lines))

        # Tokenize
        anchor_tokens = self.tokenizer(anchor_text, 
                                       truncation=True,
                                       max_length=self.max_length,
                                       return_tensors="pt")

        pos_same_tokens = [
            self.tokenizer(l,
                           truncation=True,
                           max_length=self.max_length,
                           return_tensors="pt")
            for l in positives_same_motion
        ]
        neg_other_tokens = [
            self.tokenizer(l,
                           truncation=True,
                           max_length=self.max_length,
                           return_tensors="pt")
            for l in negatives_other_motion
        ]
        neg_same_tokens = [
            self.tokenizer(l,
                           truncation=True,
                           max_length=self.max_length,
                           return_tensors="pt")
            for l in negatives_same_motion
        ]

        return {
            'anchor_tokens'         : anchor_tokens,
            'positives_same_motion' : pos_same_tokens,
            'negatives_other_motion': neg_other_tokens,
            'negatives_same_motion' : neg_same_tokens
        }
