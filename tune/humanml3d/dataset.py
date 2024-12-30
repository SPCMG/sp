import os
import random
from torch.utils.data import Dataset

class HumanML3DDataset(Dataset):
    def __init__(
        self,
        text_dir,
        negative_text_dir,
        split_file,         
        tokenizer,
        max_length=77,
        num_other_negatives=3,
        random_seed=42
    ):
        super().__init__()
        random.seed(random_seed)

        self.text_dir = text_dir
        self.negative_text_dir = negative_text_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_other_negatives = num_other_negatives

        # Read the motion IDs from the split_file
        with open(split_file, 'r') as f:
            motion_ids = [line.strip() for line in f if line.strip()]

        # For each motion_id, store the path to the "positives" file and possibly to the negatives
        self.motion_id_list = []
        self.motion_id_to_path = {}
        self.motion_id_to_neg_path = {}

        for mid in motion_ids:
            text_path = os.path.join(text_dir, f"{mid}.txt")
            if not os.path.isfile(text_path):
                # If there's no file for this motion, skip
                continue
            self.motion_id_list.append(mid)
            self.motion_id_to_path[mid] = text_path

            neg_path = os.path.join(negative_text_dir, f"{mid}.txt")
            if os.path.isfile(neg_path):
                self.motion_id_to_neg_path[mid] = neg_path
            else:
                self.motion_id_to_neg_path[mid] = None

        self.samples = self.motion_id_list
        print(f"[HumanML3DDataset] Loaded {len(self.samples)} motions from split: {split_file}")

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
        negatives_other_motion = []
        other_motion_ids = [m for m in self.motion_id_list if m != mid]
        if len(other_motion_ids) > 0:
            for _ in range(self.num_other_negatives):
                rand_mid = random.choice(other_motion_ids)
                other_file = self.motion_id_to_path[rand_mid]
                other_lines = self._read_lines(other_file)
                if len(other_lines) > 0:
                    negatives_other_motion.append(random.choice(other_lines))

        # Tokenize with SimpleTokenizer format
        anchor_tokens = {
            'input_ids': self.tokenizer(anchor_text, context_length=self.max_length),
            'attention_mask': (self.tokenizer(anchor_text, context_length=self.max_length) != 0).long()
        }

        pos_same_tokens = [
            {
                'input_ids': self.tokenizer(l, context_length=self.max_length),
                'attention_mask': (self.tokenizer(l, context_length=self.max_length) != 0).long()
            }
            for l in positives_same_motion
        ]
        
        neg_other_tokens = [
            {
                'input_ids': self.tokenizer(l, context_length=self.max_length),
                'attention_mask': (self.tokenizer(l, context_length=self.max_length) != 0).long()
            }
            for l in negatives_other_motion
        ]
        
        neg_same_tokens = [
            {
                'input_ids': self.tokenizer(l, context_length=self.max_length),
                'attention_mask': (self.tokenizer(l, context_length=self.max_length) != 0).long()
            }
            for l in negatives_same_motion
        ]

        return {
            'anchor_tokens': anchor_tokens,
            'positives_same_motion': pos_same_tokens,
            'negatives_other_motion': neg_other_tokens,
            'negatives_same_motion': neg_same_tokens
        }

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