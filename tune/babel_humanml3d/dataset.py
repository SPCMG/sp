import os
import random
import json
from torch.utils.data import Dataset

class HumanML3DDataset(Dataset):
    def __init__(
        self,
        text_dir,
        negative_text_dir,
        motion_babel_json,
        babel_caption_json,
        split_file,         
        tokenizer,
        max_text_length=77,
        num_other_negatives=3,
        num_other_positives=3,
        random_seed=42,
    ):
        super().__init__()
        random.seed(random_seed)

        self.text_dir = text_dir
        self.negative_text_dir = negative_text_dir
        self.motion_babel_json = motion_babel_json
        self.babel_caption_json = babel_caption_json    
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.num_other_negatives = num_other_negatives
        self.num_other_positives = num_other_positives

        # --- Load motion->BABEL-labels and label->captions ---
        with open(self.motion_babel_json, 'r') as f:
            self.motion_babel = json.load(f)  # e.g. { "000001": ["walk", "run"], ... }
        with open(self.babel_caption_json, 'r') as f:
            self.babel_caption = json.load(f) # e.g. { "walk": ["a person walks", ...], "run": [...] }

        # Collect all possible BABEL labels
        self.all_babel_labels = sorted(list(self.babel_caption.keys()))

        # Read the motion IDs from the split_file
        with open(split_file, 'r') as f:
            motion_ids = [line.strip() for line in f if line.strip()]

        self.motion_id_list = []
        self.motion_id_to_path = {}
        self.motion_id_to_neg_path = {}

        # Build internal maps for text paths
        for mid in motion_ids:
            text_path = os.path.join(text_dir, f"{mid}.txt")
            if not os.path.isfile(text_path):
                continue
            self.motion_id_list.append(mid)
            self.motion_id_to_path[mid] = text_path

            neg_path = os.path.join(negative_text_dir, f"{mid}.txt")
            self.motion_id_to_neg_path[mid] = neg_path if os.path.isfile(neg_path) else None

        self.samples = self.motion_id_list
        print(f"[HumanML3DDataset] Loaded {len(self.samples)} motions from split: {split_file}")

    def __len__(self):
        return len(self.samples)

    def _read_lines(self, path):
        """ Read lines from a file, skipping comments & blank lines. """
        if not os.path.isfile(path):
            return []
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Remove any trailing comment after '#'
                line = line.split('#', 1)[0].strip()
                if line:
                    lines.append(line)
        return lines

    def _tokenize_list(self, captions):
        """
        Given a list of strings, tokenize each and return
        a list of dicts with 'input_ids' and 'attention_mask'.
        """
        out = []
        for text in captions:
            tokens = self.tokenizer(text, context_length=self.max_text_length)
            mask = (tokens != 0).long()
            out.append({'input_ids': tokens, 'attention_mask': mask})
        return out

    def _build_negatives_other_motion(self, mid, k=3):
        """
        - Get BABEL labels not belonging to 'mid' 
        - Randomly pick up to k
        - Convert each label to a simple sentence: "a person <label>."
        """
        motion_labels = self.motion_babel.get(mid, [])
        neg_candidate_labels = [lbl for lbl in self.all_babel_labels if lbl not in motion_labels]
        random.shuffle(neg_candidate_labels)
        chosen_labels = neg_candidate_labels[:k]   # up to 3
        # Hard-code them into sentences
        return [f"a person {label}." for label in chosen_labels]

    def _build_positives_other_motion(self, mid, k=3):
        """
        - Get BABEL labels belonging to 'mid' 
        - Randomly pick up to k
        - Convert each label to a simple sentence: "a person <label>."
        """
        motion_labels = self.motion_babel.get(mid, [])
        if not motion_labels:
            return []
        random.shuffle(motion_labels)
        chosen_labels = motion_labels[:k]
        # Hard-code them into sentences
        return [f"a person {label}." for label in chosen_labels]
        
    # def _build_positives_other_motion(self, mid, k=3, exclude_texts=None):
    #     """
    #     - Get BABEL labels belonging to 'mid'
    #     - Randomly pick up to k
    #     - For each label, pick a random caption from self.babel_caption[label],
    #       excluding those in 'exclude_texts'.
    #     """
    #     if exclude_texts is None:
    #         exclude_texts = []
    #     motion_labels = self.motion_babel.get(mid, [])
    #     random.shuffle(motion_labels)
    #     chosen_labels = motion_labels[:k]  # up to 3
    #     results = []
    #     for label in chosen_labels:
    #         cands = self.babel_caption.get(label, [])
    #         # Exclude lines that are identical to the motion’s rephrases in 'exclude_texts'
    #         cands = [cap for cap in cands if cap not in exclude_texts]
    #         if len(cands) == 0:
    #             continue
    #         results.append(random.choice(cands))
    #     return results

    def __getitem__(self, idx):
        mid = self.samples[idx]

        # 1) Positives from the same motion
        positives_same_motion = self._read_lines(self.motion_id_to_path[mid])
        if not positives_same_motion:
            # If there's no line at all, return dummy
            return {
                'positives_same_motion': [],
                'negatives_same_motion': [],
                'positives_other_motion': [],
                'negatives_other_motion': []
            }

        # 2) Negatives from the same motion (swapped actions)
        neg_same_path = self.motion_id_to_neg_path[mid]
        negatives_same_motion = self._read_lines(neg_same_path) if neg_same_path else []

        # 3) Negatives from other motion 
        negatives_other_motion = self._build_negatives_other_motion(mid, self.num_other_negatives)

        # 4) Positives from other motion 
        positives_other_motion = self._build_positives_other_motion(mid, self.num_other_positives)

        # --- Tokenize everything ---
        pos_same_tokens = self._tokenize_list(positives_same_motion)
        neg_same_tokens = self._tokenize_list(negatives_same_motion)
        pos_other_tokens = self._tokenize_list(positives_other_motion)
        neg_other_tokens = self._tokenize_list(negatives_other_motion)

        return {
            'positives_same_motion': pos_same_tokens,
            'negatives_same_motion': neg_same_tokens,
            'positives_other_motion': pos_other_tokens,
            'negatives_other_motion': neg_other_tokens,
        }


def collate_fn(batch):
    """
    Collects the data from multiple samples and forms a batch of:
      (anchors, pos_sames, neg_sames, pos_others, neg_others)
    where each element is a tuple of:
      ([list_of_input_ids], [list_of_attention_masks]) 
    for each sample in the batch.
    """
    pos_sames = []
    neg_sames = []
    pos_others = []
    neg_others = []

    for item in batch:
        # Positives from same motion
        p_tokens = item['positives_same_motion']
        p_ids = [t['input_ids'].squeeze(0) for t in p_tokens]
        p_masks = [t['attention_mask'].squeeze(0) for t in p_tokens]
        pos_sames.append((p_ids, p_masks))

        # Negatives from same motion
        ns_tokens = item['negatives_same_motion']
        ns_ids = [t['input_ids'].squeeze(0) for t in ns_tokens]
        ns_masks = [t['attention_mask'].squeeze(0) for t in ns_tokens]
        neg_sames.append((ns_ids, ns_masks))

        # Positives from other motion
        po_tokens = item['positives_other_motion']
        po_ids = [t['input_ids'].squeeze(0) for t in po_tokens]
        po_masks = [t['attention_mask'].squeeze(0) for t in po_tokens]
        pos_others.append((po_ids, po_masks))

        # Negatives from other motion
        no_tokens = item['negatives_other_motion']
        no_ids = [t['input_ids'].squeeze(0) for t in no_tokens]
        no_masks = [t['attention_mask'].squeeze(0) for t in no_tokens]
        neg_others.append((no_ids, no_masks))

    return pos_sames, neg_sames, pos_others, neg_others
