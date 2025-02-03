import os
import random
import json
from torch.utils.data import Dataset

class WordGroupDataset(Dataset):
    def __init__(
        self,
        json_path,
        tokenizer,
        max_text_length=77,
        num_positives=2,
        num_negatives=3,
        random_seed=42,
    ):
        """
        Args:
            json_path: Path to your JSON file (the one with the keys "jump", "jump & run", etc.).
            tokenizer: Your tokenizer instance (SimpleTokenizer).
            max_text_length: Max sequence length for tokenization.
            num_positives: How many additional positives to sample from the same label.
            num_negatives: How many negatives to sample from other labels.
        """
        super().__init__()
        random.seed(random_seed)
        
        # 1) Load the entire JSON of label -> list of sentences
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # self.data is a dict like:
        # {
        #   "jump": [ "a person jumps high...", ... ],
        #   "run": [ "a person runs quickly...", ... ],
        #   ...
        # }
        
        # 2) Convert them into a list of (label, sentence)
        #    so that each index corresponds to one "sample".
        self.labels = sorted(list(self.data.keys()))  # e.g. ["jump", "jump & run", "run", ...]
        
        # Build a list of (label_idx, sentence)
        self.samples = []
        for label_idx, label_name in enumerate(self.labels):
            sentences = self.data[label_name]
            for s in sentences:
                self.samples.append((label_idx, s))
        
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.num_positives = num_positives
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.samples)

    def _tokenize_list(self, captions):
        """
        Given a list of strings, tokenize each and return
        a list of dicts with {'input_ids': ..., 'attention_mask': ...}.
        """
        out = []
        for text in captions:
            tokens = self.tokenizer(text, context_length=self.max_text_length)
            mask = (tokens != 0).long()
            out.append({'input_ids': tokens, 'attention_mask': mask})
        return out

    def __getitem__(self, idx):
        """
        Return:
          {
            'anchor': [dict_with_input_ids_and_mask],
            'positives': [list_of_token_dicts],
            'negatives': [list_of_token_dicts]
          }
        """
        label_idx, anchor_sentence = self.samples[idx]
        label_name = self.labels[label_idx]
        
        # ----- Build anchor -----
        anchor_token = self._tokenize_list([anchor_sentence])  # single-sentence list
        
        # ----- Build positives (from same label) -----
        all_sents_this_label = self.data[label_name]
        # Exclude the anchor from sampling
        positive_candidates = [s for s in all_sents_this_label if s != anchor_sentence]
        random.shuffle(positive_candidates)
        selected_pos = positive_candidates[: self.num_positives]
        pos_tokens = self._tokenize_list(selected_pos)
        
        # ----- Build negatives (from other labels) -----
        other_label_indices = [i for i in range(len(self.labels)) if i != label_idx]
        random.shuffle(other_label_indices)
        
        negative_sents = []
        # We'll keep sampling from different labels until we have enough negative sentences
        for neg_label_idx in other_label_indices:
            if len(negative_sents) >= self.num_negatives:
                break
            neg_label_name = self.labels[neg_label_idx]
            # randomly choose one sentence from that label
            candidate = random.choice(self.data[neg_label_name])
            negative_sents.append(candidate)
        
        neg_tokens = self._tokenize_list(negative_sents)

        return {
            'anchor': anchor_token,
            'positives': pos_tokens,
            'negatives': neg_tokens
        }


def collate_fn(batch):
    """
    Collate into:
        anchors, positives, negatives
    Each is a list of length [batch_size], 
    where each element is a tuple (list_of_input_ids, list_of_attention_masks)
    for that sample.
    """
    anchors_batch = []
    positives_batch = []
    negatives_batch = []

    for item in batch:
        # 1) Anchor: always exactly 1 sentence
        a_tok = item['anchor'][0]  # single dict
        anchors_batch.append((
            a_tok['input_ids'].squeeze(0), 
            a_tok['attention_mask'].squeeze(0)
        ))
        
        # 2) Positives: could be multiple
        p_ids = [t['input_ids'].squeeze(0) for t in item['positives']]
        p_masks = [t['attention_mask'].squeeze(0) for t in item['positives']]
        positives_batch.append((p_ids, p_masks))
        
        # 3) Negatives
        n_ids = [t['input_ids'].squeeze(0) for t in item['negatives']]
        n_masks = [t['attention_mask'].squeeze(0) for t in item['negatives']]
        negatives_batch.append((n_ids, n_masks))

    # We'll return (anchors, positives, negatives)
    return anchors_batch, positives_batch, negatives_batch
