from torch.utils.data import Dataset, DataLoader
import numpy as np
from os.path import join as pjoin
import random
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
from data.humanml3d.utils.word_vectorizer import WordVectorizer


class HumanML3D(Dataset):
    def __init__(self, opt, split="train"):
        self.opt = opt
        self.split = split
        self.mean = np.load(pjoin(opt.data_root, "Mean.npy"))
        self.std = np.load(pjoin(opt.data_root, "Std.npy"))
        self.max_text_len = opt.max_text_len
        self.max_motion_length = opt.max_motion_length
        self.w_vectorizer = WordVectorizer(pjoin(opt.data_root, "glove"), "our_vocab")
        
        # Load data
        split_file = pjoin(opt.data_root, f"{split}.txt")
        with open(split_file, "r") as f:
            self.data_files = [line.strip() for line in f.readlines()]
        
        # Preload data
        self.data_dict = self._load_data()

    def _load_data(self):
        data_dict = {}
        for name in tqdm(self.data_files, desc="Loading data"):
            try:
                motion = np.load(pjoin(self.opt.motion_dir, f"{name}.npy"))
                if motion.shape[0] < 40 or motion.shape[0] > 200:
                    continue

                text_file = pjoin(self.opt.text_dir, f"{name}.txt")
                with open(text_file, "r") as f:
                    text_data = [
                        {"caption": line.split("#")[0], "tokens": line.split("#")[1].split(" ")}
                        for line in f.readlines()
                    ]
                data_dict[name] = {"motion": motion, "text": text_data}
            except FileNotFoundError:
                continue
        return data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        key = list(self.data_dict.keys())[idx]
        data = self.data_dict[key]
        motion = data["motion"]
        text_entry = random.choice(data["text"])
        tokens = text_entry["tokens"]
        caption = text_entry["caption"]

        # Process tokens
        if len(tokens) < self.max_text_len:
            tokens = ["sos"] + tokens + ["eos"]
            tokens += ["unk"] * (self.max_text_len - len(tokens))
        else:
            tokens = ["sos"] + tokens[: self.max_text_len] + ["eos"]

        word_embeddings, pos_one_hots = self._process_tokens(tokens)

        # Normalize motion
        motion = (motion - self.mean) / self.std
        motion_length = motion.shape[0]
        if motion_length < self.max_motion_length:
            padding = np.zeros((self.max_motion_length - motion_length, motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)

        return word_embeddings, pos_one_hots, caption, motion

    def _process_tokens(self, tokens):
        pos_one_hots, word_embeddings = [], []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        return np.concatenate(word_embeddings, axis=0), np.concatenate(pos_one_hots, axis=0)


def collate_fn(batch):
    return default_collate(batch)


def get_dataloader(opt, batch_size, split="train"):
    dataset = HumanML3D(opt, split)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
