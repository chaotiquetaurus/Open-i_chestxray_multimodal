"""datasets.py — MLMDataset, LabelDataset et pad_collate."""

import random, torch, torch.nn.functional as F
from torch.utils.data import Dataset
from .tokenizer import SP


class MLMDataset(Dataset):
    def __init__(self, texts, tok, mask_prob=0.15):
        self.enc = tok.encode_batch(texts)
        self.mid = tok.token_to_id(SP["mask"])
        self.skip = {tok.token_to_id(v) for v in SP.values()}
        self.V, self.p = tok.get_vocab_size(), mask_prob
    def __len__(self): return len(self.enc)
    def __getitem__(self, i):
        ids = torch.tensor(self.enc[i].ids, dtype=torch.long); lab = ids.clone()
        for j in range(len(ids)):
            if ids[j].item() not in self.skip and random.random() < self.p:
                r = random.random()
                if r < 0.8:   ids[j] = self.mid
                elif r < 0.9: ids[j] = random.randint(0, self.V-1)
            else: lab[j] = -100
        return ids, lab


class LabelDataset(Dataset):
    def __init__(self, texts, labels, tok):
        self.enc = tok.encode_batch(texts)
        self.lab = torch.tensor(labels, dtype=torch.float)
    def __len__(self): return len(self.enc)
    def __getitem__(self, i):
        return torch.tensor(self.enc[i].ids, dtype=torch.long), self.lab[i]


def pad_collate(batch):
    seqs, labs = zip(*batch)
    ml = max(len(s) for s in seqs)
    seqs = torch.stack([F.pad(s, (0, ml-len(s))) for s in seqs])
    labs = torch.stack(list(labs)) if labs[0].dim() > 0 and labs[0].shape[0] > 1 \
        else torch.stack([F.pad(l, (0, ml-len(l)), value=-100) for l in labs])
    return seqs, labs
