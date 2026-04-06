"""model.py — Encoder BERT, tokenizer, datasets. Importé par pretrain.py et finetune.py."""

import random, torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from torch.utils.data import Dataset

SP = {"pad":"[PAD]","unk":"[UNK]","cls":"[CLS]","sep":"[SEP]","mask":"[MASK]"}

# ── Modèle ────────────────────────────────────────────────────────────────

class RoPE(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        freqs = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
        angles = torch.outer(torch.arange(max_len).float(), freqs)
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())
    def forward(self, x):
        T = x.size(2); c, s = self.cos[:T], self.sin[:T]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1*c - x2*s, x1*s + x2*c), -1).flatten(-2)

class MHA(nn.Module):
    def __init__(self, d, h, rope=None):
        super().__init__()
        self.h, self.dk, self.rope = h, d//h, rope
        self.W_q, self.W_k, self.W_v, self.W_o = (nn.Linear(d,d) for _ in range(4))
    def forward(self, q, k, v):
        B = q.size(0)
        proj = lambda W, x: W(x).view(B,-1,self.h,self.dk).transpose(1,2)
        Q, K, V = proj(self.W_q,q), proj(self.W_k,k), proj(self.W_v,v)
        if self.rope: Q, K = self.rope(Q), self.rope(K)
        out = F.scaled_dot_product_attention(Q, K, V)
        return self.W_o(out.transpose(1,2).contiguous().view(B,-1,self.h*self.dk))

class Block(nn.Module):
    def __init__(self, d, h, d_ff, rope):
        super().__init__()
        self.attn, self.ffn = MHA(d, h, rope), nn.Sequential(nn.Linear(d,d_ff), nn.SiLU(), nn.Linear(d_ff,d))
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
    def forward(self, x):
        xn = self.n1(x); x = x + self.attn(xn,xn,xn)
        return x + self.ffn(self.n2(x))

class Encoder(nn.Module):
    def __init__(self, V, d=256, h=8, N=6, d_ff=512):
        super().__init__()
        self.d, self.emb = d, nn.Embedding(V, d)
        rope = RoPE(d//h)
        self.layers = nn.ModuleList([Block(d,h,d_ff,rope) for _ in range(N)])
        self.norm = nn.RMSNorm(d)
    def forward(self, src):
        x = self.emb(src)
        for l in self.layers: x = l(x)
        return self.norm(x)

class BERTForMLM(nn.Module):
    def __init__(self, V, d=256, h=8, N=6, d_ff=512):
        super().__init__()
        self.encoder = Encoder(V,d,h,N,d_ff)
        self.head = nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.RMSNorm(d), nn.Linear(d,V))
    def forward(self, ids): return self.head(self.encoder(ids))

class Classifier(nn.Module):
    def __init__(self, encoder, n_labels):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(encoder.d, n_labels))
    def forward(self, ids): return self.head(self.encoder(ids)[:, 0])

# ── Tokenizer ─────────────────────────────────────────────────────────────

def build_tokenizer(texts, vocab_size=8192, max_len=256):
    tok = Tokenizer(models.BPE(unk_token=SP["unk"]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=list(SP.values()), min_frequency=2))
    cid, sid = tok.token_to_id(SP["cls"]), tok.token_to_id(SP["sep"])
    tok.post_processor = processors.TemplateProcessing(
        single=f"{SP['cls']} $A {SP['sep']}", pair=f"{SP['cls']} $A {SP['sep']} $B {SP['sep']}",
        special_tokens=[(SP["cls"], cid), (SP["sep"], sid)])
    tok.decoder = decoders.ByteLevel()
    tok.enable_padding(pad_id=tok.token_to_id(SP["pad"]), pad_token=SP["pad"])
    tok.enable_truncation(max_length=max_len)
    return tok

# ── Datasets ──────────────────────────────────────────────────────────────

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
