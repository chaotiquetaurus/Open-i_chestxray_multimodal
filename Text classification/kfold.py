"""
╔══════════════════════════════════════════════════════════════╗
║  kfold.py — 5-fold CV avec PyTorch Lightning                ║
║                                                              ║
║  Usage :  %run kfold.py 5                                   ║
║           %run kfold.py 14                                  ║
║           %run kfold.py 21                                  ║
║                                                              ║
║  Requiert : pip install pytorch-lightning                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, copy, torch, torch.nn as nn
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from tokenizers import Tokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from model import BERTForMLM, Classifier, LabelDataset, pad_collate

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════

FINETUNE_PATH = "/content/drive/MyDrive/Colab Notebooks/dataset/dataset_reports.csv"
TEXT_COLS  = ["indication", "findings"]
META_COLS  = {"xml_uid", "indication", "findings", "impression",
              "comparison", "image_ids", "num_images"}

LABELS_5  = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
LABELS_14 = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
             "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
             "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

mode = int(sys.argv[1]) if len(sys.argv) > 1 else 21
assert mode in (5, 14, 21)

D, H, N, D_FF = 256, 8, 6, 512
MAX_EPOCHS, LR, BATCH, K = 30, 2e-4, 32, 5

# ══════════════════════════════════════════════════════════════════════════
#  LIGHTNING MODULE
# ══════════════════════════════════════════════════════════════════════════

class LitClassifier(pl.LightningModule):
    def __init__(self, encoder, n_labels, pos_weight):
        super().__init__()
        self.clf = Classifier(encoder, n_labels)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.val_probs = []
        self.val_labels = []

    def forward(self, x):
        return self.clf(x)

    def training_step(self, batch, _):
        ids, lab = batch
        loss = self.loss_fn(self(ids), lab)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        ids, lab = batch
        logits = self(ids)
        loss = self.loss_fn(logits, lab)
        self.log("val_loss", loss, prog_bar=True)
        self.val_probs.append(logits.sigmoid().cpu())
        self.val_labels.append(lab.cpu())

    def on_validation_epoch_end(self):
        if self.val_probs:
            probs = torch.cat(self.val_probs).numpy()
            labs  = torch.cat(self.val_labels).numpy()
            valid = [i for i in range(labs.shape[1]) if len(np.unique(labs[:, i])) > 1]
            if valid:
                auc = roc_auc_score(labs[:, valid], probs[:, valid], average='macro')
                self.log("val_auc", auc, prog_bar=True)
        self.val_probs.clear()
        self.val_labels.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=LR, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
        return [opt], [sched]

# ══════════════════════════════════════════════════════════════════════════
#  DONNÉES
# ══════════════════════════════════════════════════════════════════════════

np.random.seed(42); torch.manual_seed(42)

df = pd.read_csv(FINETUNE_PATH)
if mode == 5:    label_cols = [c for c in LABELS_5  if c in df.columns]
elif mode == 14: label_cols = [c for c in LABELS_14 if c in df.columns]
else:            label_cols = [c for c in df.columns if c not in META_COLS]

df = df.dropna(subset=TEXT_COLS)
texts     = df[TEXT_COLS].fillna("").agg(" ".join, axis=1).str.strip().tolist()
labels_np = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

mode_name = {5: "CheXpert-5", 14: "NIH-14", 21: "IU-XRay-21"}[mode]
print(f"{mode_name} | {len(texts)} rapports | {len(label_cols)} labels | {K}-fold CV")

tok = Tokenizer.from_file("tokenizer.json")
V = tok.get_vocab_size()
bert = BERTForMLM(V, D, H, N, D_FF)
bert.load_state_dict(torch.load("bert_pretrained.pt", map_location="cpu", weights_only=True))

# pos_weight
y_all = torch.tensor(labels_np)
y_patho = y_all[y_all.sum(1) > 0]
pos = y_patho.sum(0); neg = len(y_patho) - pos
pos_weight = (neg / pos.clamp(min=1)).clamp(max=20)

full_ds = LabelDataset(texts, labels_np.tolist(), tok)

# ══════════════════════════════════════════════════════════════════════════
#  K-FOLD
# ══════════════════════════════════════════════════════════════════════════

kfold = KFold(n_splits=K, shuffle=True, random_state=42)
all_probs, all_labels = [], []

for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_ds))), 1):
    print(f"\n{'─'*40}  Fold {fold}/{K}  {'─'*40}")

    train_loader = DataLoader(Subset(full_ds, train_idx), BATCH, shuffle=True,  collate_fn=pad_collate)
    val_loader   = DataLoader(Subset(full_ds, val_idx),   BATCH, collate_fn=pad_collate)

    # Fresh model à chaque fold
    encoder = copy.deepcopy(bert.encoder)
    lit = LitClassifier(encoder, len(label_cols), pos_weight)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        callbacks=[pl.callbacks.EarlyStopping("val_loss", patience=5, mode="min")],
        enable_checkpointing=False,
        enable_model_summary=False if fold > 1 else True,
        logger=False,
    )
    trainer.fit(lit, train_loader, val_loader)

    # Collecter les prédictions du fold
    lit.eval()
    fold_probs, fold_labels = [], []
    with torch.no_grad():
        for ids, lab in val_loader:
            fold_probs.append(lit(ids.to(lit.device)).sigmoid().cpu().numpy())
            fold_labels.append(lab.numpy())
    all_probs.append(np.vstack(fold_probs))
    all_labels.append(np.vstack(fold_labels))

# ══════════════════════════════════════════════════════════════════════════
#  MÉTRIQUES AGRÉGÉES
# ══════════════════════════════════════════════════════════════════════════

probs      = np.vstack(all_probs)
labels_arr = np.vstack(all_labels)

best_t = max(np.arange(0.1, 0.6, 0.05),
             key=lambda t: f1_score(labels_arr, (probs > t).astype(int),
                                    average='macro', zero_division=0))
preds = (probs > best_t).astype(int)

f1_mac = f1_score(labels_arr, preds, average='macro', zero_division=0)
f1_mic = f1_score(labels_arr, preds, average='micro', zero_division=0)
valid  = [i for i in range(labels_arr.shape[1]) if len(np.unique(labels_arr[:, i])) > 1]
auc_mac = roc_auc_score(labels_arr[:, valid], probs[:, valid], average='macro')
auc_per = roc_auc_score(labels_arr[:, valid], probs[:, valid], average=None)
valid_names = [label_cols[i] for i in valid]

print(f"\n{'═'*55}")
print(f"  {mode_name} — {K}-fold CV — seuil={best_t:.2f}")
print(f"{'═'*55}")
print(f"  F1 macro  : {f1_mac:.4f}")
print(f"  F1 micro  : {f1_mic:.4f}")
print(f"  AUC macro : {auc_mac:.4f}")

print(f"\n── AUC par label ──")
for name, auc in sorted(zip(valid_names, auc_per), key=lambda x: x[1], reverse=True):
    print(f"  {name:25s} {auc:.4f}  {'█' * int(auc * 20)}")

display_names = label_cols.copy()
if mode == 5:
    display_names = [n if n != "Effusion" else "Pleural Effusion" for n in display_names]
print(f"\n{classification_report(labels_arr, preds, target_names=display_names, zero_division=0)}")

# ── Plot ──────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

names_g = ['F1 macro', 'F1 micro', 'AUC macro']
vals_g  = [f1_mac, f1_mic, auc_mac]
bars = ax1.bar(names_g, vals_g, color=['steelblue', 'steelblue', 'coral'])
ax1.set_ylim(0, 1); ax1.set_title(f'{mode_name} — Métriques globales')
for bar, val in zip(bars, vals_g):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

sp = sorted(zip(valid_names, auc_per), key=lambda x: x[1])
ln, av = zip(*sp)
ax2.barh(ln, av, color=['coral' if a < 0.7 else 'steelblue' for a in av])
ax2.axvline(x=0.7, color='gray', linestyle='--', linewidth=0.8)
ax2.set_xlim(0, 1); ax2.set_title('AUC par label'); ax2.set_xlabel('AUC')
for i, (_, val) in enumerate(sp):
    ax2.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)

fig.suptitle(f"{mode_name} — {K}-fold CV", fontsize=13, fontweight='bold')
fig.tight_layout(); fig.savefig(f"kfold_{mode}.png", dpi=150); plt.show()
print(f"\n✓ Sauvegardé : kfold_{mode}.png")
