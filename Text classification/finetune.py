"""
╔══════════════════════════════════════════════════════════════╗
║  Fine-tuning multi-label sur le dataset Artishow (IU X-Ray) ║
║                                                              ║
║  Usage :  %run finetune.py 5    (CheXpert competition)      ║
║           %run finetune.py 14   (NIH ChestX-ray14)          ║
║           %run finetune.py 21   (tous les labels IU X-Ray)  ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, random, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report, hamming_loss
from model import BERTForMLM, Classifier, LabelDataset, pad_collate

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════

FINETUNE_PATH = "/content/drive/MyDrive/Colab Notebooks/dataset/dataset_reports.csv"
TEXT_COL      = "findings"
META_COLS     = {"xml_uid", "indication", "findings", "impression",
                 "comparison", "image_ids", "num_images"}

# ── Jeux de labels ────────────────────────────────────────────────────────
# Mode 5 : CheXpert competition (benchmark SOTA)
# "Effusion" dans IU X-Ray = "Pleural Effusion" au sens CheXpert
LABELS_5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]

# Mode 14 : NIH ChestX-ray14 (sans Normal, implicite = tout à 0)
LABELS_14 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

# Mode 21 : tous les labels IU X-Ray (incluant Normal)

# ── Choix via argument ────────────────────────────────────────────────────
mode = int(sys.argv[1]) if len(sys.argv) > 1 else 21
assert mode in (5, 14, 21), "Usage : %run finetune.py [5|14|21]"

D, H, N, D_FF = 256, 8, 6, 512
FREEZE_ENCODER = False
EPOCHS   = 30
LR       = 2e-4
BATCH    = 32
PATIENCE = 5
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════════
#  DONNÉES
# ══════════════════════════════════════════════════════════════════════════

random.seed(42); torch.manual_seed(42); np.random.seed(42)

df = pd.read_csv(FINETUNE_PATH)

if mode == 5:
    label_cols = [c for c in LABELS_5 if c in df.columns]
elif mode == 14:
    label_cols = [c for c in LABELS_14 if c in df.columns]
else:
    label_cols = [c for c in df.columns if c not in META_COLS]

df = df.dropna(subset=[TEXT_COL])
texts     = df[TEXT_COL].astype(str).tolist()
labels_np = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
labels    = labels_np.tolist()

mode_name = {5: "CheXpert-5", 14: "NIH-14", 21: "IU-XRay-21"}[mode]
print(f"Mode     : {mode_name}")
print(f"Dataset  : {len(texts)} rapports | {len(label_cols)} labels | Device : {DEVICE}")
print(f"Labels   : {label_cols}")

# ── Tokenizer + encodeur pré-entraîné ────────────────────────────────────

tok = Tokenizer.from_file("tokenizer.json")
V = tok.get_vocab_size()

bert = BERTForMLM(V, D, H, N, D_FF)
bert.load_state_dict(torch.load("bert_pretrained.pt", map_location=DEVICE, weights_only=True))
print(f"Encodeur chargé : {sum(p.numel() for p in bert.encoder.parameters()):,} params")

# ══════════════════════════════════════════════════════════════════════════
#  CLASSIFIEUR + POS_WEIGHT
# ══════════════════════════════════════════════════════════════════════════

clf = Classifier(bert.encoder, n_labels=len(label_cols)).to(DEVICE)
if FREEZE_ENCODER:
    for p in clf.encoder.parameters(): p.requires_grad = False
print(f"Params entraînables : {sum(p.numel() for p in clf.parameters() if p.requires_grad):,}")

full_ds = LabelDataset(texts, labels, tok)
n_val = max(1, len(full_ds) // 5)
train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
print(f"Train : {len(train_ds)} | Val : {n_val}")

# pos_weight : neg/pos clampé à 5
y_all = torch.tensor(labels_np)
pos = y_all.sum(0); neg = len(y_all) - pos
pos_weight = (neg / pos.clamp(min=1)).clamp(max=5).to(DEVICE)
print(f"pos_weight : {pos_weight.cpu().numpy().round(2)}")

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
train_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=pad_collate)
val_loader   = DataLoader(val_ds, BATCH, collate_fn=pad_collate)

# ══════════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════

opt   = torch.optim.AdamW(filter(lambda p: p.requires_grad, clf.parameters()),
                          lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
best_val, wait = float("inf"), 0
history = {"train": [], "val": []}

for ep in range(1, EPOCHS + 1):
    clf.train(); tl = 0
    for ids, lab in train_loader:
        ids, lab = ids.to(DEVICE), lab.to(DEVICE)
        loss = loss_fn(clf(ids), lab)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(clf.parameters(), 1.0); opt.step()
        tl += loss.item()
    sched.step()

    clf.eval(); vl = 0
    with torch.no_grad():
        for ids, lab in val_loader:
            ids, lab = ids.to(DEVICE), lab.to(DEVICE)
            vl += loss_fn(clf(ids), lab).item()

    tl /= len(train_loader); vl /= max(len(val_loader), 1)
    history["train"].append(tl); history["val"].append(vl)
    print(f"  Epoch {ep:2d}/{EPOCHS} | train={tl:.4f} | val={vl:.4f}")

    if vl < best_val:
        best_val, wait = vl, 0
        torch.save(clf.state_dict(), f"classifier_{mode}.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"  Early stopping (epoch {ep})"); break

# ══════════════════════════════════════════════════════════════════════════
#  ÉVALUATION
# ══════════════════════════════════════════════════════════════════════════

clf.load_state_dict(torch.load(f"classifier_{mode}.pt", weights_only=True))
clf.eval()

all_probs, all_labels = [], []
with torch.no_grad():
    for ids, lab in val_loader:
        ids, lab = ids.to(DEVICE), lab.to(DEVICE)
        all_probs.append(clf(ids).sigmoid().cpu().numpy())
        all_labels.append(lab.cpu().numpy())

probs  = np.vstack(all_probs)
labels_arr = np.vstack(all_labels)

# Seuil optimal
best_t = max(np.arange(0.1, 0.6, 0.05),
             key=lambda t: f1_score(labels_arr, (probs > t).astype(int),
                                    average='macro', zero_division=0))
preds = (probs > best_t).astype(int)

# Métriques
f1_mac = f1_score(labels_arr, preds, average='macro', zero_division=0)
f1_mic = f1_score(labels_arr, preds, average='micro', zero_division=0)
f1_sam = f1_score(labels_arr, preds, average='samples', zero_division=0)
h_loss = hamming_loss(labels_arr, preds)
valid  = [i for i in range(labels_arr.shape[1]) if len(np.unique(labels_arr[:, i])) > 1]
auc_mac = roc_auc_score(labels_arr[:, valid], probs[:, valid], average='macro')
auc_mic = roc_auc_score(labels_arr[:, valid], probs[:, valid], average='micro')
auc_per = roc_auc_score(labels_arr[:, valid], probs[:, valid], average=None)
valid_names = [label_cols[i] for i in valid]

print(f"\n{'═'*50}")
print(f"  {mode_name} — seuil={best_t:.2f}")
print(f"{'═'*50}")
print(f"  F1 macro   : {f1_mac:.4f}")
print(f"  F1 micro   : {f1_mic:.4f}")
print(f"  F1 samples : {f1_sam:.4f}")
print(f"  AUC macro  : {auc_mac:.4f}")
print(f"  AUC micro  : {auc_mic:.4f}")
print(f"  Hamming    : {h_loss:.4f}")

print(f"\n── AUC par label ──")
for name, auc in sorted(zip(valid_names, auc_per), key=lambda x: x[1], reverse=True):
    bar = '█' * int(auc * 20)
    print(f"  {name:25s} {auc:.4f}  {bar}")

print(f"\n── Rapport sklearn ──")
# Afficher Effusion = Pleural Effusion en mode 5
display_names = label_cols.copy()
if mode == 5:
    display_names = [n if n != "Effusion" else "Pleural Effusion" for n in display_names]
print(classification_report(labels_arr, preds, target_names=display_names, zero_division=0))

# ══════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history["train"], 'o-', label='Train', color='tab:red')
axes[0].plot(history["val"], 's-', label='Val', color='tab:orange')
axes[0].set(title='Train vs Val Loss', xlabel='Epoch', ylabel='Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

names_g = ['F1\nmacro', 'F1\nmicro', 'F1\nsamples', 'AUC\nmacro', 'AUC\nmicro']
vals_g  = [f1_mac, f1_mic, f1_sam, auc_mac, auc_mic]
bars = axes[1].bar(names_g, vals_g, color=['steelblue']*3 + ['coral']*2)
axes[1].set_ylim(0, 1); axes[1].set_title('Métriques globales')
for bar, val in zip(bars, vals_g):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=9)
axes[1].grid(axis='y', alpha=0.3)

sp = sorted(zip(valid_names, auc_per), key=lambda x: x[1])
ln, av = zip(*sp)
colors = ['coral' if a < 0.7 else 'steelblue' for a in av]
axes[2].barh(ln, av, color=colors)
axes[2].axvline(x=0.7, color='gray', linestyle='--', linewidth=0.8)
axes[2].set_xlim(0, 1); axes[2].set_title('AUC par label'); axes[2].set_xlabel('AUC')
for i, (_, val) in enumerate(sp):
    axes[2].text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)

fig.suptitle(f"Fine-tuning {mode_name} — IU X-Ray", fontsize=13, fontweight='bold')
fig.tight_layout(); fig.savefig(f"ft_{mode}.png", dpi=150); plt.show()

print(f"\n✓ Sauvegardé : classifier_{mode}.pt, ft_{mode}.png")
