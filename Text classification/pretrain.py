"""
╔══════════════════════════════════════════════════════════════╗
║  ÉTAPE 1 — Pré-entraînement MLM sur MIMIC-CXR (HuggingFace)║
║                                                              ║
║  Produit : tokenizer.json, bert_pretrained.pt, mlm_curves.png║
╚══════════════════════════════════════════════════════════════╝
"""

import random, torch, torch.nn as nn, torch.nn.functional as F, pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import BERTForMLM, build_tokenizer, MLMDataset, pad_collate

# ── Config ────────────────────────────────────────────────────────────────

PRETRAIN_PATH = "/content/drive/MyDrive/Colab Notebooks/dataset/mimic_cxr_reports.csv"
TEXT_COL      = "reports"

D, H, N, D_FF = 256, 8, 6, 512
EPOCHS = 20
LR     = 3e-4
BATCH  = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Données ───────────────────────────────────────────────────────────────

random.seed(42); torch.manual_seed(42)

df = pd.read_csv(PRETRAIN_PATH)
texts = df[TEXT_COL].dropna().astype(str).tolist()
print(f"MIMIC-CXR : {len(texts)} rapports | Device : {DEVICE}")

# ── Tokenizer ─────────────────────────────────────────────────────────────

tok = build_tokenizer(texts)
V = tok.get_vocab_size()
print(f"Vocab     : {V} tokens")
print(f"Exemple   : {tok.encode(texts[0]).tokens[:12]}...")
tok.save("tokenizer.json")

# ── Modèle ────────────────────────────────────────────────────────────────

model = BERTForMLM(V, D, H, N, D_FF).to(DEVICE)
print(f"Params    : {sum(p.numel() for p in model.parameters()):,}")

# ── Entraînement ──────────────────────────────────────────────────────────

dataset = MLMDataset(texts, tok)
loader  = DataLoader(dataset, BATCH, shuffle=True, collate_fn=pad_collate)
opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

history = {"loss": [], "acc": [], "lr": []}

model.train()
for ep in range(1, EPOCHS + 1):
    total_loss, correct, masked = 0, 0, 0
    for ids, lab in loader:
        ids, lab = ids.to(DEVICE), lab.to(DEVICE)
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               lab.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        m = lab != -100
        if m.any():
            correct += (logits.argmax(-1)[m] == lab[m]).sum().item()
            masked  += m.sum().item()
        total_loss += loss.item()

    sched.step()
    avg_loss = total_loss / len(loader)
    acc = correct / max(masked, 1)
    history["loss"].append(avg_loss)
    history["acc"].append(acc)
    history["lr"].append(sched.get_last_lr()[0])
    print(f"  Epoch {ep:2d}/{EPOCHS} | loss={avg_loss:.4f} | acc={acc:.1%} | lr={history['lr'][-1]:.1e}")

# ── Sauvegarde ────────────────────────────────────────────────────────────

torch.save(model.state_dict(), "bert_pretrained.pt")

# ── Courbes ───────────────────────────────────────────────────────────────

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

ax1.plot(history["loss"], "o-", color="tab:red")
ax1.set(title="MLM Loss", xlabel="Epoch", ylabel="Loss")
ax1.grid(True, alpha=0.3)

ax2.plot([a * 100 for a in history["acc"]], "o-", color="tab:blue")
ax2.set(title="MLM Accuracy", xlabel="Epoch", ylabel="Acc (%)")
ax2.grid(True, alpha=0.3)

ax3.plot(history["lr"], "o-", color="tab:green")
ax3.set(title="Learning Rate", xlabel="Epoch", ylabel="LR")
ax3.grid(True, alpha=0.3)

fig.suptitle("Pré-entraînement MLM — MIMIC-CXR", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("mlm_curves.png", dpi=150)
plt.show()

print("\n✓ Sauvegardé : tokenizer.json, bert_pretrained.pt, mlm_curves.png")
