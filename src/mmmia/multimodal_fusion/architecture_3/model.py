"""model.py — Q-Former label-aligné (famille 3) : fusion image + texte.

Architecture (cf. docs/architectures_fusion_multimodale_v3.md §Famille 3) :

    indication+findings --CXR-BERT (gelé)--> T (B, L, 768) + txt_pad (B, L)
    image PNG           --ViT      (gelé)--> Z (B, 197, 768)
    14 requêtes apprenables --N blocs Q-Former--> H (B, 14, 768)
    H --tête diagonale label-alignée--> 14 logits

Une requête par pathologie (alignement positionnel). La lecture en sortie est
DIAGONALE : logit_j = w_j · H_j + b_j, donc le logit de la pathologie j ne lit
que la requête j (aucun pooling des requêtes). L'ordre des labels est figé en
amont (dataset.py importe LABELS_14 de text_classification — source de vérité).

Chaque bloc enchaîne, dans cet ordre, trois sous-couches résiduelles+normalisées :
  1. Self-attention des requêtes sur [Q; T]  (seules les requêtes sont mises à
     jour — le texte n'est jamais en position de query).
  2. Cross-attention des requêtes vers l'image Z.
  3. Feed-forward sur les requêtes.

Branchement texte (text_feature_mode) :
  - "last" : tous les blocs lisent la dernière couche de CXR-BERT (baseline).
  - "deep" : le bloc l lit la l-ième couche cachée (style BLIP-2 hiérarchique) —
     coarse→fine sur les N dernières couches. Coût nul (BERT calcule déjà tout).
"""

import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel

# Réutilisation de text_classification (convention sys.path du repo) : on injecte
# le paquet avant l'import, car model.py peut être importé avant dataset.py.
_TEXT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "..", "text_classification")
if _TEXT_ROOT not in sys.path:
    sys.path.insert(0, _TEXT_ROOT)

# Constante centralisée du projet (text_classification) — ne pas réinventer.
from models.cxr_bert_classifier import CXR_BERT_NAME  # noqa: E402

VIT_NAME = "codewithdark/vit-chest-xray"


class JointQueryBlock(nn.Module):
    """Un bloc Q-Former : SA([Q;T]) → CA(Q→Z) → FFN, chacune résiduelle + LN."""

    def __init__(self, d=768, n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, q, T, Z, txt_pad=None, return_attn=False):
        B, n = q.size(0), q.size(1)

        # (1) SA des queries sur [Q; T] (asymétrie : seules les queries sont mises à jour)
        ctx_t = torch.cat([q, T], dim=1)                                  # (B, n+L, d)
        kpm = None
        if txt_pad is not None:
            # MultiheadAttention : key_padding_mask True = position ignorée
            # (inverse de la convention HuggingFace) ; les n positions queries
            # sont toujours visibles → zeros (False).
            kpm = torch.cat(
                [torch.zeros(B, n, dtype=torch.bool, device=q.device), txt_pad], dim=1
            )
        h, sa_w = self.self_attn(
            q, ctx_t, ctx_t, key_padding_mask=kpm, need_weights=return_attn
        )
        q = self.ln1(q + h)

        # (2) CA des queries vers l'image
        h, ca_w = self.cross_attn(q, Z, Z, need_weights=return_attn)
        q = self.ln2(q + h)

        # (3) FFN
        q = self.ln3(q + self.ffn(q))

        return q, sa_w, ca_w


class QFormerHead(nn.Module):
    """N blocs JointQueryBlock + tête label-alignée (1 query par pathologie)."""

    def __init__(self, n_query=14, d=768, n_layers=3, n_labels=14,
                 n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        assert n_query == n_labels, "label-aligné : une query par pathologie"
        self.queries = nn.Parameter(torch.randn(1, n_query, d) * 0.02)
        self.blocks = nn.ModuleList(
            [JointQueryBlock(d, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.W = nn.Parameter(torch.randn(n_labels, d) * 0.02)   # (14, d)
        self.b = nn.Parameter(torch.zeros(n_labels))             # (14,)

    def forward(self, T, Z, txt_pad=None, return_attn=False):
        """T : tenseur (B, L, d) commun à tous les blocs, OU liste de N tenseurs
        (un par bloc, pour le branchement profond). Z : (B, Nv, d)."""
        is_per_layer = isinstance(T, (list, tuple))
        if is_per_layer:
            assert len(T) == len(self.blocks), \
                "branchement profond : un tenseur texte par bloc Q-Former"
            B = T[0].size(0)
        else:
            B = T.size(0)

        q = self.queries.expand(B, -1, -1)                    # (B, 14, d)
        sa_all, ca_all = [], []
        for i, blk in enumerate(self.blocks):
            T_i = T[i] if is_per_layer else T
            q, sa_w, ca_w = blk(q, T_i, Z, txt_pad, return_attn)   # q : (B, 14, d)
            sa_all.append(sa_w)
            ca_all.append(ca_w)
        # lecture diagonale : logit_j = w_j · H_j + b_j   (ici q joue le rôle de H)
        logits = (q * self.W.unsqueeze(0)).sum(-1) + self.b   # (B, 14)
        return logits, {"sa": sa_all, "ca": ca_all, "queries": q}


class FusionQFormer(nn.Module):
    """Modèle complet : CXR-BERT (texte) + ViT (image) + Q-Former label-aligné.

    Les deux encodeurs sont gelés par défaut ; le gel est paramétrable et
    indépendant (freeze_text / freeze_image).
    """

    def __init__(
        self,
        n_labels=14,
        d=768,
        n_layers=3,
        n_heads=12,
        d_ff=3072,
        dropout=0.1,
        pad_id=0,
        text_feature_mode="last",       # "last" | "deep"
        freeze_text=True,
        freeze_image=True,
        text_name=CXR_BERT_NAME,
        image_name=VIT_NAME,
    ):
        super().__init__()
        assert text_feature_mode in ("last", "deep")
        self.pad_id = pad_id
        self.text_feature_mode = text_feature_mode

        # ── Encodeurs ────────────────────────────────────────────────────
        self.text_encoder = AutoModel.from_pretrained(text_name, trust_remote_code=True)
        self.image_encoder = ViTModel.from_pretrained(image_name)

        d_text = self.text_encoder.config.hidden_size
        d_image = self.image_encoder.config.hidden_size
        assert d_text == d == d_image, (
            f"dims attendues égales à d={d} (texte={d_text}, image={d_image}). "
            "Avec un encodeur image non-768, ajouter une projection."
        )

        # Mapping bloc Q-Former → couche cachée BERT pour le mode "deep" :
        # les N dernières couches en ordre croissant (coarse→fine). hidden_states
        # a une longueur num_hidden_layers+1 (index 0 = embeddings).
        n_hidden = self.text_encoder.config.num_hidden_layers
        if text_feature_mode == "deep":
            assert n_layers <= n_hidden, (
                f"mode deep : n_layers={n_layers} > couches BERT={n_hidden}"
            )
            self.layer_map = list(range(n_hidden - n_layers + 1, n_hidden + 1))
        else:
            self.layer_map = None

        # ── Q-Former ─────────────────────────────────────────────────────
        self.qformer = QFormerHead(
            n_query=n_labels, d=d, n_layers=n_layers, n_labels=n_labels,
            n_heads=n_heads, d_ff=d_ff, dropout=dropout,
        )

        if freeze_text:
            self.set_text_trainable(False)
        if freeze_image:
            self.set_image_trainable(False)

    # ── Gel / dégel indépendant ──────────────────────────────────────────
    def set_text_trainable(self, flag: bool):
        for p in self.text_encoder.parameters():
            p.requires_grad = flag

    def set_image_trainable(self, flag: bool):
        for p in self.image_encoder.parameters():
            p.requires_grad = flag

    def freeze_encoders(self):
        self.set_text_trainable(False)
        self.set_image_trainable(False)

    def unfreeze_encoders(self):
        self.set_text_trainable(True)
        self.set_image_trainable(True)

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, ids, pixel_values, return_attn=False):
        mask = (ids != self.pad_id).long()                       # 1 = vrai token
        txt_pad = (ids == self.pad_id)                           # True = ignoré (MHA)

        need_hidden = self.text_feature_mode == "deep"
        out = self.text_encoder(
            input_ids=ids, attention_mask=mask, output_hidden_states=need_hidden
        )
        if need_hidden:
            T = [out.hidden_states[i] for i in self.layer_map]   # liste de (B,L,d)
        else:
            T = out.last_hidden_state                            # (B, L, d)

        Z = self.image_encoder(pixel_values=pixel_values).last_hidden_state  # (B,197,d)

        logits, aux = self.qformer(T, Z, txt_pad, return_attn=return_attn)
        if return_attn:
            return logits, aux
        return logits


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss — Ben-Baruch et al. (2021), arXiv:2009.14119.

    Repris verbatim d'architecture_1/train.py (le projet utilise l'ASL à la place
    de la BCE pour le déséquilibre multi-label sévère d'Open-i).
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, logits, targets):
        p     = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0)

        log_p   = torch.log(p.clamp(min=self.eps))
        log_1_p = torch.log((1 - p_neg).clamp(min=self.eps))

        loss_pos = (1 - p)  ** self.gamma_pos * log_p
        loss_neg =  p_neg   ** self.gamma_neg  * log_1_p

        return -(targets * loss_pos + (1 - targets) * loss_neg).mean()
