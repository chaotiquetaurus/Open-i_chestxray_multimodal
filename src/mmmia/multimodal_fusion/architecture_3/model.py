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
from transformers import AutoModel, AutoTokenizer, ViTModel

# Réutilisation de text_classification (convention sys.path du repo) : on injecte
# le paquet avant l'import, car model.py peut être importé avant dataset.py.
_TEXT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "..", "text_classification")
if _TEXT_ROOT not in sys.path:
    sys.path.insert(0, _TEXT_ROOT)

# Constante centralisée du projet (text_classification) — ne pas réinventer.
from models.cxr_bert_classifier import CXR_BERT_NAME  # noqa: E402

VIT_NAME = "codewithdark/vit-chest-xray"


def build_cxr_tokenizer(max_len: int = 256):
    """(backend_tokenizer, pad_id) de CXR-BERT, tronqué à max_len.

    Le tokenizer est apparié à l'encodeur texte CXR-BERT : il vit donc avec le
    modèle (cf. cxr_bert.py). On expose le backend `tokenizers.Tokenizer`
    (interface `encode_batch`) attendu par common.data.FusionDataset.
    """
    hf_tok = AutoTokenizer.from_pretrained(CXR_BERT_NAME, trust_remote_code=True)
    pad_id = hf_tok.pad_token_id
    tok = hf_tok.backend_tokenizer
    tok.enable_truncation(max_length=max_len)
    return tok, pad_id


class JointQueryBlock(nn.Module):
    """Un bloc Q-Former : SA([Q;T]) → CA(Q→Z) → FFN, chacune résiduelle + LN."""

    def __init__(self, d=768, n_heads=12, d_ff=3072, dropout=0.1, norm="post"):
        super().__init__()
        assert norm in ("post", "pre")
        self.norm = norm
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, q, T, Z, txt_pad=None, return_attn=False):
        B, n = q.size(0), q.size(1)

        # key_padding_mask pour la SA sur [Q; T] : True = ignoré (inverse HuggingFace).
        # Les n positions queries sont toujours visibles → zeros (False).
        kpm = None
        if txt_pad is not None:
            kpm = torch.cat(
                [torch.zeros(B, n, dtype=torch.bool, device=q.device), txt_pad], dim=1
            )

        if self.norm == "pre":
            # Pré-norm : le résidu q traverse HORS du LN → l'identité des requêtes
            # n'est jamais re-normalisée, elle persiste en profondeur (anti-collapse).
            qn = self.ln1(q)
            ctx_t = torch.cat([qn, T], dim=1)                             # (B, n+L, d)
            h, sa_w = self.self_attn(qn, ctx_t, ctx_t, key_padding_mask=kpm,
                                     need_weights=return_attn)
            q = q + h
            qn = self.ln2(q)
            h, ca_w = self.cross_attn(qn, Z, Z, need_weights=return_attn)
            q = q + h
            q = q + self.ffn(self.ln3(q))
        else:
            # Post-norm (défaut, comportement historique : LN après le résidu).
            ctx_t = torch.cat([q, T], dim=1)                              # (B, n+L, d)
            h, sa_w = self.self_attn(q, ctx_t, ctx_t, key_padding_mask=kpm,
                                     need_weights=return_attn)
            q = self.ln1(q + h)
            h, ca_w = self.cross_attn(q, Z, Z, need_weights=return_attn)
            q = self.ln2(q + h)
            q = self.ln3(q + self.ffn(q))

        return q, sa_w, ca_w


class QFormerHead(nn.Module):
    """N blocs JointQueryBlock + tête label-alignée (1 query par pathologie)."""

    def __init__(self, n_query=14, d=768, n_layers=3, n_labels=14,
                 n_heads=12, d_ff=3072, dropout=0.1, norm="post",
                 reinject_identity=False):
        super().__init__()
        assert n_query == n_labels, "label-aligné : une query par pathologie"
        self.norm = norm
        self.queries = nn.Parameter(torch.randn(1, n_query, d) * 0.02)
        self.blocks = nn.ModuleList(
            [JointQueryBlock(d, n_heads, d_ff, dropout, norm=norm) for _ in range(n_layers)]
        )
        # Ré-injection identité : signature label ré-ajoutée sur le résidu AVANT
        # chaque bloc → l'identité de la requête j ne décroît pas en profondeur.
        # N'a de sens qu'en pré-norm (en post-norm le LN la dilue quand même).
        self.label_emb = (nn.Parameter(torch.randn(1, n_query, d) * 0.02)
                          if reinject_identity else None)
        # Pré-norm : la sortie du dernier bloc n'est pas normalisée → LN final
        # avant la tête (échelle stable). Inutile en post-norm (déjà normalisé).
        self.final_ln = nn.LayerNorm(d) if norm == "pre" else None
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
            if self.label_emb is not None:
                q = q + self.label_emb        # re-tampon identité (sur le résidu)
            T_i = T[i] if is_per_layer else T
            q, sa_w, ca_w = blk(q, T_i, Z, txt_pad, return_attn)   # q : (B, 14, d)
            sa_all.append(sa_w)
            ca_all.append(ca_w)
        if self.final_ln is not None:
            q = self.final_ln(q)
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
        norm="post",                    # "post" (historique) | "pre" (anti-collapse)
        reinject_identity=False,        # ré-injecter E_label avant chaque bloc
        text_dropout=0.0,               # masquage de modalité texte (proba/échantillon)
        freeze_text=True,
        freeze_image=True,
        text_name=CXR_BERT_NAME,
        image_name=VIT_NAME,
    ):
        super().__init__()
        assert text_feature_mode in ("last", "deep")
        assert norm in ("post", "pre")
        self.pad_id = pad_id
        self.text_feature_mode = text_feature_mode
        self.text_dropout = text_dropout

        # ── Encodeurs ────────────────────────────────────────────────────
        self.text_encoder = AutoModel.from_pretrained(text_name, trust_remote_code=True)
        # add_pooling_layer=False : on n'utilise que last_hidden_state (197 tokens),
        # jamais pooler_output ; évite d'initialiser un pooler aléatoire inutilisé
        # (le checkpoint codewithdark/vit-chest-xray n'en a pas, c'est un ViTForImageClassification).
        self.image_encoder = ViTModel.from_pretrained(image_name, add_pooling_layer=False)

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
            norm=norm, reinject_identity=reinject_identity,
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

        # Modality dropout (ModDrop, Neverova et al. 2016) : à l'entraînement, avec
        # proba text_dropout, on masque TOUT le texte d'un échantillon → la voie
        # image doit suffire. Casse l'unimodal bias (le modèle « lit le rapport »).
        # Pas de NaN : chaque requête voit toujours les n requêtes dans la SA.
        if self.training and self.text_dropout > 0:
            drop = torch.rand(ids.size(0), device=ids.device) < self.text_dropout
            txt_pad = txt_pad | drop.unsqueeze(1)                # (B,1) broadcast → (B,L)

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
