"""dataset.py — Dataset pairé texte + image pour le Q-Former label-aligné.

Source de données : data/shared/dataset_labeled_major.csv (déjà pairé) —
  image_id   : nom du fichier PNG (ex. CXR3037_IM-1410-1001.png)
  indication : motif clinique  ┐  entrées texte (jamais `impression` : anti-fuite)
  findings   : corps du rapport ┘
  <labels>   : colonnes binaires 0/1 (on garde les 14 NIH, ordre canonique)

Le texte est tokenisé avec le tokenizer natif de CXR-BERT (backend `tokenizers`),
exactement comme text_classification/scripts/cxr_bert.py.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

# ── Réutilisation des constantes/utilitaires de text_classification ──────────
# Convention du repo (cf. architecture_1/train.py) : on insère text_classification
# dans sys.path puis on importe sans préfixe.
_TEXT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "..", "text_classification")
if _TEXT_ROOT not in sys.path:
    sys.path.insert(0, _TEXT_ROOT)

from data.datasets import LABELS_5, LABELS_14, META_COLS  # noqa: E402
from data.splits import text_group_key                    # noqa: E402
from models.cxr_bert_classifier import CXR_BERT_NAME       # noqa: E402

MODE_LABELS = {5: LABELS_5, 14: LABELS_14}  # 21 = toutes les colonnes du CSV
TEXT_COLS = ["indication", "findings"]      # `impression` EXCLU (anti-fuite)

# ── Transforms ViT (identiques à architecture_1) ─────────────────────────────
VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD  = [0.5, 0.5, 0.5]

TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.85, 1.15), shear=10),
    transforms.ElasticTransform(alpha=40.0, sigma=5.0),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
])

VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
])


def resolve_label_cols(df: pd.DataFrame, mode: int = 14) -> list[str]:
    """Colonnes labels dans l'ordre canonique (source de vérité : LABELS_* / CSV).

    mode 5/14 : sous-ensemble figé filtré sur les colonnes présentes.
    mode 21   : toutes les colonnes du CSV hors méta (ordre du CSV).
    """
    if mode in MODE_LABELS:
        return [c for c in MODE_LABELS[mode] if c in df.columns]
    return [c for c in df.columns if c not in META_COLS]


def build_cxr_tokenizer(max_len: int = 256):
    """Retourne (backend_tokenizer, pad_id) de CXR-BERT, tronqué à max_len.

    Reproduit cxr_bert.py:136-142 : on récupère le tokenizer HF puis son backend
    `tokenizers.Tokenizer` (interface `encode_batch`)."""
    hf_tok = AutoTokenizer.from_pretrained(CXR_BERT_NAME, trust_remote_code=True)
    pad_id = hf_tok.pad_token_id
    tok = hf_tok.backend_tokenizer
    tok.enable_truncation(max_length=max_len)
    return tok, pad_id


def load_paired_df(csv_path: str | Path) -> pd.DataFrame:
    """Charge le CSV image-level et supprime les lignes sans findings."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["findings"]).reset_index(drop=True)
    df["findings"] = df["findings"].astype(str).str.strip()
    df = df[df["findings"] != ""].reset_index(drop=True)
    return df


def build_texts(df: pd.DataFrame) -> list[str]:
    """Concatène indication + findings (mêmes colonnes que le BERT texte)."""
    cols = [c for c in TEXT_COLS if c in df.columns]
    return df[cols].fillna("").agg(" ".join, axis=1).str.strip().tolist()


class FusionDataset(Dataset):
    """Retourne (token_ids, pixel_values, labels) pour chaque image."""

    def __init__(self, df, label_cols, tokenizer, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.image_dir = Path(image_dir)
        self.transform = transform
        # Encodage texte en batch à l'init (gain de temps).
        self.enc = tokenizer.encode_batch(build_texts(self.df))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        ids = torch.tensor(self.enc[idx].ids, dtype=torch.long)

        img_path = self.image_dir / self.df.at[idx, "image_id"]
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(img) if self.transform is not None \
            else transforms.functional.to_tensor(img.resize((224, 224)))

        labels = torch.tensor(
            self.df.loc[idx, self.label_cols].values.astype(float),
            dtype=torch.float32,
        )
        return ids, pixel_values, labels


def fusion_collate(batch):
    """Pad les séquences texte à la longueur max du batch, stack images + labels."""
    ids_list, pixels_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in ids_list)
    ids_pad = torch.stack([F.pad(ids, (0, max_len - len(ids))) for ids in ids_list])
    pixels = torch.stack(list(pixels_list))
    labels = torch.stack(list(labels_list))
    return ids_pad, pixels, labels


def build_groups(df: pd.DataFrame):
    """Clé de groupe (hash du texte) pour un split train/val/test sans fuite."""
    return text_group_key(build_texts(df))
