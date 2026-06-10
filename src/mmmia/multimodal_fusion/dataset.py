"""dataset.py — Dataset pairé texte + image pour l'entraînement multimodal.

Source de données : dataset_labeled.csv  (déjà pairé : image_id + findings + labels)
  image_id   : nom du fichier PNG (ex. CXR3037_IM-1410-1001.png)
  findings   : texte du rapport radiologique
  <labels>   : colonnes binaires 0/1 pour chacun des 21 labels

Usage rapide:
    ds = MultimodalDataset(df, LABEL_COLS, tok, image_dir, transform)
    loader = DataLoader(ds, batch_size=16, collate_fn=multimodal_collate)
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# Labels identiques au notebook cv_model_vit_04
LABEL_COLS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Pneumonia", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Hernia", "Fracture", "Pleural_Thickening", "Opacity", "Consolidation",
    "Granuloma", "Calcinosis", "Scoliosis", "Atherosclerosis", "Normal",
]


def load_paired_df(csv_path: str | Path) -> pd.DataFrame:
    """Charge dataset_labeled.csv et supprime les lignes sans findings."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["findings"]).reset_index(drop=True)
    df["findings"] = df["findings"].str.strip()
    df = df[df["findings"] != ""].reset_index(drop=True)
    return df


class MultimodalDataset(Dataset):
    """
    Dataset retournant (token_ids, pixel_values, labels) pour chaque image.

    Args:
        df          : DataFrame avec colonnes image_id, findings, <label_cols>
        label_cols  : liste des colonnes label à utiliser
        tokenizer   : tokenizers.Tokenizer (encode_batch)
        image_dir   : répertoire contenant les fichiers PNG
        transform   : torchvision transform appliqué sur chaque image PIL
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_cols: list[str],
        tokenizer,
        image_dir: str | Path,
        transform=None,
    ):
        self.df         = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.image_dir  = Path(image_dir)
        self.transform  = transform

        # Encodage texte en batch pour gagner du temps à l'init
        self.enc = tokenizer.encode_batch(
            self.df["findings"].fillna("").tolist()
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # ── Texte ────────────────────────────────────────────────────────
        ids = torch.tensor(self.enc[idx].ids, dtype=torch.long)

        # ── Image ────────────────────────────────────────────────────────
        img_path = self.image_dir / self.df.at[idx, "image_id"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            pixel_values = self.transform(img)
        else:
            # Fallback minimal si aucun transform n'est fourni
            import torchvision.transforms.functional as TF
            pixel_values = TF.to_tensor(img.resize((224, 224)))

        # ── Labels ───────────────────────────────────────────────────────
        labels = torch.tensor(
            self.df.loc[idx, self.label_cols].values.astype(float),
            dtype=torch.float32,
        )

        return ids, pixel_values, labels


def multimodal_collate(batch):
    """
    Collate pour DataLoader : pad les séquences texte à la longueur max du batch,
    stack les images (déjà de taille fixe) et les labels.
    """
    ids_list, pixels_list, labels_list = zip(*batch)

    max_len  = max(len(ids) for ids in ids_list)
    ids_pad  = torch.stack(
        [F.pad(ids, (0, max_len - len(ids))) for ids in ids_list]
    )
    pixels = torch.stack(list(pixels_list))
    labels = torch.stack(list(labels_list))

    return ids_pad, pixels, labels
