"""gradcam_aggregate.py — Averaged, norm-corrected Grad-CAM per pathology.

Single-sample Grad-CAM (gradcam_visualize.py) revealed two problems:

  1. Some attributions were dominated by a handful of patches with unusually
     large activation NORM (likely ViT "attention-sink" / register tokens),
     regardless of their actual gradient relevance. Fix: normalize each
     patch's activation vector to unit length before the dot product with
     its gradient, so only the gradient's alignment with the activation
     DIRECTION matters — not its raw magnitude.
  2. Single-sample maps are noisy (e.g. the lateral Effusion case looked
     like random noise). Fix: average the normalized Grad-CAM grid over
     many positive samples per label, cancelling out per-sample noise and
     revealing whether a genuine label-conditioned spatial pattern exists.

Usage:
    python gradcam_aggregate.py \
        --checkpoint checkpoints/multimodal_fusion.pt \
        --tokenizer  ../text_classification/checkpoints/tokenizer.json \
        --csv        /content/drive/MyDrive/dataset_labeled.csv \
        --image_dir  /content/Png \
        --labels Pneumothorax Cardiomegaly Effusion Normal \
        --n_samples 15
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dataset import LABEL_COLS, load_paired_df              # noqa: E402
from visualize_attention import load_model, patch_grid_size, VAL_TF  # noqa: E402


def normalized_gradcam(model, input_ids, pixel_values, label_idx):
    """Grad-CAM with per-token activation normalization (removes the
    activation-norm confound from outlier/attention-sink tokens)."""
    model.zero_grad()

    image_tokens = model.vit(pixel_values=pixel_values).last_hidden_state
    image_tokens.retain_grad()

    text_tokens = model.text_encoder(input_ids)
    ca = model.cross_attn
    T = ca.text_proj(text_tokens)
    I = ca.image_proj(image_tokens)
    T_ca, _ = ca.t2i(T, I, I)
    I_ca, _ = ca.i2t(I, T, T)
    T = ca.norm_t1(T + T_ca)
    I = ca.norm_i1(I + I_ca)
    T = ca.norm_t2(T + ca.ffn_t(T))
    I = ca.norm_i2(I + ca.ffn_i(I))

    text_repr = T[:, 0]
    pool_w = F.softmax(model.image_pool_w(I), dim=1)
    image_repr = (pool_w * I).sum(dim=1)
    logits = model.head(torch.cat([text_repr, image_repr], dim=1))

    target = logits[0, label_idx]
    target.backward()

    grad = image_tokens.grad[0]                                # (197, 768)
    act = image_tokens.detach()[0]                             # (197, 768)
    act_unit = act / (act.norm(dim=-1, keepdim=True) + 1e-8)    # remove norm confound
    cam = F.relu((grad * act_unit).sum(dim=-1))                 # (197,)

    return cam.detach().cpu(), logits.detach().sigmoid()[0].cpu()


def average_cam_for_label(model, tok, df, image_dir, label, label_idx,
                          grid, n_samples, device, seed=42):
    matches = df.index[df[label] == 1].tolist()
    if not matches:
        return None, None, 0

    rng = random.Random(seed)
    chosen = rng.sample(matches, min(n_samples, len(matches)))

    grids = []
    first_image = None
    for idx in chosen:
        row = df.iloc[idx]
        img_path = os.path.join(image_dir, row["image_id"])
        pil_img = Image.open(img_path).convert("RGB")
        if first_image is None:
            first_image = pil_img

        pixel_values = VAL_TF(pil_img).unsqueeze(0).to(device)
        enc = tok.encode(row["findings"])
        ids = torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0).to(device)

        cam, _ = normalized_gradcam(model, ids, pixel_values, label_idx)
        patches = cam[1:].reshape(grid, grid)              # drop CLS token, (14, 14)
        if patches.max() > 0:
            patches = patches / patches.max()
        grids.append(patches.numpy())

    avg_grid = np.mean(np.stack(grids), axis=0)             # (14, 14)
    return avg_grid, first_image, len(chosen)


def plot_label_row(axes_row, pil_image, avg_grid, label, n_used):
    # Column 1 — raw averaged grid (the actual evidence, no image dependency)
    im = axes_row[0].imshow(avg_grid, cmap="jet", vmin=0, vmax=1)
    axes_row[0].set_title(f"{label} — averaged CAM grid (n={n_used})",
                          fontsize=12, fontweight="bold")
    axes_row[0].set_xticks([])
    axes_row[0].set_yticks([])
    plt.colorbar(im, ax=axes_row[0], fraction=0.046, pad=0.04)

    # Column 2 — overlay on one representative X-ray for anatomical context
    heat = F.interpolate(
        torch.tensor(avg_grid).unsqueeze(0).unsqueeze(0),
        size=pil_image.size[::-1], mode="bilinear", align_corners=False,
    )[0, 0].numpy()
    axes_row[1].imshow(pil_image)
    axes_row[1].imshow(heat, cmap="jet", alpha=0.45)
    axes_row[1].set_title(f"{label} — overlay on a representative X-ray",
                          fontsize=12, fontweight="bold")
    axes_row[1].axis("off")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.checkpoint, args.tokenizer, device)
    grid = patch_grid_size(model.vit)
    df = load_paired_df(args.csv)

    n = len(args.labels)
    fig, axes = plt.subplots(n, 2, figsize=(11, 5.2 * n))
    if n == 1:
        axes = axes[None, :]

    for i, label in enumerate(args.labels):
        label_idx = LABEL_COLS.index(label)
        avg_grid, ref_img, n_used = average_cam_for_label(
            model, tok, df, args.image_dir, label, label_idx,
            grid, args.n_samples, device, seed=args.seed,
        )
        if avg_grid is None:
            print(f"No samples found for '{label}', skipping.")
            continue

        plot_label_row(axes[i], ref_img, avg_grid, label, n_used)

    fig.suptitle(
        "Averaged, norm-corrected Grad-CAM per pathology\n"
        "(activation vectors normalized to unit length before the gradient dot product)",
        fontsize=13.5, fontweight="bold", y=1.0,
    )
    fig.tight_layout()

    out_path = args.out or os.path.join(ROOT, "outputs", "gradcam_aggregate.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--labels", nargs="+",
                        default=["Pneumothorax", "Cardiomegaly", "Effusion", "Normal"])
    parser.add_argument("--n_samples", type=int, default=15,
                        help="Number of positive samples to average per label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main(args)
