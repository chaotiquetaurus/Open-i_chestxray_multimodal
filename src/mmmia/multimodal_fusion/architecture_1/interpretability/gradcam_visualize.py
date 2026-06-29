"""gradcam_visualize.py — Localization via Grad-CAM on ViT tokens.

The raw attention heatmaps (visualize_attention.py / compare_attention_
samples.py) turned out to be dominated by "attention sinks": a handful of
patches (often at the image border) receive a high weight regardless of
content, a phenomenon documented for ViTs (Darcet et al., 2023,
"Vision Transformers Need Registers").

This visualization instead uses Grad-CAM on the ViT tokens: we compute the
gradient of the TARGET pathology's logit with respect to each patch's
activation, giving a map that answers "which patches, if their activation
increased, would increase the predicted probability of this pathology" —
a causal, label-specific signal, unlike raw attention which doesn't depend
on the label being explained.

Usage:
    python gradcam_visualize.py \
        --checkpoint checkpoints/multimodal_fusion.pt \
        --tokenizer  ../text_classification/checkpoints/tokenizer.json \
        --csv        /content/drive/MyDrive/dataset_labeled.csv \
        --image_dir  /content/Png \
        --labels Pneumothorax Cardiomegaly Effusion Normal
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dataset import LABEL_COLS, load_paired_df              # noqa: E402
from visualize_attention import load_model, patch_grid_size, VAL_TF  # noqa: E402


def gradcam(model, input_ids, pixel_values, label_idx):
    """Grad-CAM on the raw ViT tokens (before cross-attention).

    cam[j] = ReLU( grad(logit_label, image_tokens[j]) · image_tokens[j] )
    """
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

    grad = image_tokens.grad[0]          # (197, 768)
    act = image_tokens.detach()[0]       # (197, 768)
    cam = F.relu((grad * act).sum(dim=-1))   # (197,)
    return cam, logits.detach().sigmoid()[0]


def overlay(ax, pil_image, cam_197, grid, title):
    """cam_197: tensor (197,) including the CLS token at position 0."""
    patches = cam_197[1:].reshape(grid, grid)   # drop CLS
    if patches.max() > 0:
        patches = patches / patches.max()
    heat = F.interpolate(
        patches.unsqueeze(0).unsqueeze(0),
        size=pil_image.size[::-1], mode="bilinear", align_corners=False,
    )[0, 0].detach().cpu().numpy()

    ax.imshow(pil_image)
    ax.imshow(heat, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.checkpoint, args.tokenizer, device)
    grid = patch_grid_size(model.vit)

    df = load_paired_df(args.csv)

    n = len(args.labels)
    fig, axes = plt.subplots(n, 2, figsize=(9, 4.2 * n))
    if n == 1:
        axes = axes[None, :]

    for i, label in enumerate(args.labels):
        matches = df.index[df[label] == 1].tolist()
        if not matches:
            print(f"No sample found for '{label}', skipped.")
            continue
        idx = matches[0]
        row = df.iloc[idx]

        img_path = os.path.join(args.image_dir, row["image_id"])
        pil_img = Image.open(img_path).convert("RGB")

        pixel_values = VAL_TF(pil_img).unsqueeze(0).to(device)
        enc = tok.encode(row["findings"])
        ids = torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0).to(device)

        label_idx = LABEL_COLS.index(label)
        cam, probs = gradcam(model, ids, pixel_values, label_idx)

        axes[i, 0].imshow(pil_img)
        axes[i, 0].set_title(
            f"{label}  (idx={idx})\nP({label})={probs[label_idx]:.2f}",
            fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")

        overlay(axes[i, 1], pil_img, cam, grid, f"Grad-CAM — {label}")

    fig.suptitle(
        "Grad-CAM on ViT tokens — label-conditioned localization\n"
        "(hotspots should change position from one pathology to another)",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.tight_layout()

    out_path = args.out or os.path.join(ROOT, "outputs", "gradcam_comparison.png")
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
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main(args)
