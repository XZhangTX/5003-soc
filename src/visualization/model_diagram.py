import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils import ensure_dir


NUS_BLUE = "#003D7C"
NUS_LIGHT_BLUE = "#DCEBFA"
NUS_MID_BLUE = "#8FB9E3"
NUS_ORANGE = "#EF7C00"
NUS_LIGHT_ORANGE = "#FCE4CC"
NUS_TEXT = "#1F2937"
NUS_GREY = "#EEF2F7"


def _add_box(ax, x, y, w, h, text, fc=NUS_GREY, ec=NUS_BLUE, fontsize=11):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=NUS_TEXT)


def _add_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color=NUS_BLUE,
        )
    )


def _draw_conv_transformer(ax):
    ax.set_title("Conv-Transformer Regressor", fontsize=16, pad=14, color=NUS_BLUE, fontweight="bold")
    boxes = [
        (0.03, 0.42, 0.13, 0.16, "Input Spectrum\n[L x d_in]", NUS_LIGHT_BLUE),
        (0.20, 0.42, 0.16, 0.16, "Conv1d Stem\nstride=1", NUS_MID_BLUE),
        (0.40, 0.42, 0.17, 0.16, "Conv1d Tokenizer\nstride=patch_stride", NUS_MID_BLUE),
        (0.61, 0.42, 0.12, 0.16, "Token Embedding", NUS_LIGHT_ORANGE),
        (0.61, 0.71, 0.12, 0.12, "CLS Token", NUS_LIGHT_ORANGE),
        (0.77, 0.42, 0.17, 0.16, "Transformer Encoder\nMHSA + FFN", NUS_LIGHT_ORANGE),
        (0.77, 0.15, 0.17, 0.12, "Positional\nEncoding", NUS_GREY),
        (0.77, 0.72, 0.17, 0.10, "Token Sequence\n[1 + L' x d_model]", NUS_GREY),
        (0.77, 0.00, 0.17, 0.10, "CLS Output\nLayerNorm", NUS_GREY),
        (0.77, -0.18, 0.17, 0.12, "MLP Head\nLinear-GELU-Dropout", NUS_LIGHT_BLUE),
        (0.77, -0.38, 0.17, 0.10, "Scalar Prediction\nSOC / SOH", NUS_LIGHT_BLUE),
    ]
    for x, y, w, h, txt, color in boxes:
        _add_box(ax, x, y, w, h, txt, fc=color)

    _add_arrow(ax, 0.16, 0.50, 0.20, 0.50)
    _add_arrow(ax, 0.36, 0.50, 0.40, 0.50)
    _add_arrow(ax, 0.57, 0.50, 0.61, 0.50)
    _add_arrow(ax, 0.73, 0.50, 0.77, 0.50)
    _add_arrow(ax, 0.69, 0.71, 0.77, 0.76)
    _add_arrow(ax, 0.85, 0.15, 0.85, 0.42)
    _add_arrow(ax, 0.85, 0.72, 0.85, 0.58)
    _add_arrow(ax, 0.85, 0.42, 0.85, 0.10)
    _add_arrow(ax, 0.85, 0.00, 0.85, -0.06)
    _add_arrow(ax, 0.85, -0.18, 0.85, -0.28)

    ax.text(0.485, 0.64, "Local spectral motif extraction\nand sequence compression", ha="center", va="center", fontsize=10, color=NUS_BLUE)
    ax.text(0.855, 0.30, "Global interaction over tokens", ha="center", va="center", fontsize=10, color=NUS_ORANGE)


def _draw_transformer(ax):
    ax.set_title("Vanilla Transformer Regressor", fontsize=16, pad=14, color=NUS_BLUE, fontweight="bold")
    boxes = [
        (0.05, 0.42, 0.15, 0.16, "Input Spectrum\n[L x d_in]", NUS_LIGHT_BLUE),
        (0.26, 0.42, 0.16, 0.16, "Linear Input\nProjection", NUS_MID_BLUE),
        (0.48, 0.42, 0.16, 0.16, "Frequency Embedding\n+ Optional Gate", NUS_LIGHT_ORANGE),
        (0.48, 0.70, 0.16, 0.10, "CLS Token", NUS_LIGHT_ORANGE),
        (0.70, 0.42, 0.18, 0.16, "Transformer Encoder\nMHSA + FFN", NUS_LIGHT_ORANGE),
        (0.70, 0.16, 0.18, 0.10, "Positional\nEncoding", NUS_GREY),
        (0.70, -0.04, 0.18, 0.10, "CLS Output\nLayerNorm", NUS_GREY),
        (0.70, -0.22, 0.18, 0.12, "MLP Head", NUS_LIGHT_BLUE),
        (0.70, -0.40, 0.18, 0.10, "Scalar Prediction", NUS_LIGHT_BLUE),
    ]
    for x, y, w, h, txt, color in boxes:
        _add_box(ax, x, y, w, h, txt, fc=color)

    _add_arrow(ax, 0.20, 0.50, 0.26, 0.50)
    _add_arrow(ax, 0.42, 0.50, 0.48, 0.50)
    _add_arrow(ax, 0.64, 0.50, 0.70, 0.50)
    _add_arrow(ax, 0.56, 0.70, 0.70, 0.76)
    _add_arrow(ax, 0.79, 0.16, 0.79, 0.42)
    _add_arrow(ax, 0.79, 0.42, 0.79, 0.06)
    _add_arrow(ax, 0.79, -0.04, 0.79, -0.10)
    _add_arrow(ax, 0.79, -0.22, 0.79, -0.30)


def _draw_mlp(ax):
    ax.set_title("MLP Baseline Regressor", fontsize=16, pad=14, color=NUS_BLUE, fontweight="bold")
    boxes = [
        (0.06, 0.40, 0.16, 0.18, "Input Spectrum\n[L x d_in]", NUS_LIGHT_BLUE),
        (0.30, 0.40, 0.16, 0.18, "Flatten\n[L * d_in]", NUS_MID_BLUE),
        (0.54, 0.40, 0.16, 0.18, "Hidden Layers\nLinear + Activation", NUS_LIGHT_ORANGE),
        (0.54, 0.10, 0.16, 0.12, "Regression Head", NUS_GREY),
        (0.54, -0.12, 0.16, 0.10, "Scalar Prediction", NUS_LIGHT_BLUE),
    ]
    for x, y, w, h, txt, color in boxes:
        _add_box(ax, x, y, w, h, txt, fc=color)

    _add_arrow(ax, 0.22, 0.49, 0.30, 0.49)
    _add_arrow(ax, 0.46, 0.49, 0.54, 0.49)
    _add_arrow(ax, 0.62, 0.40, 0.62, 0.22)
    _add_arrow(ax, 0.62, 0.10, 0.62, -0.02)


def _draw_xgb(ax):
    ax.set_title("XGBoost Baseline Regressor", fontsize=16, pad=14, color=NUS_BLUE, fontweight="bold")
    boxes = [
        (0.06, 0.40, 0.16, 0.18, "Input Spectrum\n[L x d_in]", NUS_LIGHT_BLUE),
        (0.30, 0.40, 0.16, 0.18, "Flatten\n[L * d_in]", NUS_MID_BLUE),
        (0.54, 0.40, 0.18, 0.18, "Gradient-Boosted\nDecision Trees", NUS_LIGHT_ORANGE),
        (0.54, 0.10, 0.18, 0.12, "Tree Ensemble\nAggregation", NUS_GREY),
        (0.54, -0.12, 0.18, 0.10, "Scalar Prediction", NUS_LIGHT_BLUE),
    ]
    for x, y, w, h, txt, color in boxes:
        _add_box(ax, x, y, w, h, txt, fc=color)

    _add_arrow(ax, 0.22, 0.49, 0.30, 0.49)
    _add_arrow(ax, 0.46, 0.49, 0.54, 0.49)
    _add_arrow(ax, 0.63, 0.40, 0.63, 0.22)
    _add_arrow(ax, 0.63, 0.10, 0.63, -0.02)


def draw_model_diagram(model_name: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.9)
    ax.axis("off")

    if model_name == "conv_transformer":
        _draw_conv_transformer(ax)
    elif model_name == "transformer":
        _draw_transformer(ax)
    elif model_name == "mlp":
        _draw_mlp(ax)
    elif model_name == "xgb":
        _draw_xgb(ax)
    else:
        raise ValueError(f"Unsupported model_name={model_name}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(args):
    out_dir = ensure_dir(Path(args.output_root))
    out_path = out_dir / f"{args.model_name}_diagram.png"
    draw_model_diagram(args.model_name, out_path)
    print(f"Saved model diagram to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw paper-ready model architecture diagrams")
    parser.add_argument(
        "--model-name",
        type=str,
        default="conv_transformer",
        choices=["conv_transformer", "transformer", "mlp", "xgb"],
    )
    parser.add_argument("--output-root", type=str, default="output/model_diagrams")
    main(parser.parse_args())
