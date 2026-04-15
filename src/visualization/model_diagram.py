import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from src.utils import ensure_dir


NUS_BLUE = "#003D7C"
NUS_ORANGE = "#EF7C00"
NUS_TEXT = "#1F2937"
LINE_GREY = "#5B6673"
SOFT_GREY = "#F4F6F8"


def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _tint(hex_color: str, alpha: float):
    base = _hex_to_rgb(hex_color)
    white = (255, 255, 255)
    mixed = tuple(int((1 - alpha) * white[i] + alpha * base[i]) for i in range(3))
    return _rgb_to_hex(mixed)


BLUE_FILL = _tint(NUS_BLUE, 0.14)
BLUE_FILL_DARK = _tint(NUS_BLUE, 0.24)
ORANGE_FILL = _tint(NUS_ORANGE, 0.16)
ORANGE_FILL_DARK = _tint(NUS_ORANGE, 0.24)


mpl.rcParams["font.family"] = "DejaVu Serif"
mpl.rcParams["axes.titleweight"] = "bold"


def _add_box(ax, x, y, w, h, text, fc, ec, fontsize=10.5, lw=1.3):
    rect = Rectangle((x, y), w, h, linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=NUS_TEXT)


def _add_arrow(ax, x1, y1, x2, y2, color=LINE_GREY):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def _add_group_label(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=11.5, fontweight="bold", color=color, ha="left", va="bottom")


def _prepare_canvas(title: str):
    fig, ax = plt.subplots(figsize=(13.5, 5.8), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.95, title, fontsize=17, fontweight="bold", color=NUS_BLUE, ha="left", va="top")
    return fig, ax


def _prepare_subplot(ax, title: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.95, title, fontsize=14, fontweight="bold", color=NUS_BLUE, ha="left", va="top")


def _draw_conv_transformer():
    fig, ax = _prepare_canvas("Conv-Transformer Architecture")
    _add_group_label(ax, 0.03, 0.76, "Input Representation", NUS_BLUE)
    _add_group_label(ax, 0.29, 0.76, "Local Tokenization", NUS_BLUE)
    _add_group_label(ax, 0.57, 0.76, "Sequence Modeling", NUS_ORANGE)
    _add_group_label(ax, 0.84, 0.76, "Regression", NUS_BLUE)

    _add_box(ax, 0.03, 0.46, 0.14, 0.18, "Ultrasonic Spectrum\n$L \\times d_{in}$", BLUE_FILL, NUS_BLUE)
    _add_box(ax, 0.25, 0.46, 0.14, 0.18, "Conv1d Stem\nstride = 1", BLUE_FILL_DARK, NUS_BLUE)
    _add_box(ax, 0.43, 0.46, 0.14, 0.18, "Conv1d Tokenizer\nstride = patch stride", BLUE_FILL_DARK, NUS_BLUE)
    _add_box(ax, 0.43, 0.18, 0.14, 0.12, "Token Sequence\n$L' \\times d_{model}$", SOFT_GREY, LINE_GREY, fontsize=10)

    _add_box(ax, 0.63, 0.58, 0.13, 0.14, "CLS Token", ORANGE_FILL, NUS_ORANGE)
    _add_box(ax, 0.63, 0.35, 0.13, 0.14, "Token Embedding", ORANGE_FILL, NUS_ORANGE)
    _add_box(ax, 0.63, 0.12, 0.13, 0.14, "Positional Encoding", SOFT_GREY, LINE_GREY)
    _add_box(ax, 0.80, 0.40, 0.15, 0.24, "Transformer Encoder\nMHSA + FFN\nrepeated $N$ layers", ORANGE_FILL_DARK, NUS_ORANGE)

    _add_box(ax, 0.80, 0.15, 0.15, 0.12, "CLS Output\nLayerNorm", SOFT_GREY, LINE_GREY)
    _add_box(ax, 0.80, -0.02 + 0.08, 0.15, 0.12, "MLP Head", BLUE_FILL, NUS_BLUE)
    _add_box(ax, 0.80, -0.19 + 0.08, 0.15, 0.12, "Scalar Prediction\nSOC / SOH", BLUE_FILL, NUS_BLUE)

    _add_arrow(ax, 0.17, 0.55, 0.25, 0.55)
    _add_arrow(ax, 0.39, 0.55, 0.43, 0.55)
    _add_arrow(ax, 0.57, 0.55, 0.63, 0.42)
    _add_arrow(ax, 0.57, 0.24, 0.63, 0.42)
    _add_arrow(ax, 0.695, 0.58, 0.80, 0.58, color=NUS_ORANGE)
    _add_arrow(ax, 0.695, 0.42, 0.80, 0.52, color=NUS_ORANGE)
    _add_arrow(ax, 0.695, 0.19, 0.80, 0.46, color=LINE_GREY)
    _add_arrow(ax, 0.875, 0.40, 0.875, 0.27)
    _add_arrow(ax, 0.875, 0.15, 0.875, 0.10)
    _add_arrow(ax, 0.875, -0.02 + 0.08, 0.875, -0.07 + 0.08)

    ax.text(0.50, 0.69, "Overlapping local motif extraction and sequence compression", fontsize=10, color=LINE_GREY, ha="center")
    ax.text(0.875, 0.68, "Global token interaction", fontsize=10, color=NUS_ORANGE, ha="center")
    return fig


def _draw_conv_transformer_on_ax(ax):
    _prepare_subplot(ax, "Conv-Transformer")
    _add_box(ax, 0.04, 0.40, 0.18, 0.18, "Spectrum\n$L \\times d_{in}$", BLUE_FILL, NUS_BLUE, fontsize=9.5)
    _add_box(ax, 0.30, 0.40, 0.18, 0.18, "Conv Stem", BLUE_FILL_DARK, NUS_BLUE, fontsize=9.5)
    _add_box(ax, 0.56, 0.40, 0.18, 0.18, "Tokenizer\n$L' \\times d_{model}$", BLUE_FILL_DARK, NUS_BLUE, fontsize=9.5)
    _add_box(ax, 0.80, 0.40, 0.16, 0.18, "Transformer\nEncoder", ORANGE_FILL_DARK, NUS_ORANGE, fontsize=9.5)
    _add_box(ax, 0.80, 0.12, 0.16, 0.12, "MLP Head", BLUE_FILL, NUS_BLUE, fontsize=9.5)
    _add_arrow(ax, 0.22, 0.49, 0.30, 0.49)
    _add_arrow(ax, 0.48, 0.49, 0.56, 0.49)
    _add_arrow(ax, 0.74, 0.49, 0.80, 0.49)
    _add_arrow(ax, 0.88, 0.40, 0.88, 0.24)


def _draw_transformer():
    fig, ax = _prepare_canvas("Vanilla Transformer Baseline")
    _add_group_label(ax, 0.04, 0.76, "Input Projection", NUS_BLUE)
    _add_group_label(ax, 0.39, 0.76, "Token Preparation", NUS_ORANGE)
    _add_group_label(ax, 0.68, 0.76, "Regression", NUS_BLUE)

    _add_box(ax, 0.04, 0.46, 0.16, 0.18, "Ultrasonic Spectrum\n$L \\times d_{in}$", BLUE_FILL, NUS_BLUE)
    _add_box(ax, 0.27, 0.46, 0.16, 0.18, "Linear Projection\n$L \\times d_{model}$", BLUE_FILL_DARK, NUS_BLUE)
    _add_box(ax, 0.50, 0.58, 0.14, 0.12, "CLS Token", ORANGE_FILL, NUS_ORANGE)
    _add_box(ax, 0.50, 0.40, 0.14, 0.12, "Frequency Embedding", ORANGE_FILL, NUS_ORANGE)
    _add_box(ax, 0.50, 0.22, 0.14, 0.12, "Frequency Gate\n(optional)", ORANGE_FILL, NUS_ORANGE)
    _add_box(ax, 0.50, 0.04, 0.14, 0.12, "Positional Encoding", SOFT_GREY, LINE_GREY)
    _add_box(ax, 0.72, 0.40, 0.18, 0.24, "Transformer Encoder\nMHSA + FFN\nrepeated $N$ layers", ORANGE_FILL_DARK, NUS_ORANGE)
    _add_box(ax, 0.72, 0.16, 0.18, 0.10, "CLS Output\nLayerNorm", SOFT_GREY, LINE_GREY)
    _add_box(ax, 0.72, 0.00, 0.18, 0.10, "MLP Head", BLUE_FILL, NUS_BLUE)
    _add_box(ax, 0.72, -0.16 + 0.10, 0.18, 0.10, "Scalar Prediction", BLUE_FILL, NUS_BLUE)

    _add_arrow(ax, 0.20, 0.55, 0.27, 0.55)
    _add_arrow(ax, 0.43, 0.55, 0.50, 0.46)
    _add_arrow(ax, 0.57, 0.64, 0.72, 0.58, color=NUS_ORANGE)
    _add_arrow(ax, 0.57, 0.46, 0.72, 0.52, color=NUS_ORANGE)
    _add_arrow(ax, 0.57, 0.28, 0.72, 0.46, color=NUS_ORANGE)
    _add_arrow(ax, 0.57, 0.10, 0.72, 0.40, color=LINE_GREY)
    _add_arrow(ax, 0.81, 0.40, 0.81, 0.26)
    _add_arrow(ax, 0.81, 0.16, 0.81, 0.10)
    _add_arrow(ax, 0.81, 0.00, 0.81, -0.01)
    return fig


def _draw_transformer_on_ax(ax):
    _prepare_subplot(ax, "Transformer")
    _add_box(ax, 0.05, 0.40, 0.18, 0.18, "Spectrum\n$L \\times d_{in}$", BLUE_FILL, NUS_BLUE, fontsize=9.5)
    _add_box(ax, 0.31, 0.40, 0.20, 0.18, "Linear Projection", BLUE_FILL_DARK, NUS_BLUE, fontsize=9.5)
    _add_box(ax, 0.59, 0.40, 0.18, 0.18, "Embeddings\n+ Gate", ORANGE_FILL, NUS_ORANGE, fontsize=9.5)
    _add_box(ax, 0.83, 0.40, 0.14, 0.18, "Encoder", ORANGE_FILL_DARK, NUS_ORANGE, fontsize=9.5)
    _add_box(ax, 0.83, 0.12, 0.14, 0.12, "Head", BLUE_FILL, NUS_BLUE, fontsize=9.5)
    _add_arrow(ax, 0.23, 0.49, 0.31, 0.49)
    _add_arrow(ax, 0.51, 0.49, 0.59, 0.49)
    _add_arrow(ax, 0.77, 0.49, 0.83, 0.49)
    _add_arrow(ax, 0.90, 0.40, 0.90, 0.24)


def _draw_mlp():
    fig, ax = _prepare_canvas("MLP Baseline")
    _add_group_label(ax, 0.08, 0.76, "Feature Vectorization", NUS_BLUE)
    _add_group_label(ax, 0.46, 0.76, "Fully Connected Regression", NUS_ORANGE)

    _add_box(ax, 0.08, 0.44, 0.17, 0.20, "Ultrasonic Spectrum\n$L \\times d_{in}$", BLUE_FILL, NUS_BLUE)
    _add_box(ax, 0.33, 0.44, 0.17, 0.20, "Flatten\n$L \\cdot d_{in}$", BLUE_FILL_DARK, NUS_BLUE)
    _add_box(ax, 0.58, 0.50, 0.17, 0.14, "Hidden Layer 1", ORANGE_FILL, NUS_ORANGE)
    _add_box(ax, 0.58, 0.28, 0.17, 0.14, "Hidden Layer 2", ORANGE_FILL, NUS_ORANGE)
    _add_box(ax, 0.58, 0.06, 0.17, 0.14, "Regression Head", SOFT_GREY, LINE_GREY)
    _add_box(ax, 0.58, -0.16 + 0.10, 0.17, 0.10, "Scalar Prediction", BLUE_FILL, NUS_BLUE)

    _add_arrow(ax, 0.25, 0.54, 0.33, 0.54)
    _add_arrow(ax, 0.50, 0.54, 0.58, 0.57)
    _add_arrow(ax, 0.665, 0.50, 0.665, 0.42)
    _add_arrow(ax, 0.665, 0.28, 0.665, 0.20)
    _add_arrow(ax, 0.665, 0.06, 0.665, 0.04)
    return fig


def _draw_mlp_on_ax(ax):
    _prepare_subplot(ax, "MLP")
    _add_box(ax, 0.08, 0.40, 0.18, 0.18, "Spectrum\n$L \\times d_{in}$", BLUE_FILL, NUS_BLUE, fontsize=9.5)
    _add_box(ax, 0.34, 0.40, 0.18, 0.18, "Flatten", BLUE_FILL_DARK, NUS_BLUE, fontsize=9.5)
    _add_box(ax, 0.60, 0.40, 0.18, 0.18, "Hidden Layers", ORANGE_FILL, NUS_ORANGE, fontsize=9.5)
    _add_box(ax, 0.84, 0.40, 0.12, 0.18, "Head", BLUE_FILL, NUS_BLUE, fontsize=9.5)
    _add_arrow(ax, 0.26, 0.49, 0.34, 0.49)
    _add_arrow(ax, 0.52, 0.49, 0.60, 0.49)
    _add_arrow(ax, 0.78, 0.49, 0.84, 0.49)


def _draw_xgb():
    fig, ax = _prepare_canvas("XGBoost Baseline")
    _add_group_label(ax, 0.08, 0.76, "Feature Vectorization", NUS_BLUE)
    _add_group_label(ax, 0.46, 0.76, "Tree Ensemble Regression", NUS_ORANGE)

    _add_box(ax, 0.08, 0.44, 0.17, 0.20, "Ultrasonic Spectrum\n$L \\times d_{in}$", BLUE_FILL, NUS_BLUE)
    _add_box(ax, 0.33, 0.44, 0.17, 0.20, "Flatten\n$L \\cdot d_{in}$", BLUE_FILL_DARK, NUS_BLUE)
    _add_box(ax, 0.58, 0.44, 0.20, 0.20, "Gradient-Boosted\nDecision Trees", ORANGE_FILL_DARK, NUS_ORANGE)
    _add_box(ax, 0.58, 0.14, 0.20, 0.12, "Ensemble Aggregation", SOFT_GREY, LINE_GREY)
    _add_box(ax, 0.58, -0.06, 0.20, 0.10, "Scalar Prediction", BLUE_FILL, NUS_BLUE)

    _add_arrow(ax, 0.25, 0.54, 0.33, 0.54)
    _add_arrow(ax, 0.50, 0.54, 0.58, 0.54)
    _add_arrow(ax, 0.68, 0.44, 0.68, 0.26)
    _add_arrow(ax, 0.68, 0.14, 0.68, 0.04)
    return fig


def _draw_xgb_on_ax(ax):
    _prepare_subplot(ax, "XGBoost")
    _add_box(ax, 0.08, 0.40, 0.18, 0.18, "Spectrum\n$L \\times d_{in}$", BLUE_FILL, NUS_BLUE, fontsize=9.5)
    _add_box(ax, 0.34, 0.40, 0.18, 0.18, "Flatten", BLUE_FILL_DARK, NUS_BLUE, fontsize=9.5)
    _add_box(ax, 0.60, 0.40, 0.20, 0.18, "Boosted Trees", ORANGE_FILL_DARK, NUS_ORANGE, fontsize=9.5)
    _add_box(ax, 0.84, 0.40, 0.12, 0.18, "Output", BLUE_FILL, NUS_BLUE, fontsize=9.5)
    _add_arrow(ax, 0.26, 0.49, 0.34, 0.49)
    _add_arrow(ax, 0.52, 0.49, 0.60, 0.49)
    _add_arrow(ax, 0.80, 0.49, 0.84, 0.49)


def _draw_conv_transformer_detailed():
    fig, ax = plt.subplots(figsize=(15, 7.2), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "Detailed Conv-Transformer for Methodology", fontsize=18, fontweight="bold", color=NUS_BLUE, ha="left", va="top")

    _add_group_label(ax, 0.03, 0.82, "Input", NUS_BLUE)
    _add_group_label(ax, 0.22, 0.82, "Convolutional Front-End", NUS_BLUE)
    _add_group_label(ax, 0.51, 0.82, "Token Preparation", NUS_ORANGE)
    _add_group_label(ax, 0.72, 0.82, "Transformer Encoder Stack", NUS_ORANGE)
    _add_group_label(ax, 0.90, 0.82, "Prediction", NUS_BLUE)

    _add_box(ax, 0.03, 0.50, 0.14, 0.18, "Input Spectrum\n$X \\in \\mathbb{R}^{L \\times d_{in}}$", BLUE_FILL, NUS_BLUE)

    _add_box(ax, 0.22, 0.55, 0.18, 0.14, "Conv1d\n$k = K, s = 1, p = \\lfloor K/2 \\rfloor$\n$d_{in} \\rightarrow c_{conv}$", BLUE_FILL_DARK, NUS_BLUE, fontsize=10)
    _add_box(ax, 0.22, 0.30, 0.18, 0.14, "BatchNorm1d + GELU", SOFT_GREY, LINE_GREY, fontsize=10)
    _add_box(ax, 0.22, 0.05, 0.18, 0.14, "Conv1d\n$k = K, s = S, p = \\lfloor K/2 \\rfloor$\n$c_{conv} \\rightarrow d_{model}$", BLUE_FILL_DARK, NUS_BLUE, fontsize=10)

    _add_box(ax, 0.48, 0.55, 0.18, 0.14, "BatchNorm1d + GELU", SOFT_GREY, LINE_GREY, fontsize=10)
    _add_box(ax, 0.48, 0.30, 0.18, 0.14, "Token Sequence\n$Z \\in \\mathbb{R}^{L' \\times d_{model}}$\n$L' = \\lfloor (L-K)/S \\rfloor + 1$", BLUE_FILL, NUS_BLUE, fontsize=10)
    _add_box(ax, 0.48, 0.05, 0.18, 0.14, "Add Learned Token Embedding\n(optional)", ORANGE_FILL, NUS_ORANGE, fontsize=10)

    _add_box(ax, 0.72, 0.60, 0.17, 0.12, "Prepend CLS Token", ORANGE_FILL, NUS_ORANGE, fontsize=10)
    _add_box(ax, 0.72, 0.40, 0.17, 0.12, "Add Positional Encoding\n(optional)", SOFT_GREY, LINE_GREY, fontsize=10)
    _add_box(ax, 0.72, 0.18, 0.17, 0.16, "Transformer Encoder Layer × N\nLayerNorm → MHSA → FFN", ORANGE_FILL_DARK, NUS_ORANGE, fontsize=10)
    _add_box(ax, 0.72, 0.00, 0.17, 0.10, "CLS Output + LayerNorm", SOFT_GREY, LINE_GREY, fontsize=10)

    _add_box(ax, 0.91, 0.18, 0.08, 0.12, "MLP Head\nLinear → GELU\n→ Dropout → Linear", BLUE_FILL, NUS_BLUE, fontsize=9)
    _add_box(ax, 0.91, -0.02 + 0.10, 0.08, 0.10, "Prediction\n$\\hat{y}$", BLUE_FILL, NUS_BLUE, fontsize=10)

    _add_arrow(ax, 0.17, 0.59, 0.22, 0.62)
    _add_arrow(ax, 0.31, 0.55, 0.31, 0.44)
    _add_arrow(ax, 0.31, 0.30, 0.31, 0.19)
    _add_arrow(ax, 0.40, 0.12, 0.48, 0.12)
    _add_arrow(ax, 0.57, 0.30, 0.72, 0.66, color=NUS_ORANGE)
    _add_arrow(ax, 0.57, 0.12, 0.72, 0.46, color=NUS_ORANGE)
    _add_arrow(ax, 0.805, 0.60, 0.805, 0.52)
    _add_arrow(ax, 0.805, 0.40, 0.805, 0.34)
    _add_arrow(ax, 0.805, 0.18, 0.805, 0.10)
    _add_arrow(ax, 0.89, 0.05, 0.91, 0.24)
    _add_arrow(ax, 0.95, 0.18, 0.95, 0.10)

    ax.text(0.31, 0.74, "Local spectral motif extraction", fontsize=10, color=NUS_BLUE, ha="center")
    ax.text(0.57, 0.74, "Overlapping tokenization", fontsize=10, color=NUS_BLUE, ha="center")
    ax.text(0.805, 0.74, "Global self-attention over compact tokens", fontsize=10, color=NUS_ORANGE, ha="center")
    return fig


def _draw_overview_2x2():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")
    fig.text(0.03, 0.97, "Model Overview for Main Comparison", fontsize=19, fontweight="bold", color=NUS_BLUE, ha="left", va="top")
    _draw_conv_transformer_on_ax(axes[0, 0])
    _draw_transformer_on_ax(axes[0, 1])
    _draw_mlp_on_ax(axes[1, 0])
    _draw_xgb_on_ax(axes[1, 1])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def draw_model_diagram(model_name: str):
    if model_name == "conv_transformer":
        return _draw_conv_transformer()
    if model_name == "transformer":
        return _draw_transformer()
    if model_name == "mlp":
        return _draw_mlp()
    if model_name == "xgb":
        return _draw_xgb()
    if model_name == "overview_2x2":
        return _draw_overview_2x2()
    if model_name == "conv_transformer_detailed":
        return _draw_conv_transformer_detailed()
    raise ValueError(f"Unsupported model_name={model_name}")


def save_figure(fig, out_root: Path, stem: str, formats: list[str]):
    out_root.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_root / f"{stem}.{fmt}", dpi=240 if fmt == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(args):
    out_dir = ensure_dir(Path(args.output_root))
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    fig = draw_model_diagram(args.model_name)
    save_figure(fig, out_dir, f"{args.model_name}_diagram", formats)
    saved = ", ".join(str(out_dir / f"{args.model_name}_diagram.{fmt}") for fmt in formats)
    print(f"Saved model diagram to {saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw paper-ready model architecture diagrams")
    parser.add_argument(
        "--model-name",
        type=str,
        default="conv_transformer",
        choices=["conv_transformer", "transformer", "mlp", "xgb", "overview_2x2", "conv_transformer_detailed"],
    )
    parser.add_argument("--output-root", type=str, default="output/model_diagrams")
    parser.add_argument("--formats", type=str, default="png,pdf,svg")
    main(parser.parse_args())
