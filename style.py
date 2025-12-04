from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

Path("charts").mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"

RAMP = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b"
]


def shade(values):
    low, high = min(values), max(values)
    span = (high - low) or 1
    return [
        RAMP[int((v - low) / span * (len(RAMP) - 1))]
        for v in values
    ]


def new_figure(width, height, title, subtitle=""):
    figure, axes = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    axes.annotate(
        title,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 32),
        textcoords="offset points",
        fontsize=14,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom"
    )

    if subtitle:
        axes.annotate(
            subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 12),
            textcoords="offset points",
            fontsize=10,
            color=INK_SOFT,
            ha="left",
            va="bottom"
        )

    for side in ["top", "right"]:
        axes.spines[side].set_visible(False)

    for side in ["left", "bottom"]:
        axes.spines[side].set_color(AXIS)

    axes.tick_params(
        colors=INK_SOFT,
        labelsize=10,
        length=0
    )

    return figure, axes


def save(figure, filename):
    figure.savefig(
        f"charts/{filename}",
        dpi=150,
        facecolor=SURFACE,
        bbox_inches="tight",
        pad_inches=0.3
    )
    plt.close(figure)
    print(f"  wrote charts/{filename}")