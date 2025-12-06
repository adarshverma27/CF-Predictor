import pandas as pd

from style import BLUE, GRID, INK, INK_SOFT, SURFACE, new_figure, save, shade

def chart_rating_distribution(problems):
    figure, axes = new_figure(
        9, 5,
        "Most Codeforces problems sit between 800 and 2000",
        f"Difficulty rating of {len(problems):,} rated problems",
    )
    axes.hist(
        problems["rating"],
        bins=range(800, 3600, 100),
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=1
    )
    axes.set_xlabel("Difficulty rating", color=INK_SOFT, fontsize=10)
    axes.set_ylabel("Number of problems", color=INK_SOFT, fontsize=10)
    axes.grid(axis="y", color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    save(figure, "01_rating_distribution.png")


def chart_rating_by_division(problems):
    by_division = (
        problems.groupby("division")["rating"]
        .agg(["mean", "count"])
        .sort_values("mean")
    )
    by_division = by_division[by_division["count"] >= 100]

    figure, axes = new_figure(
        9, 5,
        "Division is a strong difficulty signal",
        "Average problem rating, by contest division",
    )
    bars = axes.barh(
        by_division.index,
        by_division["mean"],
        color=shade(by_division["mean"].tolist()),
        height=0.68
    )
    axes.bar_label(bars, fmt="%.0f", padding=6, color=INK, fontsize=10)
    axes.set_xlabel("Average difficulty rating", color=INK_SOFT, fontsize=10)
    axes.set_xlim(0, by_division["mean"].max() * 1.15)
    axes.grid(axis="x", color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    save(figure, "02_rating_by_division.png")


def chart_rating_by_position(problems):
    by_position = (
        problems.groupby("index_num")["rating"]
        .agg(["mean", "count"])
    )
    by_position = by_position[by_position["count"] >= 50]

    letters = [chr(ord("A") + int(i) - 1) for i in by_position.index]

    figure, axes = new_figure(
        9, 5,
        "Problems get steadily harder through a contest",
        "Average rating by position in the contest (A = first problem)",
    )
    axes.plot(
        letters,
        by_position["mean"],
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2
    )

    for position in [0, len(letters) - 1]:
        axes.annotate(
            f"{by_position['mean'].iloc[position]:.0f}",
            (position, by_position["mean"].iloc[position]),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            color=INK,
            fontsize=10,
            fontweight="bold"
        )

    axes.set_xlabel("Problem position", color=INK_SOFT, fontsize=10)
    axes.set_ylabel("Average difficulty rating", color=INK_SOFT, fontsize=10)
    axes.grid(axis="y", color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    save(figure, "03_rating_by_position.png")


def chart_hardest_tags(problems):
    exploded = problems.assign(
        tag=problems["tags"].str.split("|")
    ).explode("tag")

    exploded = exploded[
        exploded["tag"].notna() &
        (exploded["tag"] != "")
    ]

    by_tag = (
        exploded.groupby("tag")["rating"]
        .agg(["mean", "count"])
        .query("count >= 100")
        .sort_values("mean")
    )

    figure, axes = new_figure(
        9, 10,
        "Which topics make a problem hard?",
        "Average difficulty rating per tag (tags used on 100+ problems)",
    )
    bars = axes.barh(
        by_tag.index,
        by_tag["mean"],
        color=shade(by_tag["mean"].tolist()),
        height=0.72
    )
    axes.bar_label(bars, fmt="%.0f", padding=6, color=INK, fontsize=9)
    axes.set_xlabel("Average difficulty rating", color=INK_SOFT, fontsize=10)
    axes.set_xlim(0, by_tag["mean"].max() * 1.15)
    axes.grid(axis="x", color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    save(figure, "04_hardest_tags.png")

    return by_tag


def main():
    problems = pd.read_csv("data/problems.csv")

    rated = problems.dropna(subset=["rating"])

    print(
        f"Loaded {len(problems):,} problems "
        f"({len(rated):,} rated, "
        f"{len(problems) - len(rated):,} unrated)\n"
    )

    print("Building charts...")
    chart_rating_distribution(rated)
    chart_rating_by_division(rated)
    chart_rating_by_position(rated)
    by_tag = chart_hardest_tags(rated)

    print("\n--- Hardest tags ---")
    for tag, row in by_tag.tail(5)[::-1].iterrows():
        print(
            f"  {tag:<28} {row['mean']:.0f}   "
            f"({row['count']:.0f} problems)"
        )

    print("\n--- Easiest tags ---")
    for tag, row in by_tag.head(5).iterrows():
        print(
            f"  {tag:<28} {row['mean']:.0f}   "
            f"({row['count']:.0f} problems)"
        )

    baseline_error = (
        rated["rating"] - rated["rating"].mean()
    ).abs().mean()

    print(
        f"\nAcross all rated problems, always guessing the mean "
        f"({rated['rating'].mean():.0f}) is off by "
        f"{baseline_error:.0f} rating points on average."
    )

    print(
        "train.py measures this properly on a held-out test set, "
        "so the"
    )
    print(
        "number it reports differs slightly -- that one is the real bar."
    )


if __name__ == "__main__":
    main()