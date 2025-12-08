import argparse

import joblib
import numpy as np
import pandas as pd

from style import GRID, INK, INK_SOFT, new_figure, save, shade
from train import build_features

STRETCH_LOW = 100
STRETCH_HIGH = 300


def solved_problems(submissions):
    accepted = submissions[submissions["verdict"] == "OK"]
    return set(zip(accepted["contest_id"], accepted["index"]))


def current_level(submissions):
    accepted = submissions[
        (submissions["verdict"] == "OK") &
        submissions["rating"].notna()
    ].drop_duplicates(subset=["contest_id", "index"])

    if accepted.empty:
        return None, accepted

    return accepted["rating"].quantile(0.75), accepted


def tag_profile(problems, solved, level):
    band = problems[
        problems["rating"].between(level - 300, level + 100)
    ].copy()

    band["is_solved"] = [
        (c, i) in solved
        for c, i in zip(band["contest_id"], band["index"])
    ]

    exploded = (
        band.assign(
            tag=band["tags"].fillna("").str.split("|")
        )
        .explode("tag")
    )

    exploded = exploded[
        exploded["tag"].notna() &
        (exploded["tag"] != "")
    ]

    profile = exploded.groupby("tag").agg(
        in_band=("is_solved", "size"),
        solved=("is_solved", "sum"),
    )

    profile = profile[profile["in_band"] >= 25]
    profile["coverage"] = profile["solved"] / profile["in_band"]

    return profile.sort_values("coverage")


def chart_tag_profile(profile, overall_coverage, handle):
    shown = profile.sort_values("coverage", ascending=False)

    figure, axes = new_figure(
        9,
        8,
        f"Topic coverage for {handle}",
        "Share of problems at your level solved, per topic",
    )

    percentages = (shown["coverage"] * 100).tolist()

    bars = axes.barh(
        shown.index,
        percentages,
        color=shade(percentages),
        height=0.72
    )

    axes.bar_label(
        bars,
        fmt="%.0f%%",
        padding=6,
        color=INK,
        fontsize=9
    )

    axes.axvline(
        overall_coverage * 100,
        color=INK_SOFT,
        linewidth=1.5,
        linestyle="--"
    )

    axes.annotate(
        f"your average {overall_coverage * 100:.0f}%",
        xy=(overall_coverage * 100, len(shown) - 0.4),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK_SOFT,
        fontsize=9,
        va="center"
    )

    axes.set_xlabel(
        "Problems solved at your level (%)",
        color=INK_SOFT,
        fontsize=10
    )

    axes.set_xlim(
        0,
        max(max(percentages), overall_coverage * 100) * 1.25
    )

    axes.grid(axis="x", color=GRID, linewidth=1)
    axes.set_axisbelow(True)

    save(figure, "07_your_tag_profile.png")


def fill_missing_ratings(problems):
    try:
        bundle = joblib.load("models/model.pkl")
    except FileNotFoundError:
        raise SystemExit("No trained model found. Run: python train.py")

    unrated = problems["rating"].isna()

    if not unrated.any():
        return problems, 0

    features, _ = build_features(
        problems[unrated],
        bundle["tags"],
        columns=bundle["columns"]
    )

    problems.loc[unrated, "rating"] = bundle["model"].predict(features)
    problems.loc[unrated, "predicted"] = True

    return problems, int(unrated.sum())


def main():
    parser = argparse.ArgumentParser(
        description="Recommend practice problems."
    )

    parser.add_argument(
        "--handle",
        required=True,
        help="Your Codeforces handle"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="How many problems to recommend"
    )

    args = parser.parse_args()

    try:
        problems = pd.read_csv("data/problems.csv")
    except FileNotFoundError:
        raise SystemExit(
            "No data/problems.csv found. Run: python fetch_data.py"
        )

    problems["predicted"] = False

    try:
        submissions = pd.read_csv("data/submissions.csv")
    except FileNotFoundError:
        raise SystemExit(
            f"No data/submissions.csv found.\n"
            f"Run this first: python fetch_data.py --handle {args.handle}"
        )

    level, accepted = current_level(submissions)

    if level is None:
        raise SystemExit(
            f"'{args.handle}' has no solved rated problems yet."
        )

    solved = solved_problems(submissions)

    print(f"Handle:        {args.handle}")
    print(
        f"Solved:        {len(solved):,} problems "
        f"({len(accepted):,} of them rated)"
    )
    print(
        f"Your level:    {level:.0f} "
        f"(75th percentile of what you solve)"
    )
    print(
        f"Targeting:     {level + STRETCH_LOW:.0f}-"
        f"{level + STRETCH_HIGH:.0f}\n"
    )

    profile = tag_profile(problems, solved, level)

    overall_coverage = (
        profile["solved"].sum() /
        profile["in_band"].sum()
    )

    chart_tag_profile(
        profile,
        overall_coverage,
        args.handle
    )

    weak_tags = profile.head(5)

    print(
        f"--- Your weakest topics "
        f"(you average {overall_coverage * 100:.0f}% "
        f"coverage at this level) ---"
    )

    for tag, row in weak_tags.iterrows():
        print(
            f"  {tag:<28} "
            f"{row['coverage'] * 100:5.1f}%  "
            f"({row['solved']:.0f} of "
            f"{row['in_band']:.0f} solved)"
        )

    problems, filled = fill_missing_ratings(problems)

    problems["is_solved"] = [
        (c, i) in solved
        for c, i in zip(
            problems["contest_id"],
            problems["index"]
        )
    ]

    candidates = problems[
        (~problems["is_solved"]) &
        problems["rating"].between(
            level + STRETCH_LOW,
            level + STRETCH_HIGH
        )
    ].copy()

    weak_set = set(weak_tags.index)

    candidates["weak_hits"] = candidates["tags"].fillna("").apply(
        lambda tags: len(
            weak_set &
            set(tags.split("|"))
        )
    )

    candidates = candidates[
        candidates["weak_hits"] > 0
    ]

    candidates = candidates.sort_values(
        ["weak_hits", "contest_id"],
        ascending=[False, False]
    )

    print(
        f"\n--- {args.count} problems to practise next ---"
    )

    print(
        f"({filled} recent problems were rated by our own "
        f"model to get here)\n"
    )

    for _, row in candidates.head(args.count).iterrows():
        marker = "~" if row["predicted"] else " "

        url = (
            f"https://codeforces.com/problemset/problem/"
            f"{row['contest_id']}/{row['index']}"
        )

        print(
            f"  {marker}{row['rating']:.0f}  "
            f"{row['name'][:44]:<44} "
            f"{row['tags'][:34]}"
        )

        print(f"         {url}")

    print(
        "\n  ~ = difficulty estimated by our model, "
        "not yet rated by Codeforces"
    )


if __name__ == "__main__":
    main()