from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from style import BLUE, GRID, INK, INK_SOFT, SURFACE, new_figure, save, shade

NUMERIC_FEATURES = ["index_num", "points", "n_tags", "year"]


def build_features(problems, tag_vocabulary=None, columns=None):
    features = problems[NUMERIC_FEATURES].copy()

    divisions = pd.get_dummies(problems["division"], prefix="div")
    features = pd.concat([features, divisions], axis=1)

    tag_lists = problems["tags"].fillna("").str.split("|")

    if tag_vocabulary is None:
        tag_vocabulary = sorted(
            {tag for tags in tag_lists for tag in tags if tag}
        )

    for tag in tag_vocabulary:
        features[f"tag_{tag}"] = tag_lists.apply(
            lambda tags: int(tag in tags)
        )

    if columns is not None:
        features = features.reindex(
            columns=columns,
            fill_value=0
        )

    return features, tag_vocabulary


def split_by_time(problems):
    ordered = problems.sort_values("contest_id").reset_index(drop=True)
    cutoff = int(len(ordered) * 0.8)

    return (
        ordered.iloc[:cutoff],
        ordered.iloc[cutoff:]
    )


def report(name, actual, predicted, results):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(
        f"  {name:<28} "
        f"MAE {mae:7.1f}   "
        f"RMSE {rmse:7.1f}   "
        f"R2 {r2:6.3f}"
    )

    return mae


def chart_feature_importance(model, feature_names):
    gain = pd.Series(
        model.booster_.feature_importance(
            importance_type="gain"
        ),
        index=feature_names,
    )

    share = (
        gain / gain.sum() * 100
    ).sort_values().tail(18)

    figure, axes = new_figure(
        9,
        7,
        "What the model uses to judge difficulty",
        "Share of the model's total error reduction, per feature",
    )

    bars = axes.barh(
        share.index,
        share.values,
        color=shade(share.values.tolist()),
        height=0.72
    )

    axes.bar_label(
        bars,
        fmt="%.1f%%",
        padding=6,
        color=INK,
        fontsize=9
    )

    axes.set_xlabel(
        "Share of total gain (%)",
        color=INK_SOFT,
        fontsize=10
    )

    axes.set_xlim(
        0,
        share.values.max() * 1.15
    )

    axes.grid(
        axis="x",
        color=GRID,
        linewidth=1
    )

    axes.set_axisbelow(True)

    save(
        figure,
        "05_feature_importance.png"
    )


def chart_predicted_vs_actual(actual, predicted):
    figure, axes = new_figure(
        7,
        7,
        "Predicted vs actual difficulty",
        "Each dot is one problem from the held-out test set",
    )

    axes.scatter(
        actual,
        predicted,
        s=9,
        alpha=0.25,
        color=BLUE,
        edgecolors="none"
    )

    limits = [700, 3600]

    axes.plot(
        limits,
        limits,
        color=INK_SOFT,
        linewidth=1.5,
        linestyle="--"
    )

    axes.annotate(
        "perfect prediction",
        xy=(3100, 3100),
        xytext=(0, 10),
        textcoords="offset points",
        color=INK_SOFT,
        fontsize=9,
        rotation=45,
        ha="center"
    )

    axes.set_xlim(limits)
    axes.set_ylim(limits)

    axes.set_xlabel(
        "Actual rating",
        color=INK_SOFT,
        fontsize=10
    )

    axes.set_ylabel(
        "Predicted rating",
        color=INK_SOFT,
        fontsize=10
    )

    axes.grid(
        color=GRID,
        linewidth=1
    )

    axes.set_axisbelow(True)

    save(
        figure,
        "06_predicted_vs_actual.png"
    )


def run_leakage_demo(train, test, tag_vocabulary, honest_mae):
    print("\n--- Leakage experiment (deliberately doing it wrong) ---")

    leaky_train, _ = build_features(
        train,
        tag_vocabulary
    )

    leaky_test, _ = build_features(
        test,
        tag_vocabulary,
        columns=leaky_train.columns
    )

    leaky_train["solved_count"] = train["solved_count"].values
    leaky_test["solved_count"] = test["solved_count"].values

    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        verbose=-1,
        random_state=0
    )

    model.fit(
        leaky_train,
        train["rating"]
    )

    predicted = model.predict(leaky_test)

    leaky_mae = mean_absolute_error(
        test["rating"],
        predicted
    )

    print(
        f"  {'LightGBM + solved_count':<28} "
        f"MAE {leaky_mae:7.1f}   "
        f"<-- looks great, is useless"
    )

    print(
        f"\n  This 'beats' the honest model by "
        f"{honest_mae - leaky_mae:.0f} rating points."
    )

    print(
        "  But solved_count is derived from the very thing "
        "we are predicting,"
    )

    print(
        "  and does not exist for a new problem. "
        "Excluded from the real model."
    )

    return leaky_mae


def main():
    Path("models").mkdir(exist_ok=True)

    try:
        problems = pd.read_csv(
            "data/problems.csv"
        )
    except FileNotFoundError:
        raise SystemExit(
            "No data/problems.csv found. "
            "Run: python fetch_data.py"
        )

    rated = problems.dropna(
        subset=["rating"]
    ).copy()

    train, test = split_by_time(rated)

    print(
        f"Training on {len(train):,} older problems "
        f"(contests up to #{train['contest_id'].max()})"
    )

    print(
        f"Testing on  {len(test):,} newer problems "
        f"(contests from #{test['contest_id'].min()})\n"
    )

    X_train, tag_vocabulary = build_features(
        train
    )

    X_test, _ = build_features(
        test,
        tag_vocabulary,
        columns=X_train.columns
    )

    y_train = train["rating"]
    y_test = test["rating"]

    print(
        f"{X_train.shape[1]} features "
        f"({len(tag_vocabulary)} tags + division + "
        f"{len(NUMERIC_FEATURES)} numeric)\n"
    )

    results = []

    print(
        "--- Model comparison (lower MAE is better) ---"
    )

    baseline_prediction = np.full(
        len(y_test),
        y_train.mean()
    )

    report(
        "Baseline (always mean)",
        y_test,
        baseline_prediction,
        results
    )

    linear = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression(),
    )

    linear.fit(
        X_train,
        y_train
    )

    report(
        "Linear regression",
        y_test,
        linear.predict(X_test),
        results
    )

    validation_cutoff = int(
        len(X_train) * 0.85
    )

    lgbm = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.8,
        verbose=-1,
        random_state=0
    )

    lgbm.fit(
        X_train.iloc[:validation_cutoff],
        y_train.iloc[:validation_cutoff],
        eval_X=X_train.iloc[validation_cutoff:],
        eval_y=y_train.iloc[validation_cutoff:],
        eval_metric="l1",
        callbacks=[
            early_stopping(100, verbose=False),
            log_evaluation(0)
        ],
    )

    lgbm_predictions = lgbm.predict(
        X_test
    )

    lgbm_mae = report(
        "LightGBM",
        y_test,
        lgbm_predictions,
        results
    )

    print(
        f"  (stopped after "
        f"{lgbm.best_iteration_} trees)"
    )

    baseline_mae = results[0]["MAE"]

    improvement = (
        (baseline_mae - lgbm_mae)
        / baseline_mae
        * 100
    )

    print(
        f"\nLightGBM is {improvement:.1f}% "
        f"more accurate than the baseline."
    )

    print(
        f"Average error: {lgbm_mae:.0f} rating points "
        f"(down from {baseline_mae:.0f})."
    )

    print("\nBuilding charts...")

    chart_feature_importance(
        lgbm,
        X_train.columns
    )

    chart_predicted_vs_actual(
        y_test,
        lgbm_predictions
    )

    run_leakage_demo(
        train,
        test,
        tag_vocabulary,
        lgbm_mae
    )

    joblib.dump(
        {
            "model": lgbm,
            "tags": tag_vocabulary,
            "columns": list(X_train.columns)
        },
        "models/model.pkl"
    )

    print("\nSaved models/model.pkl")

    pd.DataFrame(results).to_csv(
        "data/results.csv",
        index=False
    )

    unrated = problems[
        problems["rating"].isna()
    ].copy()

    if len(unrated):
        X_unrated, _ = build_features(
            unrated,
            tag_vocabulary,
            columns=X_train.columns
        )

        unrated["predicted_rating"] = (
            lgbm.predict(X_unrated)
        )

        columns = [
            "contest_id",
            "index",
            "name",
            "tags",
            "predicted_rating"
        ]

        (
            unrated
            .sort_values(
                "predicted_rating",
                ascending=False
            )[columns]
            .to_csv(
                "data/predicted_unrated.csv",
                index=False
            )
        )

        print(
            f"Rated {len(unrated)} previously-unrated "
            f"problems -> data/predicted_unrated.csv"
        )


if __name__ == "__main__":
    main()