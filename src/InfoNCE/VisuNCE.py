import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("infonce_stats")


def set_plot_style() -> None:
    """Configure global plotting style."""
    sns.set_theme(style="whitegrid")
    plt.style.use("ggplot")
    sns.set_palette("viridis")


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from parquet file."""
    logger.info("Loading data from %s", path)
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows | shape=%s", len(df), df.shape)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset (year extraction, age grouping)."""
    logger.info("Preprocessing dataset")
    df = df.copy()

    if "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    else:
        logger.warning("'date' column missing")
        df["year"] = np.nan

    if "titulaire-age" in df.columns:
        df["titulaire-age"] = pd.to_numeric(df["titulaire-age"], errors="coerce")
    else:
        logger.warning("'titulaire-age' column missing")
        df["titulaire-age"] = np.nan

    bins = [18, 30, 40, 50, 60, 70, 120]
    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    df["age_group"] = pd.cut(
        df["titulaire-age"], bins=bins, labels=labels, right=False
    )

    logger.info(
        "Year range: %s -> %s",
        df["year"].min(),
        df["year"].max(),
    )

    return df


def compute_party_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated statistics per political party."""
    required = [
        "affiliate political party",
        "score_mean",
        "score_max",
        "score_std",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    agg = (
        df.groupby("affiliate political party")
        .agg(
            score_mean_avg=("score_mean", "mean"),
            score_mean_std=("score_mean", "std"),
            score_max_avg=("score_max", "mean"),
            score_std_avg=("score_std", "mean"),
        )
        .reset_index()
        .rename(columns={"affiliate political party": "party"})
    )

    logger.info("Computed party aggregates: %d rows", len(agg))
    return agg


def _save_current_fig(output_dir: str, filename: str) -> None:
    """Save current matplotlib figure."""
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_time_evolution(df: pd.DataFrame, output_dir: str) -> None:
    """Plot score evolution over time per political party."""
    required = ["year", "score_mean", "affiliate political party"]
    if any(c not in df.columns for c in required):
        logger.warning("Skipping time evolution plot (missing columns)")
        return

    summary = (
        df.groupby(["year", "affiliate political party"], as_index=False)
        .agg(score_mean_avg=("score_mean", "mean"))
        .sort_values(["year", "affiliate political party"])
    )

    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=summary,
        x="year",
        y="score_mean_avg",
        hue="affiliate political party",
        marker="o",
        errorbar=None,
    )

    plt.title("Evolution of InfoNCE Score Over Time")
    plt.xlabel("Year")
    plt.ylabel("Score Mean")
    plt.legend(title="Political Party", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    _save_current_fig(output_dir, "score_mean_over_time.png")


def plot_party_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Plot score distribution per political party."""
    if not {"affiliate political party", "score_mean"}.issubset(df.columns):
        logger.warning("Skipping party distribution plot")
        return

    order = (
        df.groupby("affiliate political party")["score_mean"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=df,
        y="affiliate political party",
        x="score_mean",
        order=order,
        palette="magma",
    )

    plt.title("Distribution of Scores by Political Party")
    plt.tight_layout()

    _save_current_fig(output_dir, "score_boxplot_party.png")


def plot_gender(df: pd.DataFrame, output_dir: str) -> None:
    """Plot score differences by gender."""
    if not {"titulaire-sexe", "score_mean"}.issubset(df.columns):
        logger.warning("Skipping gender plot")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="score_mean",
        y="titulaire-sexe",
        errorbar=None,
    )

    plt.title("Score by Gender")
    plt.tight_layout()

    _save_current_fig(output_dir, "score_by_gender.png")


def plot_age(df: pd.DataFrame, output_dir: str) -> None:
    """Plot score differences by age group."""
    if not {"age_group", "score_mean"}.issubset(df.columns):
        logger.warning("Skipping age plot")
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="score_mean",
        y="age_group",
        errorbar=None,
    )

    plt.title("Score by Age Group")
    plt.tight_layout()

    _save_current_fig(output_dir, "score_by_age.png")


def plot_top10(df: pd.DataFrame, output_dir: str) -> None:
    """Plot top 10 individuals by average score."""
    required = [
        "titulaire-prenom",
        "titulaire-nom",
        "affiliate political party",
        "score_mean",
    ]

    if any(c not in df.columns for c in required):
        logger.warning("Skipping top10 plot")
        return

    df = df.copy()

    df["name"] = (
        df["titulaire-prenom"].fillna("").astype(str)
        + " "
        + df["titulaire-nom"].fillna("").astype(str)
    ).str.strip()

    df["party"] = df["affiliate political party"].fillna("Unknown")

    df_agg = (
        df.groupby(["name", "party"], as_index=False)
        .agg(score_mean=("score_mean", "mean"))
    )

    top10 = df_agg.nlargest(10, "score_mean")

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=top10,
        x="score_mean",
        y=top10["name"] + "\n(" + top10["party"] + ")",
    )

    plt.title("Top 10 Individuals")
    plt.tight_layout()

    _save_current_fig(output_dir, "top10.png")


def plot_correlation(df: pd.DataFrame, output_dir: str) -> None:
    """Plot correlation matrix of score metrics."""
    cols = ["score_mean", "score_max", "score_std", "score_p90"]
    if any(c not in df.columns for c in cols):
        logger.warning("Skipping correlation plot")
        return

    corr = df[cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f")

    plt.title("Correlation Matrix")
    plt.tight_layout()

    _save_current_fig(output_dir, "correlation.png")


def main(data_path: str, output_dir: str) -> None:
    """Run full analysis pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    set_plot_style()

    df = preprocess(load_data(data_path))

    agg = compute_party_aggregates(df)
    logger.info("Aggregates shape: %s", agg.shape)

    plot_time_evolution(df, output_dir)
    plot_party_distribution(df, output_dir)
    plot_gender(df, output_dir)
    plot_age(df, output_dir)
    plot_top10(df, output_dir)
    plot_correlation(df, output_dir)

    logger.info("Analysis complete → %s", output_dir)


if __name__ == "__main__":
    main(
        data_path="data/InfoNCE/archelect_scored_NCEV2.parquet",
        output_dir="logs/InfoNCE2/",
    )