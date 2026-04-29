import os
import logging
from typing import Tuple

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("infonce_stats")


# =========================================================
# DATA LOADING & PREPROCESSING
# =========================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Load scored dataset from parquet file.

    Args:
        path: path to parquet file

    Returns:
        DataFrame
    """
    logger.info("Loading data from %s", path)
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows", len(df))
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataset for analysis (year, age groups).

    Args:
        df: input dataframe

    Returns:
        processed dataframe
    """
    logger.info("Preprocessing dataset")

    df = df.copy()

    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year

    df["titulaire-age"] = pd.to_numeric(df["titulaire-age"], errors="coerce")

    bins = [18, 30, 40, 50, 60, 70, 120]
    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    df["age_group"] = pd.cut(
        df["titulaire-age"],
        bins=bins,
        labels=labels,
        right=False
    )

    return df


# =========================================================
# AGGREGATIONS
# =========================================================

def compute_party_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregated statistics per political party.

    Args:
        df: input dataframe

    Returns:
        aggregated dataframe
    """
    logger.info("Computing party aggregates")

    agg = df.groupby("affiliate political party").agg({
        "score_mean": ["mean", "std"],
        "score_max": "mean",
        "score_std": "mean"
    }).reset_index()

    agg.columns = [
        "party",
        "score_mean_avg",
        "score_mean_std",
        "score_max_avg",
        "score_std_avg"
    ]

    return agg


# =========================================================
# PLOTTING
# =========================================================

def plot_time_evolution(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot score evolution over time by party.
    """
    logger.info("Plotting time evolution")

    plt.figure(figsize=(14, 8))

    sns.lineplot(
        data=df,
        x="year",
        y="score_mean",
        hue="affiliate political party",
        marker="o",
        errorbar=None
    )

    plt.title("Evolution of InfoNCE score over time")
    plt.xlabel("Year")
    plt.ylabel("Score mean")

    plt.tight_layout()

    path = os.path.join(output_dir, "score_mean_over_time.png")
    plt.savefig(path, dpi=300)
    plt.close()

    logger.info("Saved %s", path)


def plot_party_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot score distribution by political party.
    """
    logger.info("Plotting party distribution")

    plt.figure(figsize=(12, 8))

    order = (
        df.groupby("affiliate political party")["score_mean"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    sns.boxplot(
        data=df,
        y="affiliate political party",
        x="score_mean",
        order=order
    )

    plt.title("Score distribution by political party")

    plt.tight_layout()

    path = os.path.join(output_dir, "score_boxplot_party.png")
    plt.savefig(path, dpi=300)
    plt.close()

    logger.info("Saved %s", path)


def plot_gender(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot score by gender.
    """
    logger.info("Plotting gender analysis")

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="score_mean",
        y="titulaire-sexe",
        errorbar=None
    )

    plt.title("Score by gender")

    plt.tight_layout()

    path = os.path.join(output_dir, "score_by_gender.png")
    plt.savefig(path, dpi=300)
    plt.close()

    logger.info("Saved %s", path)


def plot_age(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot score by age group.
    """
    logger.info("Plotting age analysis")

    plt.figure(figsize=(12, 6))

    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    sns.barplot(
        data=df,
        x="score_mean",
        y="age_group",
        order=labels,
        errorbar=None
    )

    plt.title("Score by age group")

    plt.tight_layout()

    path = os.path.join(output_dir, "score_by_age.png")
    plt.savefig(path, dpi=300)
    plt.close()

    logger.info("Saved %s", path)


def plot_top10(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot top 10 highest scoring documents.
    """
    logger.info("Plotting top 10")

    df = df.copy()
    df["name"] = df["titulaire-prenom"] + " " + df["titulaire-nom"]

    top10 = df.nlargest(10, "score_mean")

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=top10,
        x="score_mean",
        y="name"
    )

    plt.title("Top 10 highest scores")

    plt.tight_layout()

    path = os.path.join(output_dir, "top10.png")
    plt.savefig(path, dpi=300)
    plt.close()

    logger.info("Saved %s", path)


def plot_correlation(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot correlation matrix between score metrics.
    """
    logger.info("Plotting correlation matrix")

    corr = df[[
        "score_mean",
        "score_max",
        "score_std",
        "score_p90"
    ]].corr()

    plt.figure(figsize=(8, 6))

    sns.heatmap(corr, annot=True, cmap="coolwarm")

    plt.title("Correlation between score metrics")

    path = os.path.join(output_dir, "correlation.png")
    plt.savefig(path, dpi=300)
    plt.close()

    logger.info("Saved %s", path)


# =========================================================
# MAIN
# =========================================================

def main(data_path: str, output_dir: str) -> None:
    """
    Main pipeline for statistical analysis and visualization.

    Args:
        data_path: path to parquet dataset
        output_dir: directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_path)
    df = preprocess(df)

    agg = compute_party_aggregates(df)
    logger.info("Aggregates computed: %d rows", len(agg))

    plot_time_evolution(df, output_dir)
    plot_party_distribution(df, output_dir)
    plot_gender(df, output_dir)
    plot_age(df, output_dir)
    plot_top10(df, output_dir)
    plot_correlation(df, output_dir)

    logger.info("Analysis complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main(
        data_path="data/InfoNCE/archelect_scored_NCE.parquet",
        output_dir="logs/InfoNCE/"
    )