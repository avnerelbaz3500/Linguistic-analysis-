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
# STYLE GLOBAL (LE TRUC QUE TU AS OUBLIÉ)
# =========================================================

def set_plot_style():
    sns.set_theme(style="whitegrid")
    plt.style.use("ggplot")
    sns.set_palette("viridis")


# =========================================================
# PLOTTING
# =========================================================

def plot_time_evolution(df: pd.DataFrame, output_dir: str) -> None:
    logger.info("Plotting time evolution")

    plt.figure(figsize=(14, 8))

    sns.lineplot(
        data=df,
        x="year",
        y="score_mean",
        hue="affiliate political party",
        marker="o",
        linewidth=2,
        errorbar=None
    )

    plt.title(
        "Evolution of InfoNCE Score Over Time",
        fontsize=16,
        pad=15
    )
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Score Mean", fontsize=12)

    plt.legend(
        title="Political Party",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()

    path = os.path.join(output_dir, "score_mean_over_time.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_party_distribution(df: pd.DataFrame, output_dir: str) -> None:
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
        order=order,
        palette="magma"
    )

    plt.title(
        "Distribution of Scores by Political Party",
        fontsize=16
    )
    plt.xlabel("Score Mean")
    plt.ylabel("Political Party")

    plt.axvline(x=0, color="red", linestyle="--", alpha=0.4)

    plt.tight_layout()

    path = os.path.join(output_dir, "score_boxplot_party.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_gender(df: pd.DataFrame, output_dir: str) -> None:
    logger.info("Plotting gender analysis")

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="score_mean",
        y="titulaire-sexe",
        palette="viridis",
        errorbar=None
    )

    plt.title(
        "Score by Gender",
        fontsize=16
    )
    plt.xlabel("Score Mean")
    plt.ylabel("Gender")

    plt.tight_layout()

    path = os.path.join(output_dir, "score_by_gender.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_age(df: pd.DataFrame, output_dir: str) -> None:
    logger.info("Plotting age analysis")

    plt.figure(figsize=(12, 6))

    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    sns.barplot(
        data=df,
        x="score_mean",
        y="age_group",
        order=labels,
        palette="cubehelix",
        errorbar=None
    )

    plt.title(
        "Score by Age Group",
        fontsize=16
    )
    plt.xlabel("Score Mean")
    plt.ylabel("Age Group")

    plt.tight_layout()

    path = os.path.join(output_dir, "score_by_age.png")
    plt.savefig(path, dpi=300)
    plt.close()

def plot_top10(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot top 10 unique individuals based on aggregated score.

    Individuals are grouped by full name, and political party is displayed
    under each name in the y-axis labels.

    Args:
        df: Input dataframe containing scores and identity columns
        output_dir: Directory where the plot will be saved
    """

    df = df.copy()

    df["name"] = (
        df["titulaire-prenom"].fillna("") + " " + df["titulaire-nom"].fillna("")
    ).str.strip()

    df["party"] = df["affiliate political party"].fillna("Unknown")

    df_agg = (
        df.groupby(["name", "party"], as_index=False)
        .agg({"score_mean": "mean"})
    )

    df_agg["label"] = df_agg["name"] + "\n(" + df_agg["party"] + ")"

    top10 = (
        df_agg.nlargest(10, "score_mean")
        .sort_values("score_mean", ascending=False)
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=top10,
        x="score_mean",
        y="label",
        palette="Reds_r"
    )

    plt.title("Top 10 Individuals (Average InfoNCE Score)", fontsize=16)
    plt.xlabel("Score Mean")
    plt.ylabel("Candidate (Party)")

    plt.tight_layout()

    path = os.path.join(output_dir, "top10.png")
    plt.savefig(path, dpi=300)
    plt.close()

def plot_correlation(df: pd.DataFrame, output_dir: str) -> None:
    logger.info("Plotting correlation matrix")

    corr = df[[
        "score_mean",
        "score_max",
        "score_std",
        "score_p90"
    ]].corr()

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        square=True
    )

    plt.title("Correlation Between Score Metrics", fontsize=14)

    plt.tight_layout()

    path = os.path.join(output_dir, "correlation.png")
    plt.savefig(path, dpi=300)
    plt.close()

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
        data_path="data/InfoNCE/archelect_scored_NCEV2.parquet",
        output_dir="logs/InfoNCE2/"
    )