import os
import logging

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
# STYLE GLOBAL
# =========================================================

def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.style.use("ggplot")
    sns.set_palette("viridis")


# =========================================================
# DATA LOADING & PREPROCESSING
# =========================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Load scored dataset from parquet file.
    """
    logger.info("Loading data from %s", path)
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows", len(df))
    print(f"[load_data] shape={df.shape}")
    print(f"[load_data] columns={list(df.columns)}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataset for analysis (year, age groups).
    """
    logger.info("Preprocessing dataset")
    df = df.copy()

    if "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    else:
        df["year"] = np.nan
        print("[preprocess] WARNING: 'date' column missing -> year set to NaN")

    if "titulaire-age" in df.columns:
        df["titulaire-age"] = pd.to_numeric(df["titulaire-age"], errors="coerce")
    else:
        df["titulaire-age"] = np.nan
        print("[preprocess] WARNING: 'titulaire-age' column missing -> age_group set to NaN")

    bins = [18, 30, 40, 50, 60, 70, 120]
    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    df["age_group"] = pd.cut(
        df["titulaire-age"],
        bins=bins,
        labels=labels,
        right=False
    )

    print("[preprocess] year min/max:", df["year"].min(), df["year"].max())
    print("[preprocess] age_group counts:")
    print(df["age_group"].value_counts(dropna=False).sort_index().to_string())
    print("[preprocess] top parties:")
    if "affiliate political party" in df.columns:
        print(df["affiliate political party"].value_counts(dropna=False).head(10).to_string())
    else:
        print("Missing column: affiliate political party")

    return df


# =========================================================
# AGGREGATIONS
# =========================================================

def compute_party_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregated statistics per political party.
    """
    logger.info("Computing party aggregates")

    required = ["affiliate political party", "score_mean", "score_max", "score_std"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for party aggregates: {missing}")

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

    print("[compute_party_aggregates] preview:")
    print(agg.head(10).to_string(index=False))

    return agg


# =========================================================
# PLOTTING HELPERS
# =========================================================

def _save_current_fig(output_dir: str, filename: str) -> None:
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


# =========================================================
# PLOTTING
# =========================================================

def plot_time_evolution(df: pd.DataFrame, output_dir: str) -> None:
    logger.info("Plotting time evolution")

    print("\n=== plot_time_evolution ===")
    print(f"[plot_time_evolution] rows={len(df)}")
    print(f"[plot_time_evolution] columns present: {list(df.columns)}")

    if "year" not in df.columns:
        print("[plot_time_evolution] ERROR: missing 'year' column")
        return
    if "score_mean" not in df.columns:
        print("[plot_time_evolution] ERROR: missing 'score_mean' column")
        return
    if "affiliate political party" not in df.columns:
        print("[plot_time_evolution] ERROR: missing 'affiliate political party' column")
        return

    df_plot = df[["year", "score_mean", "affiliate political party"]].copy()

    print("[plot_time_evolution] missing values:")
    print(df_plot.isna().sum().to_string())

    print("[plot_time_evolution] year range:", df_plot["year"].min(), "->", df_plot["year"].max())

    year_counts = df_plot["year"].value_counts(dropna=False).sort_index()
    print("[plot_time_evolution] number of rows per year:")
    print(year_counts.to_string())

    party_counts = df_plot["affiliate political party"].value_counts(dropna=False)
    print("[plot_time_evolution] number of rows per party:")
    print(party_counts.to_string())

    summary = (
        df_plot.groupby(["year", "affiliate political party"], as_index=False)
        .agg(
            score_mean_avg=("score_mean", "mean"),
            n=("score_mean", "size")
        )
        .sort_values(["year", "affiliate political party"])
    )

    print("[plot_time_evolution] grouped summary preview:")
    print(summary.head(20).to_string(index=False))

    plt.figure(figsize=(14, 8))

    sns.lineplot(
        data=summary,
        x="year",
        y="score_mean_avg",
        hue="affiliate political party",
        marker="o",
        linewidth=2,
        errorbar=None
    )

    plt.title("Evolution of InfoNCE Score Over Time", fontsize=16, pad=15)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Score Mean", fontsize=12)
    plt.legend(title="Political Party", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    _save_current_fig(output_dir, "score_mean_over_time.png")

def plot_party_distribution(df: pd.DataFrame, output_dir: str) -> None:
    logger.info("Plotting party distribution")

    if "affiliate political party" not in df.columns or "score_mean" not in df.columns:
        print("[plot_party_distribution] skipped: missing columns")
        return

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

    plt.title("Distribution of Scores by Political Party", fontsize=16)
    plt.xlabel("Score Mean")
    plt.ylabel("Political Party")
    plt.axvline(x=0, color="red", linestyle="--", alpha=0.4)
    plt.tight_layout()

    _save_current_fig(output_dir, "score_boxplot_party.png")


def plot_gender(df: pd.DataFrame, output_dir: str) -> None:
    logger.info("Plotting gender analysis")

    if "titulaire-sexe" not in df.columns or "score_mean" not in df.columns:
        print("[plot_gender] skipped: missing columns")
        return

    print("[plot_gender] gender counts:")
    print(df["titulaire-sexe"].value_counts(dropna=False).to_string())

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="score_mean",
        y="titulaire-sexe",
        palette="viridis",
        errorbar=None
    )

    plt.title("Score by Gender", fontsize=16)
    plt.xlabel("Score Mean")
    plt.ylabel("Gender")
    plt.tight_layout()

    _save_current_fig(output_dir, "score_by_gender.png")


def plot_age(df: pd.DataFrame, output_dir: str) -> None:
    logger.info("Plotting age analysis")

    if "age_group" not in df.columns or "score_mean" not in df.columns:
        print("[plot_age] skipped: missing columns")
        return

    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    print("[plot_age] age_group counts:")
    print(df["age_group"].value_counts(dropna=False).sort_index().to_string())

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=df,
        x="score_mean",
        y="age_group",
        order=labels,
        palette="cubehelix",
        errorbar=None
    )

    plt.title("Score by Age Group", fontsize=16)
    plt.xlabel("Score Mean")
    plt.ylabel("Age Group")
    plt.tight_layout()

    _save_current_fig(output_dir, "score_by_age.png")


def plot_top10(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot top 10 unique individuals based on aggregated score.
    """
    logger.info("Plotting top 10 individuals")

    required = ["titulaire-prenom", "titulaire-nom", "affiliate political party", "score_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[plot_top10] skipped: missing columns {missing}")
        return

    df = df.copy()

    df["name"] = (
        df["titulaire-prenom"].fillna("").astype(str) + " " +
        df["titulaire-nom"].fillna("").astype(str)
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

    print("[plot_top10] top 10:")
    print(top10[["label", "score_mean"]].to_string(index=False))

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

    _save_current_fig(output_dir, "top10.png")


def plot_correlation(df: pd.DataFrame, output_dir: str) -> None:
    logger.info("Plotting correlation matrix")

    cols = ["score_mean", "score_max", "score_std", "score_p90"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[plot_correlation] skipped: missing columns {missing}")
        return

    corr = df[cols].corr()
    print("[plot_correlation] matrix:")
    print(corr.to_string())

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

    _save_current_fig(output_dir, "correlation.png")


# =========================================================
# MAIN
# =========================================================

def main(data_path: str, output_dir: str) -> None:
    """
    Main pipeline for statistical analysis and visualization.
    """
    os.makedirs(output_dir, exist_ok=True)

    set_plot_style()

    print(f"[main] data_path={data_path}")
    print(f"[main] output_dir={output_dir}")

    df = load_data(data_path)
    df = preprocess(df)

    print("[main] after preprocess shape:", df.shape)
    print("[main] missing values (top 15):")
    print(df.isna().sum().sort_values(ascending=False).head(15).to_string())

    agg = compute_party_aggregates(df)
    logger.info("Aggregates computed: %d rows", len(agg))
    print("[main] aggregates preview:")
    print(agg.to_string(index=False))

    plot_time_evolution(df, output_dir)
    plot_party_distribution(df, output_dir)
    plot_gender(df, output_dir)
    plot_age(df, output_dir)
    plot_top10(df, output_dir)
    plot_correlation(df, output_dir)

    logger.info("Analysis complete. Outputs in %s", output_dir)
    print("[main] analysis complete")


if __name__ == "__main__":
    main(
        data_path="data/InfoNCE/archelect_scored_NCEV2.parquet",
        output_dir="logs/InfoNCE2/"
    )