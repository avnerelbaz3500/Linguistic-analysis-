import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from helper_function.print import *


def main():
    parser = argparse.ArgumentParser(
        description="POLAR projection for political rhetoric analysis."
    )
    parser.add_argument(
        "--output-graph",
        type=str,
        default="logs/POLAR/",
        help="Path to save the output graph.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/POLAR/archelect_scored.parquet",
        help="Path to the scored parquet file.",
    )
    args = parser.parse_args()

    plt.style.use("ggplot")
    sns.set_palette("viridis")
    os.makedirs(args.output_graph, exist_ok=True)

    df = pd.read_parquet(args.data_path)

    print(blue("Generating visualization..."))
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    sns.lineplot(
        data=df,
        x="year",
        y="ldb_score",
        hue="affiliate political party",
        marker="o",
        errorbar=None,
    )

    plt.title(
        "Evolution of 'Langue de Bois' in French Political Speeches\n(Score toward +1 = Wooden Language | Score toward -1 = Direct Speech)",
        fontsize=16,
        pad=20,
    )
    plt.ylabel("POLAR Projection Score", fontsize=12)
    plt.xlabel("Year", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Political Party")
    plt.tight_layout()

    plt.savefig(os.path.join(args.output_graph, "polar_evolution.png"), dpi=300)

    print(blue("Boxplot generation..."))
    plt.figure(figsize=(12, 8))
    order = (
        df.groupby("affiliate political party")["ldb_score"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    sns.boxplot(
        data=df,
        y="affiliate political party",
        x="ldb_score",
        order=order,
        palette="magma",
        hue="affiliate political party",
    )
    plt.title(
        "Repartition of 'Langue de Bois' by political party (1981-1993)\n(Score toward +1 = Wooden Language | Score toward -1 = Direct Speech)",
        fontsize=16,
    )
    plt.xlabel("Wooden language score (POLAR)", fontsize=12)
    plt.ylabel("Political party", fontsize=12)
    plt.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_graph, "ldb_boxplot_partis.png"), dpi=300)

    print(blue("Generate density curve..."))
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="ldb_score", fill=True, color="dodgerblue", alpha=0.5)
    plt.title("Overall distribution of wooden language scores", fontsize=16)
    plt.xlabel("POLAR score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_graph, "ldb_distribution.png"), dpi=300)

    print(blue("By sex"))
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="ldb_score",
        y="titulaire-sexe",
        palette="viridis",
        hue="titulaire-sexe",
        errorbar=None,
    )
    plt.title(
        "'langue de bois' according to the gender \n(Highest scores closest to +1)",
        fontsize=16,
    )
    plt.xlabel("POLAR score", fontsize=12)
    plt.ylabel("sex", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_graph, "ldb_sex.png"), dpi=300)

    print(blue("By age"))
    df["titulaire-age"] = pd.to_numeric(df["titulaire-age"], errors="coerce")
    bins = [18, 30, 40, 50, 60, 70, 120]
    labels = [
        "18-29 ans",
        "30-39 ans",
        "40-49 ans",
        "50-59 ans",
        "60-69 ans",
        "70 ans et +",
    ]
    df["age_group"] = pd.cut(df["titulaire-age"], bins=bins, labels=labels, right=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="ldb_score",
        y="age_group",
        palette="cubehelix",
        hue="age_group",
        order=labels,
        legend=False,
        errorbar=None,
    )
    plt.title(
        "'Langue de bois' according to age group\n(Highest scores closest to +1)",
        fontsize=16,
    )
    plt.xlabel("POLAR score", fontsize=12)
    plt.ylabel("Age Group", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_graph, "ldb_age.png"), dpi=300)

    print(blue("Generating visualization by titulaire-name..."))
    if "titulaire-nom" in df.columns and "titulaire-prenom" in df.columns:
        print(blue("Top 10..."))
        df["name"] = df["titulaire-prenom"] + " " + df["titulaire-nom"]
        top_10 = df.nlargest(10, "ldb_score")

        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_10, x="ldb_score", y="name", palette="Reds_r", hue="name")
        plt.title(
            "Top 10 most wooden language speeches\n(Highest scores closest to +1)",
            fontsize=16,
        )
        plt.xlabel("POLAR score", fontsize=12)
        plt.ylabel("Candidate", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_graph, "ldb_top10.png"), dpi=300)
    else:
        print(red("Warning: Missing columns 'titulaire-nom' or 'titulaire-prenom', ignoring top ten graph."))

    print(green(f"Analysis complete. Graphs saved to {args.output_graph}"))

if __name__ == "__main__":
    main()