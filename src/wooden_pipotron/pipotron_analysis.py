import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

from helper_function.print import *
from src.wooden_pipotron.wooden_data import wooden_80_90

DATA_PATH = "data/clean/archelect_clean.parquet"
LOG_DIR = "logs/wooden_pipotron"
os.makedirs(LOG_DIR, exist_ok=True)

# Load NLP model once
nlp = spacy.load("fr_core_news_sm", disable=["ner", "parser"])

def load_data(path):
    """
    Load dataset and extract year from date column.

    Args:
        path (str): Path to parquet file.

    Returns:
        pd.DataFrame: Loaded dataframe with 'year' column.
    """
    if not os.path.exists(path):
        print(red(f"Error: {path} not found."))
        return None

    print(blue("[ Loading dataset ]"))
    df = pd.read_parquet(path)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    return df

def build_wooden_set(word_list):
    """
    Lemmatize a list of wooden words.

    Args:
        word_list (list): List of words.

    Returns:
        set: Lemmatized set of words.
    """
    return set(token.lemma_.lower() for word in word_list for token in nlp(word))

def compute_wooden_scores(texts, wooden_set):
    """
    Compute wooden language score using spaCy batch processing.

    Args:
        texts (Iterable[str]): Text corpus.
        wooden_set (set): Set of lemmatized wooden words.

    Returns:
        list: Wooden scores for each text.
    """
    print(blue("[ Calculating Wooden Language Ratio ]"))

    scores = []
    docs = nlp.pipe(texts, batch_size=50)

    for doc in docs:
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]

        if not lemmas:
            scores.append(0)
            continue

        count = sum(1 for lemma in lemmas if lemma in wooden_set)
        scores.append((count / len(lemmas)) * 100)

    return scores

def compute_statistics(df):
    """
    Compute aggregated statistics and rankings.

    Args:
        df (pd.DataFrame): Data with wooden_score.

    Returns:
        tuple: (stats dataframe, overall ranking, top 3 parties)
    """
    stats = (
        df.groupby(["year", "affiliate political party"])["wooden_score"]
        .mean()
        .reset_index()
    )

    overall_ranking = (
        df.groupby("affiliate political party")["wooden_score"]
        .mean()
        .sort_values(ascending=False)
    )

    top_3_parties = overall_ranking.head(3).index.tolist()

    print(blue(f"Top 3 Wooden Language Users (Overall): {top_3_parties}"))

    return stats, overall_ranking, top_3_parties

def plot_results(stats, top_3_parties):
    """
    Plot wooden language evolution over time.

    Args:
        stats (pd.DataFrame): Aggregated stats.
        top_3_parties (list): Top parties to highlight.
    """
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    for party in stats["affiliate political party"].unique():
        party_data = stats[stats["affiliate political party"] == party]

        if party in top_3_parties:
            sns.lineplot(
                data=party_data,
                x="year",
                y="wooden_score",
                label=party,
                linewidth=3,
                marker="o",
            )
        else:
            sns.lineplot(
                data=party_data,
                x="year",
                y="wooden_score",
                label=party,
                alpha=0.3,
                linestyle="--",
            )

    plt.title(
        "Evolution of 'Wooden Language' (Pipotron Index) by Political Party",
        fontsize=16,
    )
    plt.ylabel("Wooden Language Ratio (%)", fontsize=12)
    plt.xlabel("Year", fontsize=12)
    plt.legend(title="Political Party", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plot_path = os.path.join(LOG_DIR, "wooden_language_evolution.png")
    plt.savefig(plot_path)

    print(f"Analysis complete. Visualization saved to {plot_path}")

def display_top3(overall_ranking, top_3_parties):
    """
    Print summary of top 3 parties.

    Args:
        overall_ranking (pd.Series): Mean scores by party.
        top_3_parties (list): Top parties.
    """
    print(green("\n--- Top 3 Summary (Mean Scores) ---"))

    for party in top_3_parties:
        score = overall_ranking[party]
        print(f"{party: <15}: {score:.4f}%")

def run_analysis():
    """
    Main pipeline execution.
    """
    df = load_data(DATA_PATH)
    if df is None:
        return

    wooden_set = build_wooden_set(wooden_80_90)

    df["wooden_score"] = compute_wooden_scores(df["raw_text"], wooden_set)

    stats, overall_ranking, top_3_parties = compute_statistics(df)

    plot_results(stats, top_3_parties)

    display_top3(overall_ranking, top_3_parties)


if __name__ == "__main__":
    run_analysis()