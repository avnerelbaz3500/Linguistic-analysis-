import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from helper_function.print import *
from src.wooden_pipotron.wooden_data import wooden_80_90

DATA_PATH = "data/clean/archelect_clean.parquet"
LOG_DIR = "logs/wooden_pipotron"
os.makedirs(LOG_DIR, exist_ok=True)

def calculate_wooden_ratio(text, wooden_list):
    if not text or not isinstance(text, str):
        return 0
    words = text.split()
    if not words:
        return 0
    count = sum(1 for word in words if word in wooden_list)
    return (count / len(words)) * 100

def run_analysis():
    if not os.path.exists(DATA_PATH):
        print(red(f"Error: {DATA_PATH} not found."))
        return

    print(blue("[ Loading dataset ]"))
    df = pd.read_parquet(DATA_PATH)
    
    df['year'] = pd.to_datetime(df['date']).dt.year
    
    print(blue("[ Calculating Wooden Language Ratio ]"))
    wooden_set = set(word.lower() for word in wooden_80_90)
    df['wooden_score'] = df['raw_text'].apply(lambda x: calculate_wooden_ratio(x, wooden_set))
    
    stats = df.groupby(['year', 'affiliate political party'])['wooden_score'].mean().reset_index()

    overall_ranking = df.groupby('affiliate political party')['wooden_score'].mean().sort_values(ascending=False)
    top_3_parties = overall_ranking.head(3).index.tolist()
    print(blue(f"Top 3 Wooden Language Users (Overall): {top_3_parties}"))

    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    for party in stats['affiliate political party'].unique():
        party_data = stats[stats['affiliate political party'] == party]
        if party in top_3_parties:
            sns.lineplot(data=party_data, x='year', y='wooden_score', label=party, linewidth=3, marker='o')
        else:
            sns.lineplot(data=party_data, x='year', y='wooden_score', label=party, alpha=0.3, linestyle='--')

    plt.title("Evolution of 'Wooden Language' (Pipotron Index) by Political Party", fontsize=16)
    plt.ylabel("Wooden Language Ratio (%)", fontsize=12)
    plt.xlabel("Year", fontsize=12)
    plt.legend(title="Political Party", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(LOG_DIR, "wooden_language_evolution.png")
    plt.savefig(plot_path)
    print(f"Analysis complete. Visualization saved to {plot_path}")

    # Display Top 3 stats for logs
    print(green("\n--- Top 3 Summary (Mean Scores) ---"))
    for party in top_3_parties:
        score = overall_ranking[party]
        print(f"{party: <15}: {score:.4f}%")

if __name__ == "__main__":
    run_analysis()
