from typing import Dict
import pandas as pd
import numpy as np
from collections import defaultdict
import logging 
import matplotlib.pyplot as plt
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("radar")


def aggregate_profiles_by_group(
    df: pd.DataFrame,
    group_col: str
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-document profiles into group-level mean profiles.

    Args:
        df: dataframe containing a 'profile' column (dict)
        group_col: column used for grouping (e.g. party, gender, age_group)

    Returns:
        dict[group -> aggregated profile]
    """

    logger.info("Aggregating profiles by %s", group_col)

    grouped_profiles = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():

        group = row[group_col]
        profile = row["profile"]

        if not isinstance(profile, dict):
            continue

        for family, values in profile.items():
            if not isinstance(values, dict):
                continue

            for k, v in values.items():

                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = 0.0

                grouped_profiles[group][f"{family}.{k}"].append(float(v))

    # mean aggregation
    result = {}

    for group, metrics in grouped_profiles.items():
        result[group] = {
            k: float(np.mean(v))
            for k, v in metrics.items() if len(v) > 0
        }

    return result


def compute_global_mean_profile(grouped_profiles: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    all_values = defaultdict(list)

    for profile in grouped_profiles.values():
        for k, v in profile.items():
            all_values[k].append(v)

    return {k: float(np.mean(v)) for k, v in all_values.items()}


def plot_grouped_radars_with_mean(
    grouped_profiles: Dict[str, Dict[str, float]],
    output_path: str,
    title: str,
    max_groups: int = 6
) -> None:
    os.makedirs(output_path, exist_ok=True)

    global_mean = compute_global_mean_profile(grouped_profiles)

    groups = list(grouped_profiles.keys())[:max_groups]
    n = len(groups)

    fig, axes = plt.subplots(
        2, n,
        figsize=(6 * n, 10),
        subplot_kw=dict(polar=True)
    )

    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    colors = plt.cm.viridis(np.linspace(0, 1, n))

    keys = sorted(global_mean.keys())
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False)
    angles_closed = np.r_[angles, angles[0]]

    for i, (group, color) in enumerate(zip(groups, colors)):
        data = grouped_profiles[group]

        values_raw = np.array([data.get(k, 0.0) for k in keys], dtype=float)
        mean_raw = np.array([global_mean.get(k, 0.0) for k in keys], dtype=float)

        values = np.r_[values_raw, values_raw[0]]
        mean_values = np.r_[mean_raw, mean_raw[0]]

        # =========================
        # 1. RADAR NORMAL
        # =========================
        ax = axes[0, i]
        ax.plot(angles_closed, values, color=color, linewidth=2)
        ax.fill(angles_closed, values, alpha=0.20, color=color)
        ax.plot(angles_closed, mean_values, color="black", linestyle="--", linewidth=1.5)

        ax.set_xticks(angles)
        ax.set_xticklabels(keys, fontsize=8)
        ax.set_yticklabels([])
        ax.set_title(f"{group}", fontsize=12, fontweight="bold")

        # =========================
        # 2. RADAR NORMALISÉ VS MEAN
        # =========================
        ax = axes[1, i]

        delta_raw = values_raw - mean_raw
        max_abs = np.max(np.abs(delta_raw))
        if max_abs == 0:
            max_abs = 1e-6

        B_raw = (delta_raw / (2 * max_abs)) + 0.5
        A_raw = np.full_like(B_raw, 0.5)

        B = np.r_[B_raw, B_raw[0]]
        A = np.r_[A_raw, A_raw[0]]

        ax.plot(angles_closed, A, color="black", linestyle="--", linewidth=1.2, zorder=2)
        ax.fill(angles_closed, A, color="red", alpha=0.35, zorder=1)

        ax.plot(angles_closed, B, color="black", linewidth=1.4, zorder=4)
        ax.fill(angles_closed, B, color="limegreen", alpha=0.35, zorder=3)

        ax.set_ylim(0, 1)
        ax.set_xticks(angles)
        ax.set_xticklabels(keys, fontsize=8)
        ax.set_yticklabels([])
        ax.set_title(f"{group} (normalized vs mean)", fontsize=11)
        sorted_rows = sorted(
            zip(keys, values_raw, mean_raw, delta_raw),
            key=lambda x: abs(x[3]),
            reverse=True
        )

        print(f"\n=== {group} | valeurs / moyenne / delta ===")
        for k, v, m, d in sorted_rows:
            print(f"{k:>10s} : valeur={v:.4f} | moyenne={m:.4f} | delta={d:+.4f}")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    path = os.path.join(output_path, "radar_normalized_vs_mean.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    df = pd.read_parquet("data/InfoNCE/archelect_scored_NCEV2.parquet")

    grouped = aggregate_profiles_by_group(
        df,
        group_col="affiliate political party"
    )

    plot_grouped_radars_with_mean(
        grouped,
        output_path="logs/InfoNCE2/radar/",
        title="InfoNCE Profil par parti politique (vs moyenne)"
    )