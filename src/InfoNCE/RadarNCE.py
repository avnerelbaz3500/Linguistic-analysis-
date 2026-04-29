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


def plot_grouped_radars(
    grouped_profiles: Dict[str, Dict[str, float]],
    output_path: str,
    title: str,
    max_groups: int = 6
) -> None:
    """
    Plot radar charts per group (party / gender / etc).

    Args:
        grouped_profiles: dict[group -> flattened profile]
        output_path: save directory
        title: global title
        max_groups: limit number of subplots
    """

    os.makedirs(output_path, exist_ok=True)

    groups = list(grouped_profiles.keys())[:max_groups]

    n = len(groups)
    fig, axes = plt.subplots(
        1, n,
        figsize=(6 * n, 6),
        subplot_kw=dict(polar=True)
    )

    if n == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0, 1, n))

    for ax, group, color in zip(axes, groups, colors):

        data = grouped_profiles[group]

        keys = sorted(data.keys())
        values = np.array([data.get(k, 0.0) for k in keys])

        if len(values) == 0:
            ax.set_title(f"{group} (vide)")
            continue

        values = np.concatenate([values, [values[0]]])

        angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, alpha=0.2, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(keys, fontsize=8)
        ax.set_yticklabels([])

        ax.set_title(str(group), fontsize=12, fontweight="bold")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    path = os.path.join(output_path, "radar_by_group.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved grouped radar plot to %s", path)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    df = pd.read_parquet("data/InfoNCE/archelect_scored_NCE.parquet")

    grouped = aggregate_profiles_by_group(
        df,
        group_col="affiliate political party"
    )

    plot_grouped_radars(
        grouped,
        output_path="logs/InfoNCE/radar/",
        title="InfoNCE Profil par parti politique"
    )