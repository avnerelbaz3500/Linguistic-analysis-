from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast

from .Constants.constraints import CONSTRAINTS

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("radar")

# =========================================================
# TYPES
# =========================================================

ConstraintID = str
Profile = Dict[ConstraintID, float]

# =========================================================
# GLOBAL IDS
# =========================================================

ALL_IDS: List[str] = [
    k for group in CONSTRAINTS.values() for k in group.keys()
]

# =========================================================
# EXTRACTION
# =========================================================

def extract_constraint_ids(constraints: Any) -> List[str]:
    """
    Extract constraint IDs from nested structure.

    Keeps only S*, I*, C* constraints.

    Args:
        constraints: raw constraint structure (dict/list/None)

    Returns:
        list of filtered constraint IDs
    """
    if not constraints:
        return []

    ids: List[str] = []

    if isinstance(constraints, dict):
        raw = constraints.get("id", [])
        if isinstance(raw, str):
            raw = [raw]

        for r in raw:
            if isinstance(r, str) and r and r[0] in {"S", "I", "C"}:
                ids.append(r)

        return ids

    if isinstance(constraints, list):
        for item in constraints:
            ids.extend(extract_constraint_ids(item))

    return ids

# =========================================================
# GLOBAL PROFILE
# =========================================================

def compute_global_profile(topk_results: List[Dict]) -> Profile:
    """
    Compute normalized frequency profile of constraints.

    Args:
        topk_results: list of query results with top-k groups

    Returns:
        dictionary with normalized S/I/C distributions
    """
    counter = {
        "S": defaultdict(int),
        "I": defaultdict(int),
        "C": defaultdict(int)
    }

    total = {"S": 0, "I": 0, "C": 0}

    for q in topk_results:
        for r in q["topk"]:

            constraints = r["group"].get("ldb_constraints", [])
            ids = extract_constraint_ids(constraints)

            for c in ids:
                if c.startswith("S"):
                    counter["S"][c] += 1
                    total["S"] += 1
                elif c.startswith("I"):
                    counter["I"][c] += 1
                    total["I"] += 1
                elif c.startswith("C"):
                    counter["C"][c] += 1
                    total["C"] += 1

    return {
        g: {
            k: counter[g][k] / total[g]
            for k in counter[g]
        } if total[g] else {}
        for g in ["S", "I", "C"]
    }

# =========================================================
# PLOT
# =========================================================

def plot_profiles(profile: Profile, output_path: str, title: str) -> None:
    """
    Plot radar profiles for S/I/C constraint groups.

    Args:
        profile: normalized constraint distribution
        output_path: directory to save figure
        title: plot title
    """
    os.makedirs(output_path, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        subplot_kw=dict(polar=True)
    )

    fig.patch.set_facecolor("#ffffff")
    plt.subplots_adjust(top=0.85)

    colors = {
        "S": "#4C78A8",
        "I": "#F58518",
        "C": "#54A24B"
    }

    for ax, g in zip(axes, ["S", "I", "C"]):

        data = profile.get(g, {})
        keys = sorted(data.keys())
        values = np.array([data.get(k, 0.0) for k in keys])

        if values.size == 0:
            ax.set_title(f"{g} (vide)", fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        values = np.concatenate([values, [values[0]]])

        n = len(keys)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        color = colors[g]

        ax.set_facecolor("#fbfbfb")

        ax.plot(
            angles,
            values,
            color=color,
            linewidth=2.8,
            marker="o",
            markersize=4
        )

        ax.fill(angles, values, color=color, alpha=0.18)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(keys, fontsize=9)
        ax.set_yticklabels([])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines["polar"].set_alpha(0.2)

        ax.set_title(
            f"Famille {g}",
            fontsize=13,
            fontweight="bold",
            color=color
        )

        max_idx = int(np.argmax(values[:-1]))
        ax.annotate(
            f"{values[max_idx]:.2f}",
            xy=(angles[max_idx], values[max_idx]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color=color
        )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.05)

    plt.tight_layout()

    path = os.path.join(output_path, "ldb_profiles_SIC.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved radar plot to %s", path)



# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    df = pd.read_csv("data/clean/top_k_test.csv")

    topk_results: List[Dict] = []

    for query, group in df.groupby("query"):

        topk = [
            {
                "group": {
                    "ldb_constraints": ast.literal_eval(row["ldb_constraints"])
                }
            }
            for _, row in group.iterrows()
        ]

        topk_results.append({
            "query": query,
            "topk": topk
        })

    profile = compute_global_profile(topk_results)

    plot_profiles(
        profile,
        output_path="logs/InfoNCE/radar_global/",
        title="PROFIL LANGUE DE BOIS (S / I / C)"
    )