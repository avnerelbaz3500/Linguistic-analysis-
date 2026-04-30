import numpy as np
import torch
import logging
import pandas as pd

from typing import List, Dict, Any, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from .InfoNCE import embed, load_json, group_dataset, precompute_groups
from .ProfilingNCE import extract_constraint_ids

# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger("infonce")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# =========================================================
# DEVICE / MODEL
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    "OrdalieTech/Solon-embeddings-base-0.1",
    device=device
)

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_topk_groups_per_chunk(
    chunks: List[str],
    dataset: List[Dict[str, Any]],
    k: int = 5,
    temperature: float = 0.1
) -> List[Dict[str, Any]]:

    logger.info("Embedding %d chunks", len(chunks))

    # (Q, D)
    q = torch.tensor(embed(chunks), device=device)

    # stack once → (G, *, D)
    p = torch.stack([g["p"] for g in dataset]).to(device)
    n = torch.stack([g["n"] for g in dataset]).to(device)

    # compute similarities in one shot
    # pos: (G, P, Q)
    pos = torch.einsum("gpd,qd->gpq", p, q) / temperature
    neg = torch.einsum("gnd,qd->gnq", n, q) / temperature

    # InfoNCE components
    log_num = torch.logsumexp(pos, dim=1)                 # (G, Q)
    log_den = torch.logsumexp(
        torch.cat([pos, neg], dim=1),
        dim=1
    )                                                     # (G, Q)

    losses = -(log_num - log_den)                         # (G, Q)

    # top-k per chunk (vectorized over chunks)
    top_vals, top_idx = torch.topk(
        losses,
        k,
        dim=0,
        largest=False
    )

    # transpose logic: now we iterate over chunks only for packaging
    results = [
        {
            "constraints": [dataset[i]["ldb_constraints"] for i in top_idx[:, qi].tolist()],
            "scores": top_vals[:, qi].tolist()
        }
        for qi in range(losses.shape[1])
    ]

    logger.info("Computed top-k for %d chunks", len(chunks))

    return results
# =========================================================
# DOC SUMMARY
# =========================================================

def build_doc_summary(
    chunk_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate chunk-level retrieval results into a document-level summary.

    Computes:
        - constraint frequency profile (S/I/C)
        - score statistics (mean, std, quantiles)

    Args:
        chunk_results: list of retrieval outputs per chunk

    Returns:
        dict with:
            - profile
            - score_stats
    """

    all_scores: List[float] = []

    counter = {
        "S": defaultdict(int),
        "I": defaultdict(int),
        "C": defaultdict(int),
    }

    total = {"S": 0, "I": 0, "C": 0}

    for r in chunk_results:
        all_scores.extend(r["scores"])

        for cg in r["constraints"]:
            ids = extract_constraint_ids(cg)

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

    if not all_scores:
        score_stats = {k: 0.0 for k in ["mean", "max", "min", "std", "p50", "p90", "p95"]}
    else:
        arr = np.array(all_scores)

        score_stats = {
            "mean": float(arr.mean()),
            "max": float(arr.max()),
            "min": float(arr.min()),
            "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    profile = {
        g: {
            k: counter[g][k] / total[g]
            for k in counter[g]
        } if total[g] else {}
        for g in ["S", "I", "C"]
    }

    return {
        "profile": profile,
        "score_stats": score_stats
    }

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    logger.info("Loading dataset")

    raw_data = load_json("data/InfoNCE/groupedNCEV2.json")

    groups = group_dataset(raw_data)
    dataset = precompute_groups(groups)

    datachunks = pd.read_parquet("data/clean/archelect_with_chunks.parquet")

    all_chunks: List[str] = []
    doc_ids: List[Any] = []

    for _, row in datachunks.iterrows():
        for c in row["chunks"]:
            all_chunks.append(c)
            doc_ids.append(row["id"])

    chunk_results = retrieve_topk_groups_per_chunk(
        all_chunks,
        dataset,
        k=5
    )

    doc_buffer: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)

    for i, res in enumerate(chunk_results):
        doc_buffer[doc_ids[i]].append(res)

    final_docs: List[Dict[str, Any]] = []

    for doc_id, chunk_list in doc_buffer.items():
        summary = build_doc_summary(chunk_list)
        stats = summary["score_stats"]
        profile = summary["profile"]

        final_docs.append({
            "id": doc_id,
            **{f"score_{k}": v for k, v in stats.items()},
            "profile": profile
        })

    df_summary = pd.DataFrame(final_docs)

    df_final = datachunks.merge(df_summary, on="id", how="left")

    df_final.to_parquet(
        "data/InfoNCE/archelect_scored_NCEV2.parquet",
        index=False
    )

    logger.info("Pipeline completed successfully")