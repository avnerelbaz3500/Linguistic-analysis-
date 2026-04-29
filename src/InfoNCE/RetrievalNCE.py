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
    """
    Compute top-k retrieval results for each chunk against a dataset of contrastive groups.

    Args:
        chunks: list of text chunks
        dataset: list of groups containing positive/negative embeddings and metadata
        k: number of top results to keep per chunk
        temperature: softmax temperature scaling

    Returns:
        List of dicts with:
            - constraints: ldb constraint structures
            - scores: retrieval losses
    """

    logger.info("Embedding %d chunks", len(chunks))

    q = torch.tensor(embed(chunks), device=device)

    results: List[Dict[str, Any]] = []

    for qi, q_i in enumerate(q):
        q_i = q_i.unsqueeze(0)

        scores: List[Tuple[int, float]] = []

        for i, group in enumerate(dataset):
            p = group["p"]
            n = group["n"]

            pos = (p @ q_i.T) / temperature
            neg = (n @ q_i.T) / temperature

            log_num = torch.logsumexp(pos, dim=0)
            log_den = torch.logsumexp(torch.cat([pos, neg], dim=0), dim=0)

            loss = -(log_num - log_den)
            scores.append((i, loss.item()))

        scores.sort(key=lambda x: x[1])
        topk = scores[:k]

        results.append({
            "constraints": [dataset[idx]["ldb_constraints"] for idx, _ in topk],
            "scores": [s for _, s in topk],
        })

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

    raw_data = load_json("data/InfoNCE/groupedNCE100.json")

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
        "data/InfoNCE/archelect_scored_NCE.parquet",
        index=False
    )

    logger.info("Pipeline completed successfully")