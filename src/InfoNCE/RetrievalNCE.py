import logging
from typing import List, Dict, Any, Tuple, DefaultDict

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from .InfoNCE import embed, load_json, group_dataset, precompute_groups
from .ProfilingNCE import extract_constraint_ids


logger = logging.getLogger("infonce")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


device: str = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", device)

model: SentenceTransformer = SentenceTransformer(
    "OrdalieTech/Solon-embeddings-base-0.1",
    device=device
)


def retrieve_topk_groups_per_chunk(
    chunks: List[str],
    dataset: List[Dict[str, Any]],
    k: int = 5,
    temperature: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Compute InfoNCE-based similarity between input chunks and precomputed groups,
    then retrieve the top-k closest groups per chunk.

    Args:
        chunks: List of input text chunks.
        dataset: Precomputed dataset containing:
            - "p": positive embeddings (Tensor)
            - "n": negative embeddings (Tensor)
            - "ldb_constraints": associated constraints
        k: Number of top groups to retrieve per chunk.
        temperature: Temperature scaling factor for similarity.

    Returns:
        A list (per chunk) of dicts:
            {
                "constraints": List[List[str]],
                "scores": List[float]
            }
    """

    if not chunks:
        logger.warning("Empty chunk list provided")
        return []

    logger.info("Embedding %d chunks", len(chunks))

    try:
        q: torch.Tensor = torch.tensor(embed(chunks), device=device)
    except Exception as e:
        logger.exception("Embedding failed")
        raise e

    logger.debug("Query tensor shape: %s", tuple(q.shape))

    try:
        p: torch.Tensor = torch.stack([g["p"] for g in dataset]).to(device)
        n: torch.Tensor = torch.stack([g["n"] for g in dataset]).to(device)
    except Exception as e:
        logger.exception("Failed to stack dataset tensors")
        raise e

    logger.debug("Positive tensor shape: %s", tuple(p.shape))
    logger.debug("Negative tensor shape: %s", tuple(n.shape))

    pos: torch.Tensor = torch.einsum("gpd,qd->gpq", p, q) / temperature
    neg: torch.Tensor = torch.einsum("gnd,qd->gnq", n, q) / temperature

    log_num: torch.Tensor = torch.logsumexp(pos, dim=1)
    log_den: torch.Tensor = torch.logsumexp(
        torch.cat([pos, neg], dim=1),
        dim=1
    )

    losses: torch.Tensor = -(log_num - log_den)

    logger.debug("Loss tensor shape: %s", tuple(losses.shape))

    top_vals, top_idx = torch.topk(
        losses,
        k,
        dim=0,
        largest=False
    )

    results: List[Dict[str, Any]] = [
        {
            "constraints": [
                dataset[i]["ldb_constraints"]
                for i in top_idx[:, qi].tolist()
            ],
            "scores": top_vals[:, qi].tolist()
        }
        for qi in range(losses.shape[1])
    ]

    logger.info("Computed top-%d for %d chunks", k, len(chunks))

    return results



def build_doc_summary(
    chunk_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate chunk-level retrieval outputs into document-level statistics.

    Includes:
        - Constraint frequency profile grouped by prefix (S/I/C)
        - Score distribution statistics

    Args:
        chunk_results: List of chunk-level retrieval outputs.

    Returns:
        Dictionary with:
            - "profile": normalized constraint frequencies
            - "score_stats": descriptive statistics on scores
    """

    all_scores: List[float] = []

    counter: Dict[str, DefaultDict[str, int]] = {
        "S": defaultdict(int),
        "I": defaultdict(int),
        "C": defaultdict(int),
    }

    total: Dict[str, int] = {"S": 0, "I": 0, "C": 0}

    for r in chunk_results:
        scores = r.get("scores", [])
        constraints = r.get("constraints", [])

        all_scores.extend(scores)

        for cg in constraints:
            ids: List[str] = extract_constraint_ids(cg)

            for c in ids:
                prefix = c[0] if c else None

                if prefix in counter:
                    counter[prefix][c] += 1
                    total[prefix] += 1

    if not all_scores:
        logger.warning("No scores found when building summary")
        score_stats: Dict[str, float] = {
            k: 0.0 for k in ["mean", "max", "min", "std", "p50", "p90", "p95"]
        }
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

    profile: Dict[str, Dict[str, float]] = {
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



if __name__ == "__main__":

    logger.info("Loading dataset")

    raw_data = load_json("data/InfoNCE/groupedNCEV2.json")
    groups = group_dataset(raw_data)
    dataset = precompute_groups(groups)

    logger.info("Loading chunk data")
    datachunks = pd.read_parquet("data/clean/archelect_with_chunks.parquet")

    all_chunks: List[str] = []
    doc_ids: List[Any] = []

    for _, row in datachunks.iterrows():
        chunks = row.get("chunks", [])
        doc_id = row.get("id")

        for c in chunks:
            all_chunks.append(c)
            doc_ids.append(doc_id)

    logger.info("Total chunks collected: %d", len(all_chunks))

    chunk_results = retrieve_topk_groups_per_chunk(
        all_chunks,
        dataset,
        k=5
    )

    doc_buffer: DefaultDict[Any, List[Dict[str, Any]]] = defaultdict(list)

    for i, res in enumerate(chunk_results):
        doc_buffer[doc_ids[i]].append(res)

    logger.info("Aggregating document summaries")

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

    output_path = "data/InfoNCE/archelect_scored_NCEV2.parquet"

    df_final.to_parquet(output_path, index=False)

    logger.info("Pipeline completed successfully → %s", output_path)