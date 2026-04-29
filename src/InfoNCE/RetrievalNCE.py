import numpy as np
import torch
import logging
import pandas as pd

from typing import List, Dict, Any
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from .InfoNCE import embed, load_json, group_dataset, precompute_groups
from .ProfilingNCE import extract_constraint_ids


logger = logging.getLogger("infonce")
logging.basicConfig(level=logging.INFO)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer(
    "OrdalieTech/Solon-embeddings-base-0.1",
    device=device
)


DATASET_P = None
DATASET_N = None


def prepare_dataset(dataset: List[Dict[str, Any]]) -> None:
    """
    Precompute positive and negative embeddings for the retrieval dataset.

    Args:
        dataset: list of grouped contrastive samples
    """
    global DATASET_P, DATASET_N

    DATASET_P = torch.stack(
        [torch.as_tensor(g["p"], device=device) for g in dataset]
    )

    DATASET_N = torch.stack(
        [torch.as_tensor(g["n"], device=device) for g in dataset]
    )


def retrieve_topk_groups_per_chunk(
    chunks: List[str],
    dataset: List[Dict[str, Any]],
    k: int = 5,
    temperature: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Compute top-k most relevant constraint groups for each text chunk.

    Args:
        chunks: input text chunks
        dataset: contrastive dataset with positive and negative embeddings
        k: number of top results per chunk
        temperature: scaling factor for similarity

    Returns:
        list of top-k constraint groups per chunk
    """

    logger.info(f"Embedding {len(chunks)} chunks")

    q = torch.as_tensor(embed(chunks), device=device)

    p = DATASET_P
    n = DATASET_N

    pos = torch.einsum("gpd,qd->gpq", p, q) / temperature
    neg = torch.einsum("gnd,qd->gnq", n, q) / temperature

    log_num = torch.logsumexp(pos, dim=1)
    log_den = torch.logsumexp(torch.cat([pos, neg], dim=1), dim=1)

    losses = -(log_num - log_den)

    top_vals, top_idx = torch.topk(losses, k, dim=0, largest=False)

    results: List[Dict[str, Any]] = []

    for qi in range(losses.shape[1]):
        idx = top_idx[:, qi]
        vals = top_vals[:, qi]

        results.append({
            "constraints": [dataset[i]["ldb_constraints"] for i in idx.tolist()],
            "scores": vals.tolist()
        })

    logger.info("Top-k retrieval completed")

    return results


def build_doc_summary(
    chunk_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate chunk-level retrieval results into document-level statistics.

    Args:
        chunk_results: retrieval outputs per chunk

    Returns:
        dictionary with score statistics and constraint distribution profile
    """

    all_scores = np.array(
        [s for r in chunk_results for s in r["scores"]],
        dtype=np.float32
    )

    counter = {
        "S": defaultdict(int),
        "I": defaultdict(int),
        "C": defaultdict(int)
    }

    total = {"S": 0, "I": 0, "C": 0}

    for r in chunk_results:
        for cg in r["constraints"]:
            ids = extract_constraint_ids(cg)

            for c in ids:
                g = c[0]
                counter[g][c] += 1
                total[g] += 1

    if all_scores.size == 0:
        score_stats = {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0
        }
    else:
        score_stats = {
            "mean": float(all_scores.mean()),
            "max": float(all_scores.max()),
            "min": float(all_scores.min()),
            "std": float(all_scores.std()),
            "p50": float(np.percentile(all_scores, 50)),
            "p90": float(np.percentile(all_scores, 90)),
            "p95": float(np.percentile(all_scores, 95))
        }

    profile = {
        g: (
            {k: counter[g][k] / total[g] for k in counter[g]}
            if total[g] > 0 else {}
        )
        for g in ["S", "I", "C"]
    }

    return {
        "profile": profile,
        "score_stats": score_stats
    }


if __name__ == "__main__":

    logger.info("Loading dataset")

    raw_data = load_json("data/InfoNCE/groupedNCE100.json")

    groups = group_dataset(raw_data)
    dataset = precompute_groups(groups)

    prepare_dataset(dataset)

    datachunks = pd.read_parquet("data/clean/archelect_with_chunks.parquet")

    all_chunks: List[str] = []
    doc_ids: List[Any] = []

    for row in datachunks.itertuples(index=False):
        all_chunks.extend(row.chunks)
        doc_ids.extend([row.id] * len(row.chunks))

    chunk_results = retrieve_topk_groups_per_chunk(
        all_chunks,
        dataset,
        k=5
    )

    doc_buffer = defaultdict(list)

    for doc_id, res in zip(doc_ids, chunk_results):
        doc_buffer[doc_id].append(res)

    final_docs: List[Dict[str, Any]] = []

    for doc_id, chunk_list in doc_buffer.items():
        summary = build_doc_summary(chunk_list)

        final_docs.append({
            "id": doc_id,
            **{f"score_{k}": v for k, v in summary["score_stats"].items()},
            "profile": summary["profile"]
        })

    df_final = datachunks.merge(
        pd.DataFrame(final_docs),
        on="id",
        how="left"
    )

    df_final.to_parquet(
        "data/InfoNCE/archelect_scored_NCE.parquet",
        index=False
    )

    logger.info("Pipeline completed successfully")