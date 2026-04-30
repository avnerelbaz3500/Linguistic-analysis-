import numpy as np
import torch
import logging
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from .Constants.query import QUERY
import json
import pandas as pd

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("infonce")

# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# MODEL
# =========================================================

model = SentenceTransformer(
    "OrdalieTech/Solon-embeddings-base-0.1",
    device=device
)

# =========================================================
# TYPES
# =========================================================

DatasetItem = Dict[str, Any]
Group = Dict[str, Any]

# =========================================================
# EMBEDDING
# =========================================================

def embed(texts: List[str]) -> np.ndarray:
    """
    Encode a list of texts into normalized embeddings.
    """
    return model.encode(texts, normalize_embeddings=True).astype(np.float32)

# =========================================================
# GROUPING
# =========================================================

def group_dataset(data: List[DatasetItem]) -> List[Group]:
    grouped: Dict[str, Group] = {}

    for item in data:
        base = item["base"]

        if base not in grouped:
            grouped[base] = {
                "base": base,
                "ldb": [],
                "direct": [],
                "ldb_constraints": [],
                "direct_constraints": []
            }

        grouped[base]["ldb"].append(item["ldb"])
        grouped[base]["direct"].append(item["direct"])
        grouped[base]["ldb_constraints"].append(item["ldb_constraints"])
        grouped[base]["direct_constraints"].append(item["direct_constraints"])

    return list(grouped.values())

# =========================================================
# PRECOMPUTE
# =========================================================

def precompute_groups(groups: List[Group]) -> List[Group]:
    processed: List[Group] = []

    for g in groups:
        p = torch.tensor(embed(g["ldb"]), device=device)
        n = torch.tensor(embed(g["direct"]), device=device)

        processed.append({
            "base": g["base"],
            "p": p,
            "n": n,
            "ldb": g["ldb"],
            "direct": g["direct"],
            "ldb_constraints": g["ldb_constraints"],
            "direct_constraints": g["direct_constraints"]
        })

    logger.info(f"Precomputed {len(processed)} groups")

    return processed

# =========================================================
# INFO NCE
# =========================================================

def info_nce(
    query: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:

    if positives.shape[0] == 0:
        return torch.tensor(float("inf"), device=query.device)

    query = query.unsqueeze(0)

    pos_logits = (positives @ query.T) / temperature
    neg_logits = (negatives @ query.T) / temperature

    logits = torch.cat([pos_logits, neg_logits], dim=0)

    log_num = torch.logsumexp(pos_logits, dim=0)
    log_den = torch.logsumexp(logits, dim=0)

    return -(log_num - log_den).squeeze()

# =========================================================
# SCORING
# =========================================================

def score_group(q_vec: torch.Tensor, group: Group) -> float:
    return info_nce(q_vec, group["p"], group["n"]).item()

# =========================================================
# RETRIEVAL SINGLE
# =========================================================

def retrieve_best_group_vectorized(
    query: str,
    dataset: List[Group]
) -> Tuple[Group, float]:

    q = torch.tensor(embed([query])[0], device=device).unsqueeze(0)

    scores = []

    for group in dataset:
        p = group["p"]
        n = group["n"]

        pos_logits = (p @ q.T) / 0.1
        neg_logits = (n @ q.T) / 0.1

        logits = torch.cat([pos_logits, neg_logits], dim=0)

        log_num = torch.logsumexp(pos_logits, dim=0)
        log_den = torch.logsumexp(logits, dim=0)

        loss = -(log_num - log_den)

        scores.append(loss)

    scores = torch.stack(scores)

    best_idx = torch.argmin(scores)

    return dataset[best_idx], scores[best_idx].item()

# =========================================================
# RETRIEVAL TOP-K
# =========================================================

def retrieve_topk_groups_per_query(
    queries: List[str],
    dataset: List[Group],
    k: int = 5,
    temperature: float = 0.1
) -> List[Dict]:

    q = torch.tensor(embed(queries), device=device)

    results: List[Dict] = []

    for qi, q_i in enumerate(q):

        q_i = q_i.unsqueeze(0)

        scores = []

        for i, group in enumerate(dataset):

            p = group["p"]
            n = group["n"]

            pos_logits = (p @ q_i.T) / temperature
            neg_logits = (n @ q_i.T) / temperature

            log_num = torch.logsumexp(pos_logits, dim=0)
            log_den = torch.logsumexp(
                torch.cat([pos_logits, neg_logits], dim=0),
                dim=0
            )

            loss = -(log_num - log_den)

            scores.append((i, loss.item()))

        scores.sort(key=lambda x: x[1])
        topk = scores[:k]

        results.append({
            "query": queries[qi],
            "topk": [
                {
                    "group": dataset[idx],
                    "score": score
                }
                for idx, score in topk
            ]
        })

    return results

# =========================================================
# IO
# =========================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    raw_data = load_json("data/InfoNCE/groupedNCE100.json")

    groups = group_dataset(raw_data)

    dataset = precompute_groups(groups)

    topk = retrieve_topk_groups_per_query(QUERY, dataset, k=5)

    rows = []

    for q_res in topk:
        query = q_res["query"]

        for rank, r in enumerate(q_res["topk"]):
            rows.append({
                "query": query,
                "rank": rank,
                "base": r["group"]["base"],
                "score": r["score"],
                "ldb": r["group"]["ldb"],
                "ldb_constraints": r["group"]["ldb_constraints"],
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/clean/top_k_test.csv", index=False)