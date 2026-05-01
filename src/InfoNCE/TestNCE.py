import numpy as np
import logging
from typing import List, Dict, Any, Tuple

from sentence_transformers import SentenceTransformer

from .InfoNCE import load_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model = SentenceTransformer(
    "OrdalieTech/Solon-embeddings-base-0.1"
)


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Compute sentence embeddings in batches.
    """
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )


def cosine_intra_diversity(X: np.ndarray) -> float:
    """
    Mean pairwise cosine diversity (1 - similarity).
    """
    if len(X) < 2:
        return 0.0

    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = X @ X.T

    mask = ~np.eye(len(X), dtype=bool)
    return float(1.0 - np.mean(sim[mask]))


def sample_matrix(X: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Random subsample for fair comparison across datasets.
    """
    n = len(X)
    idx = np.random.choice(n, min(n_samples, n), replace=False)
    return X[idx]


def bootstrap_diversity(
    X: np.ndarray,
    n_samples: int = 200,
    n_iter: int = 20
) -> Tuple[float, float]:
    """
    Returns mean and std of intra-diversity via bootstrap.
    """
    scores = []

    for _ in range(n_iter):
        Xs = sample_matrix(X, n_samples)
        scores.append(cosine_intra_diversity(Xs))

    return float(np.mean(scores)), float(np.std(scores))


def compare(X: np.ndarray, Y: np.ndarray, name_x: str, name_y: str) -> Dict[str, float]:
    """
    Compare two distributions fairly using bootstrap-balanced sampling.
    """

    mx, sx = bootstrap_diversity(X)
    my, sy = bootstrap_diversity(Y)

    diff = mx - my

    return {
        f"{name_x}_mean_div": mx,
        f"{name_x}_std": sx,
        f"{name_y}_mean_div": my,
        f"{name_y}_std": sy,
        f"{name_x}_minus_{name_y}": diff
    }


def build_base_ldb(data: List[dict], group_size: int = 3):
    """
    Extract aligned base / LDB with group sampling.

    Assumes data is structured in repeated blocks of size `group_size`,
    and we only keep the first element of each block.
    """

    base = [
        data[i]["base"]
        for i in range(0, len(data), group_size)
        if "base" in data[i]
    ]

    ldb = [
        data[i]["ldb"]
        for i in range(0, len(data), group_size)
        if "ldb" in data[i]
    ]

    return base, ldb


if __name__ == "__main__":
    np.random.seed(0)

    logger.info("Loading datasets")

    data_para = load_json("data/InfoNCE/groupedNCE100.json")
    data_cons = load_json("data/InfoNCE/groupedNCEV2.json")
    data_no_cons = load_json("data/POLAR/pairs.json")

    base_para, ldb_para = build_base_ldb(data_para)
    base_cons, ldb_cons = build_base_ldb(data_cons)
    ldb_no_cons = [x["langue_de_bois"] for x in data_no_cons]

    logger.info("Embedding datasets")

    X_base_para = embed_texts(base_para)
    X_ldb_para = embed_texts(ldb_para)

    X_base_cons = embed_texts(base_cons)
    X_ldb_cons = embed_texts(ldb_cons)

    X_ldb_no_cons = embed_texts(ldb_no_cons)

    report = {}

    logger.info("Comparing paraphrased dataset")
    report.update(compare(X_base_para, X_ldb_para, "base_para", "ldb_para"))

    logger.info("Comparing constrained dataset")
    report.update(compare(X_base_cons, X_ldb_cons, "base_cons", "ldb_cons"))

    logger.info("Cross comparison base")
    report.update(compare(X_base_para, X_base_cons, "base_para", "base_cons"))

    logger.info("Cross comparison LDB")
    report.update(compare(X_ldb_para, X_ldb_cons, "ldb_para", "ldb_cons"))

    logger.info("No constraints baseline")
    report.update(compare(X_ldb_para, X_ldb_no_cons, "ldb_para", "ldb_no_cons"))

    print(report)