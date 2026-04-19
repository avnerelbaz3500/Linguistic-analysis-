import faiss
import numpy as np
import torch
import logging

from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer

# ----------------------------
# Logging setup
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ----------------------------
# 1. Embedding model
# ----------------------------

model = SentenceTransformer("OrdalieTech/Solon-embeddings-base-0.1")


def embed(texts: List[str]) -> np.ndarray:
    """
    Encode texts into normalized embeddings.

    Args:
        texts: List of input strings.

    Returns:
        np.ndarray: Shape (n, d) normalized embeddings.
    """
    logger.info(f"Embedding {len(texts)} texts")
    return model.encode(texts, normalize_embeddings=True)


# ----------------------------
# 2. Vector database (FAISS)
# ----------------------------

class VectorDB:
    """
    FAISS-based vector database with metadata tracking.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: embedding dimension
        """
        self.index = faiss.IndexFlatIP(dim)
        self.labels: List[str] = []
        self.vectors: Optional[np.ndarray] = None

        logger.info(f"VectorDB initialized with dim={dim}")

    def add(self, vecs: np.ndarray, labels: List[str]) -> None:
        """
        Add vectors and labels to FAISS index.

        Args:
            vecs: (n, d) embeddings
            labels: metadata labels
        """
        logger.info(f"Adding {len(vecs)} vectors to index")

        vecs = vecs.astype(np.float32)

        self.index.add(vecs)
        self.labels.extend(labels)

        if self.vectors is None:
            self.vectors = vecs
        else:
            self.vectors = np.vstack([self.vectors, vecs])

    def search(self, query_vec, k: int = 5) -> Tuple[torch.Tensor, List[str]]:
        """
        Retrieve nearest neighbors.

        Args:
            query_vec: (d,) or (1, d)
            k: number of neighbors

        Returns:
            neighbors tensor (k, d), labels list
        """
        if isinstance(query_vec, torch.Tensor):
            query_vec = query_vec.detach().cpu().numpy()

        query_vec = query_vec.astype(np.float32).reshape(1, -1)

        _, I = self.index.search(query_vec, k)

        neighbors = self.vectors[I[0]]
        labels = [self.labels[i] for i in I[0]]

        return torch.from_numpy(neighbors), labels


# ----------------------------
# 3. Split positives / negatives
# ----------------------------

def split_neighbors(
    neighbors: torch.Tensor,
    labels: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split neighbors into LB (positive) and DIRECT (negative).

    Args:
        neighbors: (k, d)
        labels: metadata labels

    Returns:
        positives, negatives tensors
    """
    pos, neg = [], []

    for vec, label in zip(neighbors, labels):
        if label == "LB":
            pos.append(vec)
        else:
            neg.append(vec)

    positives = torch.stack(pos) if pos else torch.empty(0)
    negatives = torch.stack(neg) if neg else torch.empty(0)

    return positives, negatives


# ----------------------------
# 4. InfoNCE loss
# ----------------------------

def multi_positive_nce(
    query: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Multi-positive InfoNCE loss.

    Args:
        query: (d,)
        positives: (P, d)
        negatives: (N, d)
        temperature: scaling factor

    Returns:
        scalar loss tensor
    """

    if positives.numel() == 0:
        return query.new_tensor(0.0)

    pos_sim = torch.matmul(positives, query) / temperature

    if negatives.numel() > 0:
        neg_sim = torch.matmul(negatives, query) / temperature
        all_sim = torch.cat([pos_sim, neg_sim], dim=0)
    else:
        all_sim = pos_sim

    log_num = torch.logsumexp(pos_sim, dim=0)
    log_den = torch.logsumexp(all_sim, dim=0)

    return -(log_num - log_den)


# ----------------------------
# 5. Document-level analysis
# ----------------------------

def analyze_documents(
    texts: List[str],
    doc_ids: List[str],
    db: VectorDB,
    k: int = 10
) -> List[Dict]:
    """
    Compute score per document.

    Args:
        texts: list of full documents
        doc_ids: document titles / ids
        db: vector database
        k: neighbors count

    Returns:
        list of {doc_id, score}
    """

    logger.info(f"Analyzing {len(texts)} documents")

    embeddings = embed(texts)
    embeddings = torch.tensor(embeddings)

    results = []

    for doc_id, vec in zip(doc_ids, embeddings):
        neighbors, labels = db.search(vec.unsqueeze(0), k=k)
        pos, neg = split_neighbors(neighbors, labels)

        score = multi_positive_nce(vec, pos, neg).item()

        results.append({
            "doc_id": doc_id,
            "score": score
        })

    logger.info("Analysis complete")

    return results


# ----------------------------
# 6. Example usage
# ----------------------------

if __name__ == "__main__":

    dim = 768
    db = VectorDB(dim)

    texts = [
        "Nous allons augmenter les impôts",
        "Une réflexion est engagée sur une évolution de la fiscalité",
    ] * 50

    labels = ["DIRECT", "LB"] * 50

    vecs = embed(texts)
    db.add(vecs, labels)

    test_texts = [
        "Le gouvernement entend poursuivre une réflexion globale sur la situation économique"
    ]

    test_ids = ["doc_1"]

    results = analyze_documents(test_texts, test_ids, db)

    print(results)