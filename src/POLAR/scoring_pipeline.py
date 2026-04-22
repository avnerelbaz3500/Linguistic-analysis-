import argparse
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import spacy
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from helper_function.print import *

def get_device() -> torch.device:
    """Returns the optimal device (MPS for Apple Silicon, CUDA for Nvidia, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def mean_pooling(model_output: tuple, attention_mask: torch.Tensor) -> torch.Tensor:
    """Performs mean pooling on token embeddings to get a single sentence vector."""
    token_embeddings = model_output[0]  # (batch_size, seq_length, hidden_size)
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # avoid 0 division
    return sum_embeddings / sum_mask

def embed_sentences(
    sentences: list, tokenizer, model, device, batch_size=32
) -> torch.Tensor:
    """Generates normalized embeddings for a list of sentences."""
    model.eval()
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        encoded_input = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(
            device
        )  # {(input_ids: (batch_size, sequence_length), attention_mask:(batch_size, sequence_length)}

        with torch.no_grad():
            model_output = model(**encoded_input)

        embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )  # (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size)
        embeddings = torch.nn.functional.normalize(embeddings)
        all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)  # (nb of sentences, hidden_size)

def build_polar_axis(pairs_path: Path, tokenizer, model, device) -> torch.Tensor:
    """Constructs the POLAR axis representing the 'Langue de bois' direction."""
    with open(pairs_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    direct_texts = [item["direct"] for item in dataset if item.get("direct")]
    ldb_texts = [
        item["langue_de_bois"] for item in dataset if item.get("langue_de_bois")
    ]

    if len(direct_texts) != len(ldb_texts) or len(direct_texts) == 0:
        raise ValueError(red("Invalid pairs dataset format or empty dataset."))

    print("Embedding base dataset to construct POLAR axis...")
    embed_direct = embed_sentences(
        direct_texts, tokenizer, model, device
    )  # (num_sentences, hidden_size)
    embed_ldb = embed_sentences(
        ldb_texts, tokenizer, model, device
    )  # (num_sentences, hidden_size)

    polar_vector = torch.mean(
        embed_ldb - embed_direct, dim=0
    )  # mean over the number of sentences
    polar_vector = torch.nn.functional.normalize(polar_vector, p=2, dim=0)

    return polar_vector.to(device)

def compute_polar_scores_bulk(
    sentences: list, tokenizer, model, device, polar_vector, batch_size=32
) -> np.ndarray:
    """Calculate the dot product directky on GPU :-> array 1D of scores."""
    model.eval()
    all_scores = []

    for i in tqdm(
        range(0, len(sentences), batch_size), desc="Computing sentence scores"
    ):
        batch = sentences[i : i + batch_size]
        encoded_input = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(device)

        with torch.no_grad():
            model_output = model(
                **encoded_input
            )  # model_output[0] : (batch_size, seq_length, hidden_size)

        embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )  # (batch_size, hidden_size)
        embeddings = torch.nn.functional.normalize(embeddings)

        scores = torch.matmul(
            embeddings, polar_vector
        )  # (batch_size, hidden_size) @ (hidden_size) -> (batch_size)
        all_scores.extend(scores.cpu().numpy())

    return np.array(all_scores)


def main():
    parser = argparse.ArgumentParser(
        description="POLAR projection for political rhetoric analysis."
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="data/POLAR/pairs.json",
        help="Path to the generated pairs.json file.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/clean/archelect_clean.parquet",
        help="Path to the archelect_clean parquet file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dangvantuan/sentence-camembert-base",
        help="HuggingFace model for embeddings.",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="data/POLAR/",
        help="Path to save the output graph.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_data, exist_ok=True)
    device = get_device()
    print(bold(f"Using device: {device}"))

    print(blue(f"Loading tokenizer and model: {args.model}"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    polar_vector = build_polar_axis(Path(args.pairs), tokenizer, model, device)
    print(green("POLAR axis successfully constructed."))

    print(f"Loading corpus from {args.data}")
    df = pd.read_parquet(args.data)

    if (
        "date" not in df.columns
        or "raw_text" not in df.columns
        or "affiliate political party" not in df.columns
    ):
        raise ValueError(
            red(
                "The parquet file must contain 'date', 'raw_text', and 'affiliate political party' columns."
            )
        )

    df["year"] = pd.to_datetime(df["date"]).dt.year

    print(blue("Tokenizing and flattening corpus..."))
    nlp = spacy.blank("fr")
    nlp.add_pipe("sentencizer")
    doc_indices = []
    all_sentences = []

    texts = df["raw_text"].fillna("").tolist()

    for idx, doc in tqdm(
        enumerate(nlp.pipe(texts, batch_size=300)),
        total=len(texts),
        desc="Tokenisation",
    ):
        if doc.text.strip():
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            all_sentences.extend(sentences)
            doc_indices.extend([idx] * len(sentences))  # Store indices
    print(blue(f"Total sentences to process globally: {len(all_sentences)}"))

    sentence_scores = compute_polar_scores_bulk(
        all_sentences, tokenizer, model, device, polar_vector, args.batch_size
    )

    print(blue("Aggregating scores per document..."))
    score_df = pd.DataFrame({"doc_idx": doc_indices, "score": sentence_scores})
    mean_scores = score_df.groupby("doc_idx")["score"].mean()

    df["ldb_score"] = np.nan
    df.loc[mean_scores.index, "ldb_score"] = mean_scores.values
    df = df.dropna(subset=["ldb_score"])

    output_df_path = os.path.join(args.output_data, "archelect_scored.parquet")
    print(blue(f"Saving new data_base with POLAR in : {output_df_path}"))
    df.to_parquet(output_df_path, index=False)

if __name__ == "__main__":
    main()
