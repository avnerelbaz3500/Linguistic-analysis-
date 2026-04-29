"""
Chunking pipeline using RecursiveCharacterTextSplitter.

Goal:
- Clean OCR-like political documents
- Remove headers / flyers / metadata noise (line-based filtering)
- Preserve semantic paragraphs
- Produce clean chunks for embedding + InfoNCE
"""

from __future__ import annotations

import logging
import re
from typing import List

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("chunking")

# =========================================================
# CONFIG
# =========================================================

TEXT_COL = "raw_text"
CHUNK_COL = "chunks"

MIN_LINE_LEN = 60
MIN_CHUNK_LEN = 100

# =========================================================
# SPLITTER
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=120,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
    ],
)

# =========================================================
# CLEAN NEWLINES
# =========================================================

def normalize_newlines(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# =========================================================
# UTIL: proper uppercase ratio (FIX IMPORTANT)
# =========================================================

def upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)

# =========================================================
# LINE FILTER (FIXED)
# =========================================================

def is_noise_line(line: str, min_len: int = MIN_LINE_LEN) -> bool:
    if not isinstance(line, str):
        return True

    l = " ".join(line.split())  # normalize spaces

    if len(l) < min_len:
        return True

    words = l.split()
    if len(words) < 3:
        return True

    # OCR / flyer detection via real uppercase ratio
    if upper_ratio(l) > 0.3:
        return True

    return False

# =========================================================
# PRECHUNKING
# =========================================================

def prechunk(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = normalize_newlines(text)

    lines = text.split("\n")

    kept = [
        l.strip()
        for l in lines
        if not is_noise_line(l)
    ]

    return " ".join(kept)

# =========================================================
# FIX SPLITTER EDGE EFFECTS
# =========================================================

def fix_chunk_boundaries(chunk: str) -> str:
    if not isinstance(chunk, str):
        return ""

    chunk = chunk.strip()

    # remove junk at start (OCR artifacts)
    chunk = re.sub(r"^[\.\!\?\,\;\:\)\-\s]+", "", chunk)

    return chunk

# =========================================================
# VALIDATION
# =========================================================

def is_valid_chunk(chunk: str) -> bool:
    if not isinstance(chunk, str):
        return False
    return len(chunk.strip()) >= MIN_CHUNK_LEN

# =========================================================
# PIPELINE
# =========================================================

def chunk_document(text: str) -> List[str]:

    if not isinstance(text, str) or not text.strip():
        return []

    cleaned = prechunk(text)

    raw_chunks = splitter.split_text(cleaned)

    return [
        fix_chunk_boundaries(c)
        for c in raw_chunks
        if is_valid_chunk(c)
    ]

# =========================================================
# DATAFRAME PIPELINE
# =========================================================

def add_chunks_column(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Starting chunking pipeline...")

    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    df[CHUNK_COL] = df[TEXT_COL].map(chunk_document)

    empty = (df[CHUNK_COL].map(len) == 0).sum()

    logger.info(f"Chunking complete. Empty docs: {empty}")

    return df

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    INPUT_PATH = "data/clean/archelect_clean.parquet"
    OUTPUT_PATH = "data/clean/archelect_with_chunks.parquet"

    df = pd.read_parquet(INPUT_PATH)

    df = add_chunks_column(df)

    df.to_parquet(OUTPUT_PATH, index=False)

    logger.info("Saved output")