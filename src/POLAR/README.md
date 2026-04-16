# POLAR: Quantifying "Langue de Bois" in Political Discourse

This module implements a sentence-level adaptation of the **POLAR (Polarized Orthogonal Linear Axis Representation)** framework to quantify the degree of "Langue de Bois" (wooden language) in French political speeches.

## Overview

The methodology is inspired by the work of **Kohler & Mill (2025)** in their study of the "beauty-is-success" stereotype and the foundational POLAR framework by **Mathew et al. (2020)**. 

While the original POLAR framework was designed for word embeddings (e.g., FastText, Word2Vec), this implementation adapts the concept to **sentence embeddings** using `sentence-camembert-base`. This adaptation is crucial because "Langue de Bois" is a rhetorical phenomenon that manifests at the sentence and discourse level rather than through isolated lexical units.

## Methodology

The pipeline follows three main stages:

### 1. Axis Construction (`axis_generation.py`)
To define a semantic direction for "Langue de Bois", we construct a synthetic dataset of antagonistic pairs. 
- **Direct Speech (A):** Blunt, transparent, and explicit political statements.
- **Wooden Language (B):** The technocratic, evasive, and abstract equivalent of the same statement.

We use Large Language Models (LLMs) to generate these pairs, ensuring they reflect the specific rhetorical style of French politics in the 1980s and 1990s.

### 2. Sentence Projection (`scoring_pipeline.py`)
We define the **POLAR vector** (V_ldb) as the normalized mean difference between the embeddings of the synthetic pairs:

> **V_ldb = normalize( (1/n) * Σ( E_ldb_i - E_direct_i ) )**

Where:
- **E_ldb_i**: Embedding of the "wooden" sentence in pair *i*.
- **E_direct_i**: Embedding of the "direct" sentence in pair *i*.
- **n**: Total number of generated pairs.

Each sentence *s* from the target corpus is then embedded (**E_s**) and projected onto this axis using the dot product:

> **Score(s) = E_s · V_ldb**

A score approaching **+1** indicates strong "Langue de Bois" characteristics, while a score approaching **-1** indicates more "Direct Speech".

### 3. Visual Analysis (`visual_analysis.py`)
The resulting scores are aggregated by speaker, political party, and year to analyze the evolution and distribution of rhetorical styles across the French political landscape.

## References

- **Mathew, B., Ittepu, S., Saha, P., & Mukherjee, A. (2020).** *Interpretable Word Embeddings via the POLAR Framework.* Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).
- **Kohler, B., & Mill, W. (2025).** *Cultural differences in the beauty premium.* Scientific Reports, 15(1), 17632. [https://doi.org/10.1038/s41598-025-02857-4](https://doi.org/10.1038/s41598-025-02857-4)

## Usage

1. **Generate the axis:** `python src/POLAR/axis_generation.py`
2. **Project the corpus:** `python src/POLAR/scoring_pipeline.py`
3. **Visualize results:** `python src/POLAR/visual_analysis.py`
