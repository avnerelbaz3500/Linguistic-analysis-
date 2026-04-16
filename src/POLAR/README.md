# POLAR: Quantifying "Langue de Bois" in Political Discourse

This module implements a sentence-level adaptation of the **POLAR (Polarized Orthogonal Linear Axis Representation)** framework to quantify the degree of "Langue de Bois" (wooden language) in French political speeches.

## Overview

The methodology is inspired by the foundational POLAR framework by **Mathew et al. (2020)** and its recent application in **Kohler & Mill (2025)**. While the original POLAR framework was designed for word embeddings (e.g., FastText, Word2Vec), this implementation adapts the concept to **sentence embeddings** using `sentence-camembert-base`. 

## Intuition

The core intuition behind this project is to mimic the experimental design of **Kohler & Mill (2025)**, who investigated the "beauty-is-success" stereotype. In their study, they defined a "Success Axis" in a vector space (e.g., the direction from the word *failure* to *success*) and projected the word *beauty* onto it to measure cultural associations.

Applying this logic to political science, we seek to measure the "Woodenness" of a speech. However, "Langue de Bois" is not a single word; it is a **rhetorical style**. Therefore, our intuition is to:
1.  Move from a word-level space to a **high-dimensional sentence embedding space**.
2.  Define a **"Wooden Axis"** that captures the transition from semantic clarity to administrative abstraction.
3.  Project any political sentence onto this axis to quantify its rhetorical "evasiveness" regardless of its specific topic.

## Methodology

The pipeline follows three main stages:

### 1. Axis Construction & Space Definition (`axis_generation.py`)
To produce a "Langue de Bois" space, we don't just look for keywords; we define a **latent direction** in the embedding space. 

We use Large Language Models (LLMs) to act as "style translators," generating a dataset of antagonistic pairs where the semantic core remains the same, but the rhetoric shifts:
- **Direct Speech (A):** Blunt, transparent, and explicit political statements (the "Ground Truth" of the intent).
- **Wooden Language (B):** The technocratic, evasive, and abstract equivalent (the "Rhetorical Mask").

By calculating the difference between these embeddings ($\vec{E}_{ldb} - \vec{E}_{direct}$), we isolate the **rhetorical delta**. The mean of these deltas defines our POLAR vector. This vector acts as the "North Star" of the wooden language space, pointing exactly in the direction of maximum technocratic abstraction.

### 2. Sentence Projection (`scoring_pipeline.py`)
The **POLAR vector** ($V_{ldb}$) is normalized to unit length:

> **V_ldb = normalize( (1/n) * Σ( E_ldb_i - E_direct_i ) )**

Each sentence *s* from the target corpus is then embedded (**E_s**) and projected onto this axis using the dot product:

> **Score(s) = E_s · V_ldb**

This projection effectively measures how much of the sentence's vector is "aligned" with the direction of wooden language. A score approaching **+1** indicates strong alignment with technocratic abstraction, while a score approaching **-1** indicates a style closer to blunt, direct speech.

### 3. Visual Analysis (`visual_analysis.py`)
The resulting scores are aggregated by speaker, political party, and year to analyze the evolution and distribution of rhetorical styles across the French political landscape.

## Conclusion

The analysis of the French political corpus from 1981 to 1993 reveals several key insights into the use of "Langue de Bois":

1.  **Stable Temporal Baseline:** Contrary to the common perception of a rapid "technocratization" of speech, the average POLAR score remained remarkably stable between 1981 (0.118) and 1993 (0.120), suggesting that the foundations of modern political rhetoric were already well-established by the early 80s.
2.  **Partisan Disparities:** The methodology successfully distinguishes between different political "flavors" of speech. Interestingly, groups like the **Extreme Droite** and **Ecologistes** show higher alignment with our "Wooden Axis" (scores > 0.145), possibly reflecting a reliance on highly structured, abstract ideological frameworks. Conversely, the **Front National** and the **PCF** (Communist Party) exhibit lower scores, indicating a more direct or populistic rhetorical style.
3.  **Generational Shift:** A clear trend emerges across age groups: younger and mid-career politicians (30-49 years old) tend to utilize more "Langue de Bois" than their older counterparts (70+). This may reflect the professionalization of political communication among the newer generations of the era.
4.  **Gender Parity:** The data shows no significant difference in the use of wooden language between men and women (maybe a little bit more wooden language from the men), suggesting that rhetorical style in French politics is driven more by institutional and partisan norms than by gendered communication patterns.

## Limits

As with the original POLAR framework and its applications in **Kohler & Mill (2025)**, several scientific limits must be acknowledged:

1.  **Synthetic Data Sensitivity:** The definition of the "Wooden Axis" is entirely dependent on the synthetic pairs generated by the LLM. Any cultural bias or specific rhetorical assumptions held by the LLM (e.g., what constitutes "political speak") are directly encoded into the measurement axis.
2.  **Linearity Assumption:** Reducing a complex, multi-faceted rhetorical phenomenon to a single linear vector is a significant simplification. "Langue de Bois" may manifest in ways that are not captured by a single direction in a latent space.
3.  **Contextual Entanglement:** Although the goal is to isolate *style* from *content*, sentence embeddings are inherently semantic. It remains possible that certain political topics (e.g., economics vs. social issues) naturally lean towards more abstract embeddings, potentially biasing the scores regardless of the actual rhetorical intent.
4.  **Temporal Bias:** The pre-trained model used (`sentence-camembert-base`) may be more sensitive to modern French than the specific political lexicon and syntax of the 1980s and 1990s.

## References

- **Mathew, B., Ittepu, S., Saha, P., & Mukherjee, A. (2020).** *Interpretable Word Embeddings via the POLAR Framework.* Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).
- **Kohler, B., & Mill, W. (2025).** *Cultural differences in the beauty premium.* Scientific Reports, 15(1), 17632. [https://doi.org/10.1038/s41598-025-02857-4](https://doi.org/10.1038/s41598-025-02857-4)

## Usage

1. **Generate the axis:** `python src/POLAR/axis_generation.py`
2. **Project the corpus:** `python src/POLAR/scoring_pipeline.py`
3. **Visualize results:** `python src/POLAR/visual_analysis.py`
