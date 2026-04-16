# Linguistic Analysis: The Evolution of "Langue de Bois" in French Politics (1981–1993)

**Is political discourse becoming increasingly irrelevant?**  
*"Langue de bois"*, or "wooden language", is a hallmark of political communication: a rhetorical strategy characterized by technocratic abstraction, euphemisms, and evasive structures designed to convey a sense of action while avoiding concrete accountability. This project provides a comparative scientific framework to quantify this phenomenon within the French political landscape during the pivotal years of 1981 to 1993.

## The Problem: Quantifying the Elusive

While "Langue de bois" is intuitively recognizable, its systematic quantification poses a significant challenge for computational linguistics. It is not merely the presence of specific words, but a complex interplay of syntax, abstraction, and semantic evasion. 

Our objective is to determine:
1. **Temporal Evolution:** Has political speech become more technocratic over time?
2. **Partisan Variation:** Do different political ideologies employ "Langue de bois" with varying intensity or for different strategic purposes?
3. **Methodological Comparison:** Can a simple lexical count compete with modern vector-space projections?

---

## Methodology

This project employs two distinct, complementary methodologies to capture the essence of wooden language from both a lexical and a semantic perspective.

### 1. The Pipotron Index: Lexical Density Analysis
The first approach is a frequency-based analysis utilizing a curated dictionary of administrative and technocratic jargon—the **"Pipotron"** list. 

- **Concept:** We measure the "density" of jargon by calculating the ratio of specific keywords (e.g., *synergie, optimisation, dispositif, acteurs*) against the total word count of a speech.
- **Goal:** To identify the reliance on "empty" nominalized abstractions that characterize the French administrative style of the 80s and 90s.
- **Score:** A percentage indicating the concentration of technocratic markers.

### 2. POLAR Projection: Semantic Vector-Space Analysis
The second approach is a sophisticated adaptation of the **POLAR (Polarized Orthogonal Linear Axis Representation)** framework, originally proposed by **Mathew et al. (2020)** and recently applied in **Kohler & Mill (2025)**.

- **Adaptation:** Unlike traditional POLAR which works on word embeddings, we operate at the **sentence level** using `sentence-camembert-base`. This is crucial because "Langue de bois" is a discourse-level phenomenon.
- **Mechanism:** We construct a synthetic semantic axis by generating antagonistic pairs (e.g., *"We will fire 500 people"* vs. *"A plan for the adaptation of resources is being deployed to safeguard competitiveness"*). 
- **Projection:** Every sentence in our corpus is projected onto this **Direct-to-Wooden axis**. 
- **Score:** A continuous value from **-1 (Direct)** to **+1 (Wooden)**, providing a nuanced measure of rhetorical abstraction.

---

## Project Structure

```text
├── data/
│   ├── clean/            # Preprocessed political speeches (Archelect corpus)
│   └── POLAR/            # Results and synthetic axis data
├── logs/                 # Visualizations and statistical reports
└── src/
    ├── POLAR/            # Implementation of the projection methodology
    ├── wooden_pipotron/  # Implementation of the lexical density methodology
    └── preprocessing/    # Data cleaning and tokenization pipelines
```

## References

- **Kohler, B., & Mill, W. (2025).** *Cultural differences in the beauty premium.* Scientific Reports, 15(1), 17632. [https://doi.org/10.1038/s41598-025-02857-4](https://doi.org/10.1038/s41598-025-02857-4)
- **Mathew, B., Ittepu, S., Saha, P., & Mukherjee, A. (2020).** *Interpretable Word Embeddings via the POLAR Framework.* Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).
- **Ardeur Collective.** *Dictionnaire Collectif de la Langue de Bois.* [ardeur.net](https://www.ardeur.net/wp-content/uploads/2015/04/dictionnaire_collectif_de_la_langue_de_bois-2.pdf)

---

## Getting Started

To reproduce the analysis:
1. **Preprocess the data:** `python src/preprocessing/preprocess_archelect.py`
2. **Run Lexical Analysis:** `python -m src.wooden_pipotron.pipotron_analysis`
3. **Run POLAR Pipeline:** 
   - `python src/POLAR/axis_generation.py`
   - `python src/POLAR/scoring_pipeline.py`
   - `python src/POLAR/visual_analysis.py`
