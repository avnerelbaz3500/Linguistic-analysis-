# Linguistic Analysis: The Evolution of "Langue de Bois" in French Politics (1981–1993)

**Is political discourse becoming increasingly irrelevant?**  
*"Langue de bois"*, or *wooden language*, is a hallmark of political communication: a rhetorical strategy characterized by abstraction, euphemism, depersonalization, and evasive formulations that create the appearance of action while avoiding direct accountability.

This project proposes a comparative computational framework to study this phenomenon in French political discourse between **1981 and 1993**. It combines three complementary approaches:
1. a **lexical density baseline** based on technocratic vocabulary,
2. a **semantic projection framework** based on POLAR,
3. a **contrastive representation learning framework** based on InfoNCE-inspired retrieval. 

---

## Research Questions

This project is built around three core questions:

1. **Temporal evolution:** Does political discourse become more technocratic or rhetorically opaque over time?
2. **Partisan variation:** Do political families differ in their use of *langue de bois*?
3. **Methodological comparison:** Can lexical methods compete with semantic and contrastive representation-based approaches? 

---

## Methodology

The project relies on three complementary methodologies designed to capture *langue de bois* from different analytical perspectives.

### 1. Pipotron Index: Lexical Density Analysis

The first approach is a frequency-based baseline built from a curated dictionary of administrative, bureaucratic, and technocratic vocabulary.

- **Concept:** measure the proportion of jargon-related words in each speech.
- **Examples of markers:** *synergie, optimisation, dispositif, acteurs*.
- **Goal:** detect reliance on nominalized and abstract expressions typical of bureaucratic discourse.
- **Score:** a percentage representing the density of lexical markers associated with technocratic language. 

This method is simple and interpretable, but it remains limited to surface lexical overlap and cannot fully capture discourse-level rhetorical deformation. 

### 2. POLAR Projection: Semantic Axis Modeling

The second approach is an adaptation of the **POLAR (Polarized Orthogonal Linear Axis Representation)** framework.

- **Adaptation:** instead of word embeddings, the method operates at the **sentence level**, using `sentence-camembert-base`.
- **Mechanism:** a synthetic semantic axis is built from oppositional pairs contrasting **direct** and **wooden** political formulations.
- **Projection:** each sentence is projected onto this axis.
- **Score:** a continuous value from **-1 (Direct)** to **+1 (Wooden)**. 

This approach provides a more semantic and discourse-sensitive measure than a lexical count, while remaining interpretable through a single rhetorical axis. 

### 3. InfoNCE Contrastive Geometry: Retrieval-Based Discourse Modeling

The third approach models *langue de bois* through **contrastive representation learning**.

Instead of reducing political language to keywords or to a single handcrafted axis, this framework learns a latent space in which discourse is structured by controlled rhetorical deformations between **direct formulations** and **langue de bois formulations**. 

- **Synthetic contrastive data:** politically grounded statements are generated and transformed into aligned **DIRECT / LDB** pairs while preserving factual content.
- **Controlled constraints:** transformations are guided by explicit linguistic constraints covering **conceptual**, **structural**, and **informational** deformations.
- **Document representation:** real political speeches are segmented into overlapping chunks of about **450 tokens** with a stride of **120 tokens**.
- **Objective:** an **InfoNCE / NCE-inspired retrieval objective** evaluates whether a chunk aligns more strongly with direct formulations or with rhetorically constrained ones.
- **Scores:**  
  - a **max-based score** capturing extreme alignment with the most discriminative rhetorical transformation,  
  - a **profile-based score** capturing the distribution of activated constraint families across retrieved groups. 

This framework treats *langue de bois* as a **structured deformation of meaning in embedding space**, rather than as a simple lexical property. According to the report, the resulting representation space is **non-isotropic**, stable across time, and organized by political affiliation. 

---

## Main Findings

The report suggests several robust patterns.

- The learned InfoNCE space exhibits a **structured latent geometry** aligned with political groups rather than a homogeneous similarity space. 
- Some parties, such as the **Front National** and the **PCF**, tend to occupy higher regions of the aggregated InfoNCE score distribution, while **Ecologists** and **Independent** actors tend to occupy lower regions. 
- This organization remains **globally stable over time** from 1981 to 1993, suggesting that the model captures persistent discursive structure rather than transient lexical variation. 
- The profile-based analysis shows that party differences are **multidimensional** and correspond to directional deviations from a global mean profile rather than uniform scalar shifts. 

These conclusions should still be interpreted with methodological caution: the contrastive framework depends on manually designed rhetorical transformations and synthetic generation choices, even if the report argues that the main structural effects remain robust across settings. 

---

## Dataset

The project relies on a corpus of **12,746 French political documents** collected from electoral manifestos, campaign declarations, and institutional speeches spanning **1981 to 1993**. After preprocessing and metadata filtering, **12,497 documents** are retained for analysis. 

Each document is associated with structured metadata such as candidate identity, electoral context, and political affiliation. The real corpus is used as the evaluation domain, while a separate synthetic dataset is used to define contrastive rhetorical transformations. 

---

## Project Structure

```text
├── data/                       # Input datasets
├── helper_function/            # Utility helpers
├── logs/                       # Generated visualizations and analysis outputs
├── notebook/                   # Exploration notebooks
└── src/
    ├── InfoNCE/                # InfoNCE-based profiling and analysis pipeline
    │   ├── Constants/          # Prompting, constraints, and query constants
    │   │   ├── __init__.py
    │   │   ├── constraints.py
    │   │   ├── generation_context.py
    │   │   ├── prompts.py
    │   │   └── query.py
    │   ├── __init__.py
    │   ├── DataNCE.py          # Data loading / preparation for InfoNCE
    │   ├── InfoNCE.py          # Main InfoNCE scoring pipeline
    │   ├── ProfilingNCE.py     # Profile construction from InfoNCE outputs
    │   ├── RadarNCE.py         # Radar plot visualizations
    │   ├── RetrievalNCE.py     # Retrieval pipeline for comparative analysis
    │   ├── TestNCE.py          # Experimental / testing script
    │   └── VisuNCE.py          # Additional InfoNCE visualizations
    ├── POLAR/                  # POLAR projection methodology
    │   ├── axis_generation.py
    │   ├── README.md
    │   ├── scoring_pipeline.py
    │   └── visual_analysis.py
    ├── preprocessing/          # Cleaning and preprocessing pipeline
    │   ├── chunking.py
    │   └── preprocess_archelect.py
    └── wooden_pipotron/        # Lexical density methodology
```

---

## Output and Analysis

The project produces several kinds of outputs:

- lexical scores from the Pipotron baseline,
- sentence-level semantic projections from POLAR,
- document-level contrastive scores and profiles from InfoNCE,
- party-level comparisons,
- temporal evolution plots,
- radar visualizations of deviations from the global mean profile. 

The radar profile analysis is especially useful for identifying how each political group departs from the global contrastive baseline across multiple rhetorical constraint dimensions. 

---

## Interpretation

The three methods should not be seen as mutually exclusive.

- **Pipotron** provides a transparent lexical baseline.
- **POLAR** introduces a semantic projection onto a controlled rhetorical axis.
- **InfoNCE** captures discourse as a structured geometric space shaped by controlled transformations. 

Together, they form a multi-level framework for studying *langue de bois* as a phenomenon that is lexical, semantic, and geometric at once. 

---

## Reproducibility

The report states that all data, code, and experiments are publicly available in the repository, with the goal of ensuring reproducibility of preprocessing, scoring, and analysis pipelines. 

To improve reproducibility further, a useful next step would be to pin environment versions, document data access assumptions explicitly, and separate synthetic generation parameters from downstream evaluation settings. 