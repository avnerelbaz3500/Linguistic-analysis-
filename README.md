# Linguistic-analysis-

An NLP-driven research tool designed to quantify political rhetoric, measure "Wooden Language" (*Langue de Bois*), and track the evolution of linguistic simplicity across political landscapes.

---

## 🚀 The Goal

The objective of this project is to provide a quantitative lens for sociological and political discourse analysis. Instead of analyzing *what* is said (sentiment or topics), we focus on *how* it is said. 

By measuring the structural density and complexity of speech, we can uncover patterns related to:
* **Professional Bias:** Do former lawyers speak differently than former activists?
* **Political Strategy:** Do specific parties utilize "The Populism of Simplicity" to reach broader audiences?
* **Institutional Shift:** How has political language evolved over time?

---

## 🧠 The NLP Objective

The core mission is to transform raw text into **stylometric features**. We use computational linguistics to test the hypothesis that political legitimacy is often constructed through specific ratios of abstraction and simplicity. 

By cross-referencing linguistic metrics with metadata like `titulaire-profession` (speaker's job) and `titulaire-soutien` (political support), we create a multidimensional map of modern rhetoric.

---

## ⚙️ Under the Hood

The engine operates on two primary analytical axes:

### 1. Semantic Density & "Wooden Language"
We define "Wooden Language" (*Langue de Bois*) as a high ratio of abstract filler to concrete meaning. 
* **The Metric:** We calculate the ratio of "Meaning-Carrying" words (Nouns, Verbs, Adjectives) against "Abstract/Filler" words (Functional words, nominalized abstractions, and modal adverbs).
* **Aggregation:** These scores are cross-referenced against the speaker’s background and political backing to identify trends in obfuscation vs. clarity.

### 2. The Populism of Simplicity
This module tracks linguistic complexity over time or across party lines using validated readability indices.
* **Readability Scores:** We utilize formulas like the Flesch-Kincaid Grade Level and the Gunning Fog Index:
    $$Index = 0.4 \left[ \left( \frac{\text{words}}{\text{sentences}} \right) + 100 \left( \frac{\text{complex words}}{\text{words}} \right) \right]$$
* **Temporal Analysis:** The system plots mean complexity scores against the `date`, allowing users to visualize whether political discourse is trending toward "Simple Populism" or "Technocratic Complexity."

---

## 🛠 Features

| Feature | Description |
| :--- | :--- |
| **PoS Tagging** | Deep linguistic analysis to categorize word functions. |
| **Complexity Engine** | Calculation of readability scores for every document in the corpus. |
| **Metadata Mapping** | Direct correlation between linguistic style and `titulaire-profession`. |
| **Visualization** | Automated plotting of complexity vs. time and political support. |

---