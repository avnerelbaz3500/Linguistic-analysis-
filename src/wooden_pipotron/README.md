# Wooden Language Analysis (Pipotron Index)

This module implements a naive strategy to quantify "Wooden Language" (*Langue de Bois*) in political discourse. 

## Methodology

The analysis is based on a "Pipotron" approach, where we count the occurrences of specific abstract keywords and "empty" administrative jargon within the documents using lemmatization method. 

### Key Sources:
- **Dictionnaire Collectif de la Langue de Bois**: Inspiration and vocabulary drawn from the [Ardeur Collective Dictionary](https://www.ardeur.net/wp-content/uploads/2015/04/dictionnaire_collectif_de_la_langue_de_bois-2.pdf).
- **Pipotron Base**: The selection of words in `wooden_data.py` reflects the classic French *Pipotron* structure, focusing on:
    - **Functional Fillers**: *Acteurs, décideurs, partenaires, dispositifs, processus.*
    - **Nominalized Abstractions**: *Optimisation, rationalisation, citoyenneté, solidarité.*
    - **Modal Adverbs/Verbs**: *Accompagne, dynamise, sécurise.*

### Calculation:
For each document, the **Wooden Score** is calculated as:

**Score = (keywords / total words) × 100**

## Components

- `wooden_data.py`: Contains the `wooden_80_90` list of keywords representing typical jargon from the 1980s-1990s.
- `pipotron_analysis.py`: The main execution script that:
    - Loads the cleaned `archelect` dataset.
    - Calculates scores for each document.
    - Aggregates results by **Year** and **Affiliate Political Party**.
    - Generates a time-series visualization highlighting the top users.

## 📊 Results

The results are stored in `logs/wooden_pipotron/`. 
The visualization highlights the **Top 3** parties utilizing these linguistic markers throughout the epochs, allowing for a temporal comparison of technocratic vs. direct rhetoric.

## How to Run

From the project root:
```bash
python -m src.wooden_pipotron.pipotron_analysis
```
