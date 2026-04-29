GEN_PROMPT = """
Tu appartiens à {actors} en France entre 1980 et 1999.

Tu prends la parole lors de {situations}.

Tu énonces UNE phrase issue d’un discours politique sur {topics}.

RÈGLE ABSOLUE :
Réponds UNIQUEMENT en JSON valide.

----------------------------------
CONTRAINTES
----------------------------------

- une seule phrase
- maximum 30 mots
Au moins 50% des phrases un des éléments suivants :
- un nombre (date, volume, montant)
- un lieu précis (ville, région)
- une institution nommée
----------------------------------
STYLE
----------------------------------

- ton institutionnel
- vocabulaire concret
- formulation crédible pour les années 80-90
- éviter toute tournure générique répétée

----------------------------------
FORMAT STRICT
----------------------------------

{{
  "sentences": ["...", "..."]
}}
"""

PARA_PROMPT = """
Tu es un générateur de dataset NLP contrastif basé sur des transformations linguistiques contrôlées.

RÈGLE ABSOLUE :
Tu réponds UNIQUEMENT en JSON valide.
Aucun texte hors JSON.

========================================================
HIÉRARCHIE DES RÈGLES (PRIORITÉ DÉCROISSANTE)
========================================================

RÈGLE 1 (PRIORITÉ ABSOLUE) :
Tu DOIS produire une opposition claire et marquée :
- LDB = langue de bois évidente (dilution, euphémisation, abstraction)
- DIRECT = formulation directe, explicite, voire plus brutale

Si l’opposition est faible, la sortie est invalide.

RÈGLE 2 :
Tu DOIS appliquer TOUTES les contraintes associées à chaque mode.

INTERDICTION ABSOLUE :
- ne jamais reformuler une contrainte
- ne jamais inventer de contrainte
- ne jamais fusionner des contraintes

Chaque contrainte doit être recopiée exactement comme fournie.

IMPORTANT :
Les contraintes ne sont PAS binaires.
Elles doivent être appliquées avec une INTENSITÉ ajustable.

- Si possible → application forte
- Si conflit avec RÈGLE 1 → application atténuée mais visible
- Interdit : ignorer complètement une contrainte

RÈGLE 3 :
Ne jamais modifier le fait (contenu informationnel strictement identique).


========================================================
OBJECTIF
========================================================

À partir d'une phrase factuelle, générer EXACTEMENT 1 paire (LDB / DIRECT).

========================================================
PRINCIPE
========================================================

- LDB = transformation langue de bois
- DIRECT = transformation directe

========================================================
ENTRÉES
========================================================

PHRASE SOURCE :
{sentence}

CONTRAINTES :
{constraints}

========================================================
RÈGLES D’APPLICATION
========================================================

1. Générer LDB avec un style langue de bois maximal
2. Générer DIRECT avec un style frontal maximal
3. Appliquer toutes les contraintes :
   - Priorité : maximiser leur présence
   - Ajuster leur intensité si nécessaire
4. Ne jamais modifier le fait

========================================================
VALIDATION
========================================================

Invalide si :
- opposition faible entre LDB et DIRECT
- perte ou ajout d'information
- contrainte absente ou non perceptible

Valide si :
- toutes les contraintes sont présentes
- certaines contraintes sont atténuées mais détectables
- opposition forte maintenue

========================================================
FORMAT DE SORTIE STRICT
========================================================

{{
  "ldb": "...",
  "direct":"..."
}}

========================================================
RÈGLE FINALE
========================================================

JSON strict uniquement.
Aucune explication.
Aucun texte hors structure.

PHRASE SOURCE :
{sentence}
"""