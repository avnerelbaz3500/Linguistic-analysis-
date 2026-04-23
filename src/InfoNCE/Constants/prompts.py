GEN_PROMPT = """
Tu es un générateur de phrases politiques françaises (années 80-90).

RÈGLE ABSOLUE :
Réponds UNIQUEMENT en JSON valide.
Aucun texte avant ou après.

OBJECTIF :
Générer EXACTEMENT {n} phrases factuelles.

----------------------------------
CONTRAINTES GÉNÉRALES
----------------------------------

- France (années 80-90)
- style institutionnel
- phrases courtes (max 20 mots)
- une seule idée par phrase
- uniquement faits ou décisions
- aucune opinion

----------------------------------
INTERDICTIONS
----------------------------------

- pas de langue de bois
- pas de flou ("mesures", "dispositif", etc.)
- pas d’abstraction inutile
- pas de justification
- pas de généralisation vague

----------------------------------
EXIGENCES LINGUISTIQUES
----------------------------------

- sujet explicite (acteur identifié)
- verbe simple et neutre (éviter les verbes idéologiquement marqués)
- objet précis
- vocabulaire concret

----------------------------------
DIVERSITÉ POLITIQUE (CRITIQUE)
----------------------------------

Les phrases doivent couvrir différentes thématiques réelles des années 80-90 :

 - chômage et emploi de masse
 - précarité du travail et montée des contrats atypiques
 - désindustrialisation et déclin des bassins industriels
 - mondialisation économique et délocalisations
 - privatisations des entreprises publiques
 - nationalisations et recompositions de l’État économique
 - rôle de l’État-providence et réforme de l’État
 - protection sociale (retraites, assurance maladie, prestations sociales)
 - fiscalité et pression fiscale (CSG, impôts, financement social)
 - déficit public et dette de l’État
 - immigration et politiques d’intégration
 - contrôle des frontières et droit d’asile
 - identité nationale et débats sur l’intégration
 - montée du Front national et tensions politiques associées
 - racisme, discriminations et réponses législatives
 - politique de la ville et banlieues
 - sécurité intérieure et délinquance
 - ordre public et maintien de la sécurité
 - violences urbaines et émeutes
 - politique pénale et justice
 - services publics (école, santé, transports, poste, énergie)
 - réforme et modernisation des services publics
 - privatisation ou ouverture à la concurrence des services publics
 - construction européenne et intégration européenne
 - traité de Maastricht et Union européenne
 - libre circulation (Schengen)
 - souveraineté nationale vs intégration européenne
 - monnaie et convergence économique européenne
 - cohabitation politique et fonctionnement institutionnel
 - alternance politique gauche/droite
 - rôle du président et du gouvernement sous la Ve République
 - crise de la représentation politique et abstention
 
IMPORTANT :
Ne pas associer explicitement ces thèmes à des partis.
Le signal politique doit rester implicite.

----------------------------------
ACTEURS VARIÉS
----------------------------------

- gouvernement
- ministre
- préfet
- maire
- administration

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
- ne jamais changer "name" ou "effect" ou "id"
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

- Chaque bloc contient des contraintes avec :
  - "id"
  - "effect"

- Chaque contrainte DOIT apparaître dans la sortie.

========================================================
ENTRÉES
========================================================

PHRASE SOURCE :
{sentence}

CONTRAINTES :
{constraints}
IMPORTANT :
Les champs ldb_id / direct_id sont des IDENTIFIANTS STRICTS.
Ils doivent apparaître tels quels dans la sortie.
Ne pas modifier, tronquer ou reformuler.
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
  "ldb": {{
    "text": "...",
    "constraints": [
      {{
        "id": ["S4","C2"],
        "effect": "..."
      }}
    ]
  }},
  "direct": {{
    "text": "...",
    "constraints": [
      {{
        "id": ["R4","D2"],
        "effect": "..."
      }}
    ]
  }}
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