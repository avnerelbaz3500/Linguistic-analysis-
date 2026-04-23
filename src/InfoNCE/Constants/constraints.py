CONSTRAINTS = {
    "INTENTION": {
        "C1": {
            "name": "euphémisation",
            "effect": "remplacer un terme direct ou négatif par une formulation atténuée ou neutre"
        },
        "D1": {
            "name": "explicitation",
            "effect": "remplacer toute atténuation par un terme direct et explicite"
        },

        "C2": {
            "name": "effacement_du_sujet",
            "effect": "supprimer ou masquer le sujet (passif, impersonnel)"
        },
        "D2": {
            "name": "sujet_explicite",
            "effect": "placer un sujet clair et identifié en début de phrase"
        },

        "C3": {
            "name": "flou",
            "effect": "rendre au moins un élément imprécis (temps, lieu, quantité, acteur)"
        },
        "D3": {
            "name": "precision",
            "effect": "rendre tous les éléments explicites et non ambigus"
        },

        "C4": {
            "name": "abstraction",
            "effect": "remplacer une action concrète par une formulation générique ou abstraite"
        },
        "D4": {
            "name": "concret",
            "effect": "utiliser des actions et objets concrets et observables"
        },


        "C5": {
            "name": "dilution_responsabilite",
            "effect": "éviter d’assigner clairement une responsabilité (passif, neutre)"
        },
        "D5": {
            "name": "responsabilite_claire",
            "effect": "attribuer explicitement l’action à un acteur identifié"
        },

        "C6": {
            "name": "jargon",
            "effect": "introduire vocabulaire administratif, technique ou bureaucratique"
        },
        "D6": {
            "name": "langage_simple",
            "effect": "utiliser un vocabulaire simple et courant"
        },

        "C7": {
            "name": "generalisation",
            "effect": "remplacer un élément précis par une formulation vague"
        },
        "D7": {
            "name": "specification",
            "effect": "remplacer les termes vagues par des éléments précis"
        }
    },
    "STRUCTURE": {
        "S1": {
            "name": "phrase_longue",
            "effect": "phrase ≥15 mots avec au moins une subordination (ex: que, afin de, dans la mesure où)"
        },
        "R1": {
            "name": "phrase_courte",
            "effect": "phrase ≤12 mots sans subordination, structure simple sujet-verbe-objet"
        },

        "S2": {
            "name": "passif",
            "effect": "voix passive avec auxiliaire être + participe passé, sujet non acteur"
        },
        "R2": {
            "name": "actif",
            "effect": "voix active, sujet explicite en position initiale qui réalise l’action"
        },
        "S3": {
            "name": "nominalisation",
            "effect": "remplacement des verbes par noms abstraits (ex: mise en œuvre, réalisation, gestion)"
        },
        "R3": {
            "name": "verbe_direct",
            "effect": "remplacement des noms abstraits par verbes d’action concrets"
        },
        "S4": {
            "name": "superflu",
            "effect": "ajout d’éléments non nécessaires à la compréhension du fait (précisions, reformulations, contexte inutile)"
        },
        "R4": {
            "name": "sans_superflu",
            "effect": "suppression de tout élément non essentiel, conservation du fait brut"
        },
        "S5": {
            "name": "syntaxe_complexe",
            "effect": "structure avec subordination ou enchâssement de propositions"
        },
        "R5": {
            "name": "syntaxe_simple",
            "effect": "une seule proposition principale sans subordination"
        },

        "S6": {
            "name": "sujet_eloigne",
            "effect": "sujet absent, implicite ou placé après le verbe principal"
        },
        "R6": {
            "name": "sujet_initial",
            "effect": "sujet explicite placé en début de phrase avant l’action"
        },

        "S7": {
            "name": "multi_idees",
            "effect": "au moins deux actions ou informations distinctes dans la même phrase"
        },
        "R7": {
            "name": "idee_unique",
            "effect": "une seule action ou information par phrase"
        },

        "S8": {
            "name": "connecteurs_bureaucratiques",
            "effect": "usage de connecteurs institutionnels (dans le cadre de, afin de, en vue de)"
        },
        "R8": {
            "name": "sans_connecteur",
            "effect": "absence totale de connecteurs, formulation directe"
        },

        "S9": {
            "name": "redondance",
            "effect": "répétition d’une même information sous plusieurs formes"
        },
        "R9": {
            "name": "non_redondant",
            "effect": "chaque information apparaît une seule fois"
        },

        "S10": {
            "name": "justification",
            "effect": "ajout d’une explication ou justification non nécessaire au fait"
        },
        "R10": {
            "name": "sans_justification",
            "effect": "aucune justification, uniquement le fait brut"
        }
    },
    "INFORMATION": {
        "I1": {
            "name": "dilution",
            "effect": "ajout de mots sans information nouvelle, augmentation de longueur sans gain sémantique"
        },
        "J1": {
            "name": "densite",
            "effect": "chaque segment lexical apporte une information nouvelle et utile"
        },

        "I2": {
            "name": "retard_info",
            "effect": "information principale apparaît après les premiers mots de la phrase"
        },
        "J2": {
            "name": "info_immediate",
            "effect": "information principale placée dès le début de la phrase"
        },

        "I3": {
            "name": "contexte_avant_action",
            "effect": "phrase commence par contexte ou cadrage avant l’action principale"
        },
        "J3": {
            "name": "action_directe",
            "effect": "phrase commence directement par l’action principale"
        },

        "I4": {
            "name": "implicite",
            "effect": "au moins un élément nécessaire à la compréhension est sous-entendu"
        },
        "J4": {
            "name": "explicite",
            "effect": "tous les éléments nécessaires sont formulés sans ambiguïté"
        },

        "I5": {
            "name": "fragmentation",
            "effect": "informations dispersées dans la phrase, ordre non compact"
        },
        "J5": {
            "name": "compacte",
            "effect": "informations regroupées de manière linéaire et compacte"
        }
    }
}