import json
import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from helper_function.print import *

import os


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("dataset")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# =========================================================
# MODEL
# =========================================================
CACHE_DIR = "cache/Qwen7B" 
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR

MODEL = "Qwen/Qwen2.5-7B-Instruct"

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE_DIR)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    dtype=torch.float16 if device == "cuda" else torch.float32
).eval()

# =========================================================
# PROMPTS (RENFORCÉS JSON STRICT)
# =========================================================

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

OBJECTIF :
À partir d'une phrase factuelle, générer EXACTEMENT {n} paires (LDB / DIRECT).

Chaque paire doit :
- conserver STRICTEMENT le même contenu factuel
- appliquer des contraintes LDB
- appliquer les contraintes DIRECT EXACTEMENT inverses

========================================================
MODÈLE DE CONTRAINTES
========================================================

Chaque transformation repose sur 3 types de contraintes :

C = intention (sens et positionnement)
S = structure (syntaxe et forme)
I = information (organisation et densité)

Chaque contrainte LDB possède un miroir DIRECT.

------------------------
LDB → DIRECT (MIROIR)
------------------------

INTENTION (C)
C1 euphémisation              → D1 explicitation
C2 effacement du sujet        → D2 sujet explicite
C3 flou                       → D3 précision
C4 abstraction                → D4 concret
C5 dilution responsabilité    → D5 responsabilité claire
C6 jargon                     → D6 langage simple
C7 généralisation             → D7 spécification

STRUCTURE (S)
S1 phrase longue              → R1 phrase courte
S2 passif                     → R2 actif
S3 nominalisation             → R3 verbes directs
S4 ajout inutile              → R4 suppression du superflu
S5 complexité syntaxique      → R5 simplicité syntaxique
S6 sujet éloigné/inexistant   → R6 sujet en début
S7 multi-idées                → R7 une seule idée
S8 connecteurs bureaucratiques→ R8 aucun connecteur
S9 redondance                 → R9 aucune répétition
S10 justification ajoutée     → R10 aucune justification

INFORMATION (I)
I1 dilution information       → J1 densité informationnelle
I2 retard info clé            → J2 information immédiate
I3 contexte avant action      → J3 action immédiate
I4 implicite                  → J4 explicite
I5 fragmentation              → J5 information compacte

========================================================
RÈGLES DE GÉNÉRATION
========================================================

Pour chaque paire :

1. Sélectionner aléatoirement 1 à 4 contraintes :
   - 0 à 3 contraintes C
   - 0 à 3 contraintes S
   - 0 à 2 contraintes I

2. Générer la phrase LDB en respectant STRICTEMENT ces contraintes.

3. Générer la phrase DIRECT en appliquant EXACTEMENT les contraintes inverses correspondantes.

========================================================
CONTRAINTES GLOBALES
========================================================

INTERDICTIONS :
- modifier le fait
- ajouter une nouvelle information
- supprimer une information essentielle

========================================================
FORMAT STRICT JSON
========================================================

{{
  "pairs": [
    {{
      "ldb": {{
        "text": "...",
        "constraints": ["C2", "S2", "I2"]
      }},
      "direct": {{
        "text": "...",
        "constraints": ["D2", "R2", "J2"]
      }}
    }}
  ]
}}

PHRASE SOURCE :
{sentence}
"""




# =========================================================
# UTILS
# =========================================================

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extraction robuste JSON"""

    try:
        return json.loads(text)
    except:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except:
            return None

    return None


def build_batches(data: List[str], batch_size: int):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
def clean_output(text: str) -> str:
    if "assistant" in text:
        text = text.split("assistant")[-1]
    return text.strip()
# =========================================================
# GENERATOR
# =========================================================

def call_generator(n: int) -> List[str]:

    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "JSON only."},
                {"role": "user", "content": GEN_PROMPT.format(n=1)}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for _ in range(n)
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.6,
            top_p=0.9,
            do_sample=True
        )

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

    results = []
    for d in decoded:
        d = clean_output(d)
        data = extract_json(d)
        if data:
            results.extend(data.get("sentences", []))

    return results

# =========================================================
# PARAPHRASER
# =========================================================
def call_paraphraser(sentences: List[str], n: int):

    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "JSON only."},
                {"role": "user", "content": PARA_PROMPT.format(sentence=s, n=n)}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for s in sentences
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.2,
            top_p=0.9,
            do_sample=True
        )

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

    print("\n--- RAW PARAPHRASER OUTPUT ---\n", decoded, "\n")

    results = []
    for d in decoded:
        d = clean_output(d)
        data = extract_json(d)
        results.append(data)

    return results
# =========================================================
# QUALITY CHECK
# =========================================================

def quality_check(res: Dict[str, Any]) -> bool:
    if not res or "pairs" not in res:
        return False

    for p in res["pairs"]:
        if not p.get("ldb") or not p.get("direct"):
            return False
        if not p["ldb"].get("text") or not p["direct"].get("text"):
            return False

    return True

# =========================================================
# RETRY
# =========================================================

def safe_generate(sentence: str, n: int, retries: int = 2):
    for _ in range(retries):
        res = call_paraphraser([sentence], n)[0]
        if quality_check(res):
            return res
    return None

# =========================================================
# PIPELINE
# =========================================================

def generate_dataset(n_base=10, n_variants=2, batch_size=2):

    dataset = []

    base = call_generator(n_base)
    logger.info(f"Base sentences: {len(base)}")

    batches = build_batches(base, batch_size)

    for batch in batches:

        results = call_paraphraser(batch, n_variants)

        for sentence, res in zip(batch, results):

            if not quality_check(res):
                res = safe_generate(sentence, n_variants)

            if not res:
                continue

            for p in res["pairs"]:
                dataset.append({
                    "base": sentence,
                    "ldb": p["ldb"]["text"],
                    "direct": p["direct"]["text"],
                    "ldb_constraints": p["ldb"]["constraints"],
                    "direct_constraints": p["direct"]["constraints"]
                })

    return dataset

# =========================================================
# SAVE
# =========================================================

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    dataset = generate_dataset(n_base=4, n_variants=3, batch_size=2)
    save_json(dataset, "data/InfoNCE/pairsNCE.json")
    logger.info(f"DONE: {len(dataset)} samples")