import json
import re
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# MODELS
# =========================================================

GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
PARA_MODEL = "Qwen/Qwen2.5-7B-Instruct"

gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL, torch_dtype=torch.float16, device_map="auto"
).eval()

para_tokenizer = AutoTokenizer.from_pretrained(PARA_MODEL)
para_model = AutoModelForCausalLM.from_pretrained(
    PARA_MODEL, load_in_4bit=True, torch_dtype=torch.float16, device_map="auto"
).eval()


# =========================================================
# PROMPTS (améliorés mais conservent ton idée)
# =========================================================

GEN_PROMPT = """
Tu es un générateur de données politiques françaises réalistes (style années 80-90).

OBJECTIF :
Produire des phrases crédibles de discours politique ou administratif.

CONTRAINTES STRICTES :
- uniquement France
- style institutionnel
- phrases factuelles simples
- annonces, décisions, constats
- pas d'analyse, pas d'opinion
- pas de langue de bois volontaire
- pas de phrases longues

SORTIE STRICT JSON :
{
  "sentences": ["..."]
}

Nombre : {n}
"""


# PARA_PROMPT = """
# Tu es un générateur de dataset NLP contrastif.

# Entrée : phrase politique FACTUELLE.

# Tu dois produire EXACTEMENT :
# - {n} variantes "LANGUE_DE_BOIS"
# - {n} variantes "DIRECT"

# RÈGLES IMPORTANTES :

# LANGUE_DE_BOIS :
# - euphémisation
# - passif
# - flou administratif
# - abstraction institutionnelle
# - dilution du responsable
# - formulation bureaucratique
# - phrases crédibles politiquement

# DIRECT :
# - clair
# - action explicite
# - sujet identifié
# - concret
# - sans ambiguïté

# CONTRAINTES :
# - même sens exact
# - aucune info ajoutée
# - même événement uniquement reformulé

# LONGUEUR :
# - phrases similaires en longueur

# SORTIE STRICT JSON :
# {
#   "ldb": ["..."],
#   "direct": ["..."]
# }

# PHRASE :
# {sentence}
# """


PARA_PROMPT = """
Tu es un générateur de dataset NLP contrastif pour analyse de langue politique.

Entrée :
Une phrase politique FACTUELLE.

OBJECTIF :
Générer EXACTEMENT :
- {n} variantes "LANGUE_DE_BOIS"
- {n} variantes "DIRECT"

Chaque variante LDB doit correspondre à une variante DIRECT avec le même index.

========================
CONTRAINTES SYMÉTRIQUES
========================

Chaque paire doit utiliser EXACTEMENT les mêmes contraintes structurelles,
mais appliquées en version inversée.

------------------------
CONTRAINTES LDB
------------------------
C1: euphémisation
C2: passif / effacement du sujet
C3: flou volontaire
C4: abstraction institutionnelle
C5: dilution de responsabilité
C6: jargon administratif
C7: généralisation
C8: modalisation prudente

------------------------
CONTRAINTES DIRECT (miroir exact)
------------------------
D1: reformulation explicite (anti-C1)
D2: sujet explicite (anti-C2)
D3: précision maximale (anti-C3)
D4: concrétisation (anti-C4)
D5: responsabilité identifiée (anti-C5)
D6: langage simple non institutionnel (anti-C6)
D7: spécification factuelle (anti-C7)
D8: absence de modalisation (anti-C8)

========================
RÈGLE DE PAIRING OBLIGATOIRE
========================
Pour chaque i de 1 à n :
- LDB[i] et DIRECT[i] doivent :
  - décrire exactement le même contenu factuel
  - utiliser des structures inversées
  - avoir 1 à 4 contraintes chacune
  - partager un lien sémantique strict

========================
SORTIE STRICT JSON
========================

{
  "pairs": [
    {
      "ldb": {
        "text": "...",
        "constraints": ["C1", "C4"]
      },
      "direct": {
        "text": "...",
        "constraints": ["D1", "D4"]
      }
    }
  ]
}

PHRASE :
{sentence}
"""

# =========================================================
# UTILS
# =========================================================


def build_batches(data: List[str], batch_size: int) -> List[List[str]]:
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except:
                return None
    return None


# =========================================================
# QUALITY CHECK
# =========================================================


def quality_check(sample: Dict[str, Any]) -> bool:
    """
    Vérifie si le sample est exploitable.
    """

    if not sample:
        return False

    ldb = sample.get("ldb", [])
    direct = sample.get("direct", [])

    if not isinstance(ldb, list) or not isinstance(direct, list):
        return False

    if len(ldb) == 0 or len(direct) == 0:
        return False

    # check diversité minimale
    if len(set(ldb)) < len(ldb) * 0.7:
        return False

    if len(set(direct)) < len(direct) * 0.7:
        return False

    return True


# =========================================================
# AGENT 1 — BASE SENTENCES
# =========================================================


def call_generator(n: int) -> List[str]:
    prompt = GEN_PROMPT.format(n=n)

    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)

    with torch.no_grad():
        out = gen_model.generate(
            **inputs, max_new_tokens=400, temperature=0.9, top_p=0.95
        )

    text = gen_tokenizer.decode(out[0], skip_special_tokens=True)

    data = extract_json(text)
    if not data:
        return []

    return data.get("sentences", [])


# =========================================================
# AGENT 2 — PARAPHRASER (batched GPU)
# =========================================================


def call_paraphraser(sentences: List[str], n: int) -> List[Optional[Dict[str, Any]]]:

    prompts = [PARA_PROMPT.format(sentence=s, n=n) for s in sentences]

    inputs = para_tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(para_model.device)

    with torch.no_grad():
        outputs = para_model.generate(
            **inputs, max_new_tokens=450, temperature=0.75, top_p=0.9
        )

    input_len = inputs["input_ids"].shape[1]

    decoded = para_tokenizer.batch_decode(
        outputs[:, input_len:], skip_special_tokens=True
    )

    return [extract_json(d) for d in decoded]


# =========================================================
# RETRY LOGIC
# =========================================================


def safe_generate_paraphrase(sentence: str, n: int, retries: int = 2):

    for _ in range(retries):
        res = call_paraphraser([sentence], n)[0]

        if quality_check(res):
            return res

    return None


# =========================================================
# PIPELINE
# =========================================================


def generate_full_dataset(
    n_base: int = 20, n_variants: int = 5, batch_size: int = 4
) -> List[Dict[str, Any]]:

    dataset = []

    # -------------------------
    # Agent 1
    # -------------------------
    base_sentences = call_generator(n_base)
    print(f"[Agent1] base sentences: {len(base_sentences)}")

    batches = build_batches(base_sentences, batch_size)

    # -------------------------
    # Agent 2
    # -------------------------
    for i, batch in enumerate(batches):
        print(f"[Agent2] batch {i + 1}/{len(batches)}")

        results = call_paraphraser(batch, n_variants)

        for sentence, res in zip(batch, results):
            if not quality_check(res):
                res = safe_generate_paraphrase(sentence, n_variants)

            if not res:
                continue

            dataset.append(
                {"base": sentence, "ldb": res["ldb"], "direct": res["direct"]}
            )

    return dataset


# =========================================================
# SAVE
# =========================================================


def save_json(data: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    dataset = generate_full_dataset(n_base=15, n_variants=5, batch_size=3)

    save_json(dataset, "data/InfoNCE/pairsNCE.json")

    print("DONE:", len(dataset))
