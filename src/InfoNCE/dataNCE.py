import json
import re
import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
# =========================================================
# MODELS
# =========================================================

GMODEL = "Qwen/Qwen2.5-3B-Instruct"
PARA_MODEL = "Qwen/Qwen2.5-7B-Instruct"

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL, torch_dtype=torch.float16, device_map = {"": 0} if device == "cuda" else "auto"
).eval()


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

para_tokenizer = AutoTokenizer.from_pretrained(PARA_MODEL)
para_model = AutoModelForCausalLM.from_pretrained(
    PARA_MODEL, quantization_config=bnb_config, device_map = {"": 0} if device == "cuda" else "auto"
).eval()


# =========================================================
# PROMPTS
# =========================================================

GEN_PROMPT = """
Tu es un générateur de phrases politiques françaises réalistes (années 80-90).

Objectif :
Créer des phrases FACTUELLES de discours politique ou administratif.

Contraintes :
- France
- style institutionnel
- phrases courtes
- constats ou décisions
- aucune opinion
- aucune langue de bois volontaire

SORTIE STRICT JSON :
{{
  "sentences": ["..."]
}}

Nombre : {n}
"""

PARA_PROMPT = """
Tu es un générateur de dataset NLP contrastif.

Entrée :
Une phrase politique FACTUELLE.

OBJECTIF :
Générer EXACTEMENT :
- {n} variantes LANGUE_DE_BOIS
- {n} variantes DIRECT

Chaque index i correspond à une paire stricte.

========================
CONTRAINTES LDB
========================
C1: euphémisation
C2: passif / effacement sujet
C3: flou administratif
C4: abstraction institutionnelle
C5: dilution responsabilité
C6: jargon bureaucratique
C7: généralisation

========================
CONTRAINTES DIRECT (miroir)
========================
D1: explicite
D2: sujet identifié
D3: précision
D4: concret
D5: responsabilité claire
D6: langage simple
D7: spécifique

========================
FORMAT STRICT JSON
========================

{{
  "pairs": [
    {{
      "ldb": {{
        "text": "...",
        "constraints": ["C1", "C4"]
      }},
      "direct": {{
        "text": "...",
        "constraints": ["D1", "D4"]
      }}
    }}
  ]
}}

PHRASE :
{sentence}
"""

# =========================================================
# UTILS
# =========================================================


def build_batches(data: List[str], batch_size: int) -> List[List[str]]:
    """
    Split a list of strings into fixed-size batches.

    Args:
        data (List[str]): Input sentences.
        batch_size (int): Number of items per batch.

    Returns:
        List[List[str]]: Batched data.
    """
    logger.debug(f"Building batches with batch_size={batch_size}, total={len(data)}")

    return [data[i: i + batch_size] for i in range(0, len(data), batch_size)]


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction from model output.

    Args:
        text (str): Raw model output.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON or None if invalid.
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            logger.warning("Failed to parse extracted JSON block.")
            return None

    logger.warning("No JSON found in model output.")
    return None


# =========================================================
# QUALITY CHECK
# =========================================================


def quality_check_pairs(res: Dict[str, Any]) -> bool:
    """
    Validate structure and minimal quality of generated pairs.

    Args:
        res (Dict[str, Any]): Parsed model output.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not res or "pairs" not in res:
        return False

    if not res["pairs"]:
        return False

    for p in res["pairs"]:
        try:
            if not p.get("ldb") or not p.get("direct"):
                return False

            if not p["ldb"].get("text") or not p["direct"].get("text"):
                return False

            if len(p["ldb"].get("constraints", [])) == 0:
                return False

            if len(p["direct"].get("constraints", [])) == 0:
                return False

        except Exception:
            return False

    return True


# =========================================================
# AGENT 1 — BASE SENTENCES
# =========================================================


def call_generator(n: int) -> List[str]:
    """
    Generate base political sentences.

    Args:
        n (int): Number of sentences.

    Returns:
        List[str]: Generated sentences.
    """
    logger.info(f"Generating {n} base sentences...")

    prompt = GEN_PROMPT.format(n=n)

    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)

    with torch.no_grad():
        out = gen_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.8
        )

    text = gen_tokenizer.decode(out[0], skip_special_tokens=True)

    data = extract_json(text)

    if not data:
        logger.error("Failed to parse base sentence JSON.")
        return []

    sentences = data.get("sentences", [])

    logger.info(f"Generated {len(sentences)} base sentences.")
    return sentences


# =========================================================
# AGENT 2 — PARAPHRASER (GPU BATCHED)
# =========================================================


def call_paraphraser(sentences: List[str], n: int) -> List[Optional[Dict[str, Any]]]:
    """
    Generate LDB vs DIRECT pairs in batch.

    Args:
        sentences (List[str]): Input sentences.
        n (int): Number of variants per sentence.

    Returns:
        List[Optional[Dict[str, Any]]]: Parsed outputs or None.
    """
    logger.info(f"Paraphrasing batch of size {len(sentences)}")

    prompts = [PARA_PROMPT.format(sentence=s, n=n) for s in sentences]

    inputs = para_tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(para_model.device)

    with torch.no_grad():
        outputs = para_model.generate(
            **inputs, max_new_tokens=500, temperature=0.6, top_p=0.9
        )

    input_len = inputs["input_ids"].shape[1]

    decoded = para_tokenizer.batch_decode(
        outputs[:, input_len:], skip_special_tokens=True
    )

    results = []
    for i, d in enumerate(decoded):
        parsed = extract_json(d)
        if parsed is None:
            logger.warning(f"Failed to parse output at index {i}")
        results.append(parsed)

    return results


# =========================================================
# RETRY LOGIC
# =========================================================


def safe_generate(sentence: str, n: int, retries: int = 2) -> Optional[Dict[str, Any]]:
    """
    Retry generation until valid output is obtained.

    Args:
        sentence (str): Input sentence.
        n (int): Number of variants.
        retries (int): Retry count.

    Returns:
        Optional[Dict[str, Any]]: Valid result or None.
    """
    logger.info(f"Retrying generation for sentence: {sentence[:50]}...")

    for attempt in range(retries):
        res = call_paraphraser([sentence], n)[0]

        if quality_check_pairs(res):
            return res

        logger.warning(f"Attempt {attempt + 1} failed quality check.")

    logger.error("All retries failed.")
    return None


# =========================================================
# PIPELINE
# =========================================================


def generate_dataset(
    n_base: int = 20, n_variants: int = 5, batch_size: int = 4
) -> List[Dict[str, Any]]:
    """
    Full dataset generation pipeline.

    Args:
        n_base (int): Number of base sentences.
        n_variants (int): Variants per sentence.
        batch_size (int): Batch size for GPU inference.

    Returns:
        List[Dict[str, Any]]: Final dataset.
    """
    dataset: List[Dict[str, Any]] = []

    base_sentences = call_generator(n_base)
    logger.info(f"Agent1 produced {len(base_sentences)} sentences")

    batches = build_batches(base_sentences, batch_size)

    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i + 1}/{len(batches)}")

        results = call_paraphraser(batch, n_variants)

        for j, (sentence, res) in enumerate(zip(batch, results)):
            if not quality_check_pairs(res):
                logger.warning(f"Invalid output, retrying sentence {j}")
                res = safe_generate(sentence, n_variants)

            if not res:
                continue

            for p in res["pairs"]:
                dataset.append(
                    {
                        "base": sentence,
                        "ldb": p["ldb"]["text"],
                        "direct": p["direct"]["text"],
                        "ldb_constraints": p["ldb"]["constraints"],
                        "direct_constraints": p["direct"]["constraints"],
                        "pair_id": i,
                        "variant_id": j,
                    }
                )

    logger.info(f"Dataset generation complete: {len(dataset)} samples")
    return dataset


# =========================================================
# SAVE
# =========================================================


def save_json(data: List[Dict[str, Any]], path: str) -> None:
    """
    Save dataset to JSON file.

    Args:
        data (List[Dict[str, Any]]): Dataset.
        path (str): Output path.
    """
    logger.info(f"Saving dataset to {path}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    dataset = generate_dataset(n_base=15, n_variants=5, batch_size=3)

    save_json(dataset, "data/InfoNCE/pairsNCE.json")

    logger.info(f"DONE: {len(dataset)} samples")
