import json
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import os
from .Constants.constraints import CONSTRAINTS
import gc
from .Constants.prompts import GEN_PROMPT, PARA_PROMPT

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
# CONSTRAINTS FLAT MAP
# =========================================================

def build_constraint_maps(constraints):
    all_items = {}
    keys = []

    mirror = {
        "C": "D",
        "D": "C",
        "S": "R",
        "R": "S",
        "I": "J",
        "J": "I"
    }

    for cat in constraints.values():
        for k, v in cat.items():
            all_items[k] = v
            keys.append(k)

    return keys, all_items, mirror


keys, all_items, mirror_map = build_constraint_maps(CONSTRAINTS)

# =========================================================
# SAMPLING (PROPRE + RAPIDE)
# =========================================================

def sample_ldb_batches(keys, n, k):
    ldb_keys = np.array([x for x in keys if x[0] in ["C", "S", "I"]])

    noise = np.random.rand(n, len(ldb_keys))
    idx = np.argsort(noise, axis=1)[:, :k]

    return ldb_keys[idx]
# =========================================================
# JSON SAFE PARSER
# =========================================================

def extract_json(text: str):
    try:
        return json.loads(text)
    except:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except:
                return None
    return None


def clean_output(text: str):
    if "assistant" in text:
        text = text.split("assistant")[-1]
    return text.strip()

# =========================================================
# PROMPT BUILDER (CORE LOGIC)
# =========================================================

def build_constraints_payload(batch):
    return [
        {
            "ldb": {"id": l, **all_items[l]},
            "direct": {"id": d, **all_items[d]}
        }
        for l, d in batch
    ]
def format_constraints(batch):
    lines = []

    for item in build_constraints_payload(batch):
        ldb = item["ldb"]
        direct = item["direct"]

        lines.append(f"""
    LDB[{ldb['id']}]
    name: {ldb['name']}
    effect: {ldb['effect']}

    DIRECT[{direct['id']}]
    name: {direct['name']}
    effect: {direct['effect']}
    """)

    return "\n".join(lines)
# =========================================================
# PARAPHRASER (VRAI BATCH GPU)
# =========================================================

def call_paraphraser(sentence: str, n: int = 4, k: int = 3):

    sampled = sample_ldb_batches(keys, n, k)

    flat = []
    for row in sampled:
        for c in row:
            prefix = c[0]
            if prefix not in mirror_map:
                continue
            flat.append((c, mirror_map[prefix] + c[1:]))

    batches = [
        flat[i:i+k]
        for i in range(0, len(flat), k)
    ]

    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "JSON only."},
                {"role": "user", "content": PARA_PROMPT.format(
                    sentence=sentence,
                    constraints=format_constraints(batch)
                )}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for batch in batches
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

    results = []
    for d in decoded:
        data = extract_json(clean_output(d))
        if not data:
            continue

        if "ldb" in data and "direct" in data:
            results.append(data)

    return results

# =========================================================
# GENERATOR (SIMPLE)
# =========================================================

def call_generator(n: int):
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

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=250, temperature=0.6, top_p=0.9)

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

    sentences = []
    for d in decoded:
        j = extract_json(clean_output(d))
        if j:
            sentences.extend(j.get("sentences", []))

    return sentences

# =========================================================
# PIPELINE
# =========================================================
def generate_dataset(n_base=10, n_variants=3, batch_size=2):
    torch.cuda.empty_cache()
    gc.collect()

    dataset = []

    base = call_generator(n_base)
    logger.info(f"Base sentences: {len(base)}")

    for i in tqdm(range(0, len(base), batch_size), desc="Dataset building", unit="batch"):

        batch = base[i:i + batch_size]

        for sentence in batch:

            results = call_paraphraser(sentence, n_variants)

            for res in results:

                if not res or "ldb" not in res or "direct" not in res:
                    continue

                dataset.append({
                    "base": sentence,
                    "ldb": res["ldb"]["text"],
                    "direct": res["direct"]["text"],
                    "ldb_constraints": res["ldb"]["constraints"],
                    "direct_constraints": res["direct"]["constraints"]
                })

            torch.cuda.empty_cache()
            gc.collect()
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
    dataset = generate_dataset(n_base=100, n_variants=3, batch_size=4)
    save_json(dataset, "data/InfoNCE/pairsNCE100.json")
    logger.info(f"DONE: {len(dataset)} samples")