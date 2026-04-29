from __future__ import annotations

import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import gc

from .Constants.constraints import CONSTRAINTS
from .Constants.prompts import GEN_PROMPT, PARA_PROMPT
from .Constants.generation_context import SITUATIONS, ACTORS, TOPICS
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
# MODEL SETUP
# =========================================================

CACHE_DIR = "cache/Qwen7B"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR

MODEL = "Qwen/Qwen2.5-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE_DIR)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).eval()

# =========================================================
# TYPES
# =========================================================

ConstraintKey = str
ConstraintMap = Dict[str, Any]
DatasetItem = Dict[str, Any]
Batch = List[Tuple[str, str]]

# =========================================================
# CONSTRAINTS
# =========================================================


def build_constraint_maps(constraints: Dict) -> Tuple[List[str], Dict[str, Any], Dict[str, str]]:
    """
    Flatten constraint dictionary and build mirror mapping.

    Returns:
        keys: list of constraint IDs
        all_items: flattened constraint metadata
        mirror: mapping between direct / LDB constraints
    """
    all_items: Dict[str, Any] = {}
    keys: List[str] = []

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
# SAMPLING
# =========================================================


def sample_ldb_batches(keys: List[str], n: int, k: int) -> np.ndarray:
    """
    Sample random constraint batches.

    Args:
        keys: constraint keys
        n: number of batches
        k: batch size

    Returns:
        np.ndarray of shape (n, k)
    """
    ldb_keys = np.array([x for x in keys if x[0] in ["C", "S", "I"]])

    noise = np.random.rand(n, len(ldb_keys))
    idx = np.argsort(noise, axis=1)[:, :k]

    return ldb_keys[idx]

# =========================================================
# JSON UTIL
# =========================================================


def extract_json(text: str) -> Optional[Dict]:
    """
    Robust JSON extraction from LLM output.
    """
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
    return None


def clean_output(text: str) -> str:
    """
    Clean model output artifacts.
    """
    if "assistant" in text:
        text = text.split("assistant")[-1]
    return text.strip()

# =========================================================
# PROMPT BUILDING
# =========================================================


def build_constraints_payload(batch: "Batch") -> List[Dict[str, Any]]:
    """
    Build structured constraint payload (SOURCE OF TRUTH).

    IMPORTANT:
    This payload is NOT interpreted by the LLM.
    It is only injected as context for conditioning generation.

    Args:
        batch: list of (ldb_id, direct_id) pairs

    Returns:
        structured constraint objects
    """

    payload: List[Dict[str, Any]] = []

    for ldb_id, direct_id in batch:

        ldb = all_items[ldb_id]
        direct = all_items[direct_id]

        payload.append({
            "ldb": {
                "id": ldb_id,
                **ldb
            },
            "direct": {
                "id": direct_id,
                **direct
            }
        })

    return payload


def format_constraints(payload: List[Dict[str, Any]]) -> str:
    """
    Format constraints ONLY for LLM conditioning.
    No IDs exposed in output schema.
    """

    lines: List[str] = []

    for item in payload:

        ldb = item["ldb"]
        direct = item["direct"]

        lines.append(
            f"""
LDB
name: {ldb.get("name", "")}
effect: {ldb.get("effect", "")}

DIRECT
name: {direct.get("name", "")}
effect: {direct.get("effect", "")}
"""
        )

    return "\n".join(lines)


def aggregate_constraints(payload: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Transform payload into grouped constraint format.

    Returns:
        ldb_constraints, direct_constraints
    """

    ldb_ids = []
    direct_ids = []

    ldb_effects = []
    direct_effects = []

    for item in payload:
        ldb = item["ldb"]
        direct = item["direct"]

        ldb_ids.append(ldb["id"])
        direct_ids.append(direct["id"])

        ldb_effects.append(ldb["effect"])
        direct_effects.append(direct["effect"])

    ldb_constraints = [{
        "id": ldb_ids,
        "effect": " ; ".join(ldb_effects)
    }]

    direct_constraints = [{
        "id": direct_ids,
        "effect": " ; ".join(direct_effects)
    }]

    return ldb_constraints, direct_constraints


def call_paraphraser_batch(
    sentences: List[str],
    n: int = 4,
    k: int = 3
) -> List[List[Dict[str, Any]]]:
    """
    Micro-batched paraphrasing:
    - multiple sentences per forward pass
    - strict separation in output
    - no semantic mixing
    """

    # =========================================================
    # 1. PREP constraints ONCE per call (important gain)
    # =========================================================
    sampled = sample_ldb_batches(keys, n, k)

    mirror = mirror_map
    all_batches = []

    for row in sampled:
        batch = []
        for c in row:
            if c[0] in mirror:
                batch.append((c, mirror[c[0]] + c[1:]))

        if batch:
            all_batches.append(batch)

    if not all_batches:
        return [[] for _ in sentences]

    # =========================================================
    # 2. BUILD PROMPTS (micro-batch over sentences)
    # =========================================================
    system_msg = "Return STRICT JSON only. No explanation."

    prompts = []
    mapping = []  # (sentence_idx, batch_idx)

    apply = tokenizer.apply_chat_template

    for si, sentence in enumerate(sentences):
        for bi, batch in enumerate(all_batches):

            payload = build_constraints_payload(batch)

            user_msg = PARA_PROMPT.format(
                sentence=sentence,
                constraints=format_constraints(payload)
            )

            prompts.append(
                apply(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                )
            )

            mapping.append((si, bi, payload))

    # =========================================================
    # 3. TOKENIZE (single shot)
    # =========================================================
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # =========================================================
    # 4. GENERATION (unchanged)
    # =========================================================
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # =========================================================
    # 5. PARSE + GROUP BACK PER SENTENCE
    # =========================================================
    results: List[List[Dict[str, Any]]] = [[] for _ in sentences]

    for (text, (si, bi, payload)) in zip(decoded, mapping):

        data = extract_json(clean_output(text))

        if not data:
            continue

        ldb_constraints, direct_constraints = aggregate_constraints(payload)

        results[si].append({
            "base": sentences[si],
            "ldb": data.get("ldb"),
            "direct": data.get("direct"),
            "ldb_constraints": ldb_constraints,
            "direct_constraints": direct_constraints
        })

    return results

# =========================================================
# GENERATOR
# =========================================================


def sample_with_replacement(lst, k):
    arr = np.asarray(lst)

    replace = len(arr) < k
    idx = np.random.choice(len(arr), size=k, replace=replace)

    return arr[idx].tolist()


def call_generator(
    n: int,
    chunk_size: int = 8
) -> List[str]:

    sentences: List[str] = []
    system_msg = "Return JSON only."

    for i in tqdm(range(0, n, chunk_size), desc="Generating base"):

        batch_n = min(chunk_size, n - i)

        sub_actors = sample_with_replacement(ACTORS, batch_n)
        sub_situations = sample_with_replacement(SITUATIONS, batch_n)
        sub_topics = sample_with_replacement(TOPICS, batch_n)

        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": GEN_PROMPT.format(
                            n=1,
                            actors=a,
                            situations=s,
                            topics=t
                        )
                    }
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for a, s, t in zip(sub_actors, sub_situations, sub_topics)
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1  # petit bonus utile
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in decoded:
            j = extract_json(clean_output(text))
            if j and "sentences" in j:
                sentences.extend(j["sentences"])

        torch.cuda.empty_cache()

    return sentences
# =========================================================
# PIPELINE
# =========================================================


CHECKPOINT_PATH = "data/InfoNCE/checkpoint_dataset.json"


def ask_resume() -> bool:
    """
    Ask user if they want to resume from checkpoint.
    """
    ans = input("\nResume from checkpoint ? (y/n) >>> ").strip().lower()

    return ans in {"y", "yes", "o", "oui"}


def save_checkpoint(path: str, idx: int, data: List[Dict[str, Any]]) -> None:
    """
    Save intermediate dataset state.
    """
    tmp = {
        "done_idx": idx,
        "data": data
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(tmp, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: str, resume: bool):
    """
    Load checkpoint only if resume is True.
    """
    if not resume:
        return 0, []

    if not os.path.exists(path):
        return 0, []

    with open(path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)

    return ckpt.get("done_idx", 0), ckpt.get("data", [])


def generate_dataset(
    n_base: int = 10,
    n_variants: int = 3,
    batch_size: int = 4,
    chunk_size: int = 8,
    resume: bool = True
) -> List[Dict[str, Any]]:

    torch.cuda.empty_cache()
    gc.collect()

    start_idx, dataset = load_checkpoint(CHECKPOINT_PATH, resume)

    logger.info("Resume: %s | start_idx: %d | loaded: %d",
                resume, start_idx, len(dataset))

    # =========================================================
    # BASE GENERATION (unchanged)
    # =========================================================
    base = call_generator(n_base, chunk_size)
    base = base[start_idx:]

    logger.info("Base sentences: %d", len(base))

    # =========================================================
    # LOCAL BINDINGS (micro-opt)
    # =========================================================
    paraphraser = call_paraphraser_batch
    saver = save_checkpoint

    # =========================================================
    # MAIN LOOP
    # =========================================================
    for i in tqdm(range(0, len(base), batch_size), desc="Dataset building"):

        batch = base[i:i + batch_size]
        global_idx = start_idx + i

        try:
            append = dataset.append

            # =========================================================
            # MICRO-BATCH PARAPHRASER (NEW)
            # =========================================================
            batch_results = paraphraser(batch, n_variants)

            for sentence_results in batch_results:
                for res in sentence_results:

                    if not res:
                        continue

                    append({
                        "base": res["base"],
                        "ldb": res["ldb"],
                        "direct": res["direct"],
                        "ldb_constraints": res["ldb_constraints"],
                        "direct_constraints": res["direct_constraints"]
                    })

            # =========================================================
            # CHECKPOINT (unchanged but cleaner)
            # =========================================================
            if i % (batch_size * 5) == 0:
                saver(
                    CHECKPOINT_PATH,
                    global_idx + batch_size,
                    dataset
                )

        except RuntimeError:
            logger.error("GPU crash → checkpoint save")

            saver(
                CHECKPOINT_PATH,
                global_idx,
                dataset
            )

            raise

    return dataset
# =========================================================
# SAVE
# =========================================================


def save_json(data: Any, path: str) -> None:
    """
    Save dataset to JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================================================
# MAIN
# =========================================================


if __name__ == "__main__":

    resume = ask_resume()

    if not resume:
        print("Resetting checkpoint...")
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)

    dataset = generate_dataset(
        n_base=200,
        n_variants=3,
        batch_size=10,
        chunk_size=25,
        resume=resume
    )

    save_json(dataset, "data/InfoNCE/groupedNCEV2.json")

    logger.info("DONE: %d samples", len(dataset))