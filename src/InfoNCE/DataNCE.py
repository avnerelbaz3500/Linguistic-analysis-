from __future__ import annotations

import gc
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .Constants.constraints import CONSTRAINTS
from .Constants.generation_context import ACTORS, SITUATIONS, TOPICS
from .Constants.prompts import GEN_PROMPT, PARA_PROMPT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("dataset")

device: str = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", device)


CACHE_DIR = "cache/Qwen7B"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).eval()


ConstraintKey = str
ConstraintMap = Dict[str, Any]
DatasetItem = Dict[str, Any]
Batch = List[Tuple[str, str]]


def build_constraint_maps(
    constraints: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], Dict[str, Any], Dict[str, str]]:
    """Flatten constraint dict and build mirror mapping."""
    all_items: Dict[str, Any] = {}
    keys: List[str] = []

    mirror = {
        "C": "D",
        "D": "C",
        "S": "R",
        "R": "S",
        "I": "J",
        "J": "I",
    }

    for cat in constraints.values():
        for k, v in cat.items():
            all_items[k] = v
            keys.append(k)

    return keys, all_items, mirror


keys, all_items, mirror_map = build_constraint_maps(CONSTRAINTS)


def sample_ldb_batches(keys: List[str], n: int, k: int) -> np.ndarray:
    """Sample random batches of LDB constraints."""
    ldb_keys = np.array([x for x in keys if x[0] in {"C", "S", "I"}])
    noise = np.random.rand(n, len(ldb_keys))
    idx = np.argsort(noise, axis=1)[:, :k]
    return ldb_keys[idx]


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from model output."""
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
    return None


def clean_output(text: str) -> str:
    """Remove chat artifacts from model output."""
    if "assistant" in text:
        text = text.split("assistant")[-1]
    return text.strip()


def build_constraints_payload(batch: Batch) -> List[Dict[str, Any]]:
    """Build structured constraint payload."""
    payload: List[Dict[str, Any]] = []

    for ldb_id, direct_id in batch:
        payload.append(
            {
                "ldb": {"id": ldb_id, **all_items[ldb_id]},
                "direct": {"id": direct_id, **all_items[direct_id]},
            }
        )

    return payload


def format_constraints(payload: List[Dict[str, Any]]) -> str:
    """Format constraints for LLM conditioning."""
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


def aggregate_constraints(
    payload: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Aggregate constraints into grouped format."""
    ldb_ids, direct_ids = [], []
    ldb_effects, direct_effects = [], []

    for item in payload:
        ldb = item["ldb"]
        direct = item["direct"]

        ldb_ids.append(ldb["id"])
        direct_ids.append(direct["id"])
        ldb_effects.append(ldb["effect"])
        direct_effects.append(direct["effect"])

    return (
        [{"id": ldb_ids, "effect": " ; ".join(ldb_effects)}],
        [{"id": direct_ids, "effect": " ; ".join(direct_effects)}],
    )


def call_paraphraser_batch(
    sentences: List[str], n: int = 4, k: int = 3
) -> List[List[Dict[str, Any]]]:
    """Generate paraphrases with constraint conditioning."""
    sampled = sample_ldb_batches(keys, n, k)

    all_batches: List[Batch] = []
    for row in sampled:
        batch: Batch = [
            (c, mirror_map[c[0]] + c[1:])
            for c in row
            if c[0] in mirror_map
        ]
        if batch:
            all_batches.append(batch)

    if not all_batches:
        return [[] for _ in sentences]

    system_msg = "Return STRICT JSON only. No explanation."
    prompts: List[str] = []
    mapping: List[Tuple[int, int, List[Dict[str, Any]]]] = []

    for si, sentence in enumerate(sentences):
        for bi, batch in enumerate(all_batches):
            payload = build_constraints_payload(batch)

            user_msg = PARA_PROMPT.format(
                sentence=sentence,
                constraints=format_constraints(payload),
            )

            prompts.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            mapping.append((si, bi, payload))

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    results: List[List[Dict[str, Any]]] = [[] for _ in sentences]

    for text, (si, _, payload) in zip(decoded, mapping):
        data = extract_json(clean_output(text))
        if not data:
            continue

        ldb_c, direct_c = aggregate_constraints(payload)

        results[si].append(
            {
                "base": sentences[si],
                "ldb": data.get("ldb"),
                "direct": data.get("direct"),
                "ldb_constraints": ldb_c,
                "direct_constraints": direct_c,
            }
        )

    return results


def sample_with_replacement(lst: List[Any], k: int) -> List[Any]:
    """Sample with replacement if needed."""
    arr = np.asarray(lst)
    replace = len(arr) < k
    idx = np.random.choice(len(arr), size=k, replace=replace)
    return arr[idx].tolist()


def call_generator(n: int, chunk_size: int = 8) -> List[str]:
    """Generate base sentences."""
    sentences: List[str] = []
    system_msg = "Return JSON only."

    for i in tqdm(range(0, n, chunk_size), desc="Generating base"):
        batch_n = min(chunk_size, n - i)

        actors = sample_with_replacement(ACTORS, batch_n)
        situations = sample_with_replacement(SITUATIONS, batch_n)
        topics = sample_with_replacement(TOPICS, batch_n)

        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": GEN_PROMPT.format(
                            n=1, actors=a, situations=s, topics=t
                        ),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for a, s, t in zip(actors, situations, topics)
        ]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in decoded:
            j = extract_json(clean_output(text))
            if j and "sentences" in j:
                sentences.extend(j["sentences"])

        torch.cuda.empty_cache()

    return sentences


CHECKPOINT_PATH = "data/InfoNCE/checkpoint_dataset.json"


def ask_resume() -> bool:
    """Ask user whether to resume from checkpoint."""
    ans = input("\nResume from checkpoint ? (y/n) >>> ").strip().lower()
    return ans in {"y", "yes", "o", "oui"}


def save_checkpoint(path: str, idx: int, data: List[DatasetItem]) -> None:
    """Save checkpoint to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"done_idx": idx, "data": data}, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: str, resume: bool) -> Tuple[int, List[DatasetItem]]:
    """Load checkpoint if resume is enabled."""
    if not resume or not os.path.exists(path):
        return 0, []

    with open(path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)

    return ckpt.get("done_idx", 0), ckpt.get("data", [])


def generate_dataset(
    n_base: int = 10,
    n_variants: int = 3,
    batch_size: int = 4,
    chunk_size: int = 8,
    resume: bool = True,
) -> List[DatasetItem]:
    """Main dataset generation pipeline."""
    torch.cuda.empty_cache()
    gc.collect()

    start_idx, dataset = load_checkpoint(CHECKPOINT_PATH, resume)

    logger.info(
        "Resume=%s | start_idx=%d | loaded=%d",
        resume,
        start_idx,
        len(dataset),
    )

    base = call_generator(n_base, chunk_size)[start_idx:]
    logger.info("Base sentences: %d", len(base))

    for i in tqdm(range(0, len(base), batch_size), desc="Dataset building"):
        batch = base[i : i + batch_size]
        global_idx = start_idx + i

        try:
            results = call_paraphraser_batch(batch, n_variants)

            for sentence_results in results:
                for res in sentence_results:
                    if res:
                        dataset.append(res)

            if i % (batch_size * 5) == 0:
                save_checkpoint(
                    CHECKPOINT_PATH, global_idx + batch_size, dataset
                )

        except RuntimeError:
            logger.error("GPU crash → checkpoint saved")
            save_checkpoint(CHECKPOINT_PATH, global_idx, dataset)
            raise

    return dataset


def save_json(data: Any, path: str) -> None:
    """Save JSON to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    resume = ask_resume()

    if not resume and os.path.exists(CHECKPOINT_PATH):
        print("Resetting checkpoint...")
        os.remove(CHECKPOINT_PATH)

    dataset = generate_dataset(
        n_base=200,
        n_variants=3,
        batch_size=10,
        chunk_size=25,
        resume=resume,
    )

    save_json(dataset, "data/InfoNCE/groupedNCEV2.json")
    logger.info("DONE: %d samples", len(dataset))