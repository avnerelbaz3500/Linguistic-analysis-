import argparse
import json
import re
import sys
import requests
from pathlib import Path

from helper_function.print import *

def remove_duplicates(dataset: list) -> list:
    """Removes duplicate dictionaries from the dataset based on the 'direct' key."""
    seen = set()
    unique_dataset = []
    for item in dataset:
        identifier = item.get("direct", "")
        if identifier not in seen:
            seen.add(identifier)
            unique_dataset.append(item)
    return unique_dataset

def extract_json(text: str) -> str:
    """Extracts the first JSON array found in the raw text output."""
    match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def generate_with_ollama(messages: list, model: str, temperature: float) -> str:
    """Handles text generation using the local Ollama REST API."""
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")
    except requests.RequestException as e:
        print(f"Ollama API request failed: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Iterative dataset generator for political rhetoric.")
    parser.add_argument("-e", "--engine", type=str, choices=["mlx", "ollama"], default="mlx", help="Inference engine to use.")
    parser.add_argument("-m", "--model", type=str, default="mlx-community/Qwen2.5-7B-Instruct-4bit", help="Model name (HuggingFace repo for MLX, or local model name for Ollama).")
    parser.add_argument("-t", "--total_iterations", type=int, default=20, help="Number of successful generation loops to complete.")
    parser.add_argument("-p", "--pairs_per_iter", type=int, default=3,help="Number of pairs requested per prompt.")
    parser.add_argument("-r", "--max_retries", type=int, default=3,help="Maximum attempts per iteration if JSON formatting fails.")
    parser.add_argument("-o", "--output", type=str, default="src/POLAR/data/pairs.json",help="Output JSON file path.")
    parser.add_argument("--temp", type=float, default=0.8, help="Generation temperature.")
    parser.add_argument("--verbose", "-v", action='store_true', help="display more information")
    
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mlx_model, mlx_tokenizer = None, None

    if args.engine == "mlx":
        try:
            from mlx_lm import load
            print(blue(f"Loading MLX model {args.model} into memory..."))
            mlx_model, mlx_tokenizer = load(args.model)
            print(green("Model loaded successfully."))
        except ImportError:
            print(red("Error: mlx-lm is not installed. Run 'pip install mlx-lm'."))
            sys.exit(1)

    system_prompt = "Tu es un expert en linguistique et en rhétorique politique française des années 1980 et 1990."
    
    dataset = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                dataset = json.load(f)
                print(f"Resuming existing dataset: {len(dataset)} pairs found.")
            except json.JSONDecodeError:
                print("Existing file is corrupted. Starting fresh.")

    for i in range(args.total_iterations):
        print(f"\n--- Iteration {i+1}/{args.total_iterations} ---")
        
        user_prompt = f"""Génère {args.pairs_per_iter} paires de phrases inédites. 
        La phrase A doit être une affirmation politique brute et directe.
        La phrase B doit être la traduction EXACTE de la phrase A en 'langue de bois' politique française typique des années 80/90 (évasif, abstrait, vocabulaire technocratique).
        
        Renvoie **UNIQUEMENT un tableau JSON valide** avec les clés 'direct' et 'langue_de_bois'. Ne mets aucun texte avant ou après le tableau.
        Les phrases que tu livre doivent être **uniquement** en français. 
        Exemple attendu :
        [
        {{
            "direct": "Nous allons licencier 500 personnes.",
            "langue_de_bois": "Nous devons engager une dynamique de sauvegarde de la compétitivité à travers un plan d'adaptation des effectifs."
        }}
        ]"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        iteration_success = False

        for attempt in range(args.max_retries):
            print(f"Attempt {attempt + 1}/{args.max_retries}...")
            response_text = ""

            if args.engine == "mlx":
                from mlx_lm import generate
                from mlx_lm.sample_utils import make_sampler
                prompt = mlx_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                sampler = make_sampler(temp=args.temp)
                response_text = generate(
                    mlx_model, 
                    mlx_tokenizer, 
                    prompt=prompt, 
                    max_tokens=1500, 
                    verbose=args.verbose,
                    sampler=sampler
                )
                if args.verbose:
                    print(response_text)
            elif args.engine == "ollama":
                response_text = generate_with_ollama(messages, args.model, args.temp)
                if args.verbose:
                    print(response_text)
            clean_json_str = extract_json(response_text)
            
            if clean_json_str:
                try:
                    new_pairs = json.loads(clean_json_str)
                    dataset.extend(new_pairs)
                    dataset = remove_duplicates(dataset)
                    print(f"Success: {len(new_pairs)} pairs added (Total: {len(dataset)}).")
                    
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(dataset, f, ensure_ascii=False, indent=4)
                    
                    iteration_success = True
                    break 
                
                except json.JSONDecodeError:
                    print(orange(f"Warning: JSON parsing failed. \n returned {response_text} \n Retrying."))
            else:
                print(red("Warning: No JSON format detected in response. Retrying."))
                
        if not iteration_success:
            print(orange(f"Failed to generate valid data for iteration {i+1} after {args.max_retries} attempts. Moving to next iteration."))

    print(green(f"\nGeneration complete. {len(dataset)} pairs available in {args.output}"))

if __name__ == "__main__":
    main()