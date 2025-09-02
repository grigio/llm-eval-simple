import os
import glob
import time
import requests
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any
from tabulate import tabulate
from dotenv import load_dotenv

# --- Configuration ---

@dataclass
class Config:
    """Holds the configuration for the evaluation script."""
    prompt_dir: str = "prompts"
    answer_dir: str = "answers"
    endpoint_url: str = "http://localhost:9292/v1/chat/completions"
    model_names: List[str] = field(default_factory=list)
    model_evaluator: str = "some-quite-powerful-model-8B"
    pattern: str = "*"

def load_config() -> Config:
    """Loads configuration from environment variables and command-line arguments."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Test models on prompts with optional filtering.")
    parser.add_argument('--pattern', type=str, default="*", help="Glob pattern to filter prompt files (e.g., '*CODE*')")
    args = parser.parse_args()

    model_names_str = os.getenv("MODEL_NAMES", "gemma-3-270m-it-Q4_K_M,Qwen3-8B-Q4_K_M")
    model_names = [name.strip() for name in model_names_str.split(",") if name.strip()]

    return Config(
        prompt_dir="prompts",
        answer_dir="answers",
        endpoint_url=os.getenv("ENDPOINT_URL", "http://localhost:9292/v1/chat/completions"),
        model_names=model_names,
        model_evaluator=os.getenv("MODEL_EVALUATOR", "some-quite-powerful-model-8B"),
        pattern=args.pattern
    )

# --- API Interaction ---

def get_model_response(endpoint_url: str, model: str, prompt: str) -> Dict[str, Any]:
    """Gets a response from the specified model."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(endpoint_url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

# --- Evaluation ---

def evaluate_correctness(endpoint_url: str, evaluator_model: str, expected_answer: str, generated_answer: str) -> bool:
    """Evaluates the correctness of a generated answer using an evaluator model."""
    if not evaluator_model:
        return generated_answer.lower() == expected_answer.lower()

    system_prompt = "You are an evaluator. Compare the expected answer with the generated answer, ignore the tag <think> content, the generated answers may vary slightly in wording but should preserve the original meaning, and respond with only 'CORRECT' or 'INCORRECT'"
    user_prompt = f"Expected Answer: {expected_answer}\nGenerated Answer: {generated_answer}"
    
    try:
        eval_response = get_model_response(endpoint_url, evaluator_model, user_prompt)
        eval_result = eval_response.get('choices', [{}])[0].get('message', {}).get('content', '').strip().lower()
        
        if "incorrect" in eval_result or "not correct" in eval_result:
            return False
        if "correct" in eval_result:
            return True
        return False  # Default to incorrect if unclear
    except requests.exceptions.RequestException as e:
        print(f"Evaluator error: {str(e)}")
        return False

# --- File Handling ---

def get_prompt_files(prompt_dir: str, pattern: str) -> List[str]:
    """Gets a sorted list of prompt files matching the pattern."""
    return sorted(glob.glob(os.path.join(prompt_dir, pattern)))

# --- Result Processing ---

def process_prompt(prompt_path: str, model_name: str, config: Config) -> Dict[str, Any]:
    """Processes a single prompt file and returns the result."""
    base_name = os.path.basename(prompt_path)
    answer_path = os.path.join(config.answer_dir, base_name)

    if not os.path.exists(answer_path):
        print(f"Skipping {base_name}: No matching answer file found.")
        return None

    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()
    with open(answer_path, 'r', encoding='utf-8') as f:
        expected_answer = f.read().strip()

    start_time = time.time()
    try:
        response_json = get_model_response(config.endpoint_url, model_name, prompt)
        end_time = time.time()
        
        generated_answer = response_json.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        is_correct = evaluate_correctness(config.endpoint_url, config.model_evaluator, expected_answer, generated_answer)

        result = {
            "model": model_name,
            "file": base_name,
            "correct": is_correct,
            "response_time": end_time - start_time,
            "expected": expected_answer,
            "generated": generated_answer
        }
        
        print(f"File: {base_name}")
        print(f"Response Time: {result['response_time']:.2f} seconds")
        print(f"Correct: {is_correct}")
        if not is_correct:
            print(f"Expected: {expected_answer}")
            print(f"Generated: {generated_answer}")
        print("-" * 40)
        
        return result

    except requests.exceptions.RequestException as e:
        end_time = time.time()
        print(f"Error for {base_name}: {str(e)}")
        print(f"Time elapsed before error: {end_time - start_time:.2f} seconds")
        print("-" * 40)
        return None

# --- Reporting ---

def print_summary(results: List[Dict[str, Any]]):
    """Prints the detailed and summary tables of the results."""
    if not results:
        print("No results to display.")
        return

    # Detailed table
    detailed_table = [
        [r["model"], r["file"], "🮱" if r["correct"] else "𐄂", f"{r['response_time']:.2f}s"]
        for r in results
    ]
    print("\nDetailed Results")
    print(tabulate(detailed_table, headers=["Model", "File", "Correct", "Response Time"], tablefmt="fancy_grid"))

    # Summary table
    model_summary = {}
    for r in results:
        model = r["model"]
        if model not in model_summary:
            model_summary[model] = {"total": 0, "correct": 0, "total_time": 0}
        
        model_summary[model]["total"] += 1
        if r["correct"]:
            model_summary[model]["correct"] += 1
        model_summary[model]["total_time"] += r["response_time"]

    summary_table = []
    for model, stats in model_summary.items():
        total = stats["total"]
        correct = stats["correct"]
        total_time = stats["total_time"]
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0
        
        bar_length = int(accuracy / 10)
        bar = "█" * bar_length + "░" * (10 - bar_length)
        
        summary_table.append([
            model,
            f"{correct}/{total} ({accuracy:.1f}%) [{bar}]",
            f"{avg_time:.2f}s"
        ])

    print("\nModel Performance Summary")
    print(tabulate(summary_table, headers=["Model", "Correct", "Avg Response Time"], tablefmt="fancy_grid"))

# --- Main Execution ---

def main():
    """Main function to run the model evaluation."""
    config = load_config()
    prompt_files = get_prompt_files(config.prompt_dir, config.pattern)
    results = []

    for i, model_name in enumerate(config.model_names):
        emoji = "🚀" if i == 0 else "🔬"
        print(f"\n{emoji} Testing model: {model_name}")
        print("𑁋" + "𑁋" * 48)
        
        for prompt_path in prompt_files:
            result = process_prompt(prompt_path, model_name, config)
            if result:
                results.append(result)
    
    print_summary(results)

if __name__ == "__main__":
    main()
