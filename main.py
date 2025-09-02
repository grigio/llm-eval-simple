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
    actions: List[str] = field(default_factory=list)
    api_key: str = None

def load_config() -> Config:
    """Loads configuration from environment variables and command-line arguments."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Test models on prompts with optional filtering.")
    parser.add_argument('--pattern', type=str, default="prompts/*", help="Glob pattern to filter prompt files (e.g., '*CODE*')")
    parser.add_argument('--actions', type=str, default="answer,evaluate,render", help="Comma-separated list of actions to perform (answer,evaluate,render)")
    args = parser.parse_args()

    model_names_str = os.getenv("MODEL_NAMES", "gemma-3-270m-it-Q4_K_M,Qwen3-8B-Q4_K_M")
    model_names = [name.strip() for name in model_names_str.split(",") if name.strip()]

    return Config(
        prompt_dir="prompts",
        answer_dir="answers",
        endpoint_url=os.getenv("ENDPOINT_URL", "http://localhost:9292/v1/chat/completions"),
        model_names=model_names,
        model_evaluator=os.getenv("MODEL_EVALUATOR", "some-quite-powerful-model-8B"),
        pattern=args.pattern,
        actions=[action.strip() for action in args.actions.split(',')],
        api_key=os.getenv("API_KEY")
    )

# --- API Interaction ---

def get_model_response(endpoint_url: str, model: str, prompt: str, system_prompt: str = None, api_key: str = None) -> Dict[str, Any]:
    """Gets a response from the specified model."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    response = requests.post(endpoint_url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

# --- Evaluation ---

def evaluate_correctness(endpoint_url: str, evaluator_model: str, expected_answer: str, generated_answer: str, api_key: str = None) -> bool:
    """Evaluates the correctness of a generated answer using an evaluator model."""
    if not evaluator_model:
        return generated_answer.lower() == expected_answer.lower()

    system_prompt = "You are an evaluator. Compare the expected answer with the generated answer. Ignore the tag  content. The generated answers may vary slightly in wording but should preserve the original meaning. If the answers are equivalent in meaning, mark as correct. Respond with only 'CORRECT' or 'INCORRECT'."
    user_prompt = f"Expected Answer: {expected_answer}\nGenerated Answer: {generated_answer}"
    
    try:
        eval_response = get_model_response(endpoint_url, evaluator_model, user_prompt, system_prompt, api_key)
        eval_result = eval_response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        # NOTE: keep this to debug evaluator model
        # print(eval_result)

        # More flexible evaluation - look for clear indicators of correctness
        if "CORRECT" == eval_result:
            return True
        if "INCORRECT" == eval_result:
            return False

        return False
    except requests.exceptions.RequestException as e:
        print(f"Evaluator error: {str(e)}")
        return False

# --- File Handling ---

def get_prompt_files(pattern: str) -> List[str]:
    """Gets a sorted list of prompt files matching the pattern."""
    return sorted(glob.glob(pattern))

# --- Constants ---
GENERATED_ANSWERS_DIR = "answers-generated"
RAW_REPORT_PATH = os.path.join(GENERATED_ANSWERS_DIR, "report.json")
EVALUATED_REPORT_PATH = os.path.join(GENERATED_ANSWERS_DIR, "report-evaluated.json")


# --- Action Functions ---

def answer_prompt(prompt_path: str, model_name: str, config: Config) -> Dict[str, Any]:
    """Processes a single prompt file and returns the generated answer."""
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
        response_json = get_model_response(config.endpoint_url, model_name, prompt, api_key=config.api_key)
        end_time = time.time()
        
        generated_answer = response_json.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        result = {
            "model": model_name,
            "file": base_name,
            "prompt": prompt,
            "response_time": end_time - start_time,
            "expected": expected_answer,
            "generated": generated_answer
        }
        
        print(f"Answered: {base_name} with {model_name}")
        return result

    except requests.exceptions.RequestException as e:
        print(f"Error for {base_name} with {model_name}: {str(e)}")
        return None

def answer(config: Config):
    """Generates answers for all prompts and models."""
    print("--- Starting Answer Generation ---")
    prompt_files = get_prompt_files(config.pattern)
    results = []

    for model_name in config.model_names:
        print(f"\n🔬 Testing model: {model_name}")
        for prompt_path in prompt_files:
            result = answer_prompt(prompt_path, model_name, config)
            if result:
                results.append(result)
    
    with open(RAW_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGenerated {len(results)} answers. Report saved to {RAW_REPORT_PATH}")

def evaluate(config: Config):
    """Evaluates the generated answers."""
    print("\n--- Starting Evaluation ---")
    try:
        with open(RAW_REPORT_PATH, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {RAW_REPORT_PATH} not found. Please run the 'answer' action first.")
        return

    evaluated_results = []
    for result in results:
        is_correct = evaluate_correctness(
            config.endpoint_url, 
            config.model_evaluator, 
            result["expected"], 
            result["generated"],
            config.api_key
        )
        result["correct"] = is_correct
        result["evaluator_model"] = config.model_evaluator
        evaluated_results.append(result)
        print(f"Evaluated: {result['file']} for {result['model']} -> {'Correct' if is_correct else 'Incorrect'}")

    with open(EVALUATED_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(evaluated_results, f, indent=2)

    print(f"\nEvaluation complete. Report saved to {EVALUATED_REPORT_PATH}")

def render(config: Config):
    """Renders the final report."""
    print("\n--- Rendering Report ---")
    try:
        with open(EVALUATED_REPORT_PATH, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {EVALUATED_REPORT_PATH} not found. Please run the 'evaluate' action first.")
        return
    
    print_summary(results)

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
    os.makedirs(GENERATED_ANSWERS_DIR, exist_ok=True)

    if "answer" in config.actions:
        answer(config)
    if "evaluate" in config.actions:
        evaluate(config)
    if "render" in config.actions:
        render(config)

if __name__ == "__main__":
    main()
