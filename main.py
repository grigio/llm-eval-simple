import os
import glob
import time
import requests
import json
from tabulate import tabulate
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file if it exists
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Test models on prompts with optional filtering.")
parser.add_argument('--pattern', type=str, default="*", help="Glob pattern to filter prompt files (e.g., '*CODE*')")
args = parser.parse_args()

# Configuration - load from environment variables or use defaults
prompt_dir = "prompts"
answer_dir = "answers"
endpoint_url = os.getenv("ENDPOINT_URL", "http://localhost:9292/v1/chat/completions")
model_names_str = os.getenv("MODEL_NAMES", "gemma-3-270m-it-Q4_K_M,Qwen3-8B-Q4_K_M")
model_evaluator = os.getenv("MODEL_EVALUATOR", "some-quite-powerful-model-8B")

# Convert comma-separated model names string to list
model_names = [name.strip() for name in model_names_str.split(",") if name.strip()]

# List prompt files based on the pattern
prompt_files = sorted(glob.glob(os.path.join(prompt_dir, args.pattern)))

# Results table
results = []

for model_name in model_names:
    print(f"\nTesting model: {model_name}")
    print("=" * 50)
    
    # Test all prompts with current model
    for prompt_path in prompt_files:
        if not os.path.isfile(prompt_path):
            continue
        
        base_name = os.path.basename(prompt_path)
        answer_path = os.path.join(answer_dir, base_name)
        
        if not os.path.exists(answer_path) or not os.path.isfile(answer_path):
            print(f"Skipping {base_name}: No matching answer file found.")
            continue
        
        # Read prompt and expected answer
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        with open(answer_path, 'r', encoding='utf-8') as f:
            expected_answer = f.read().strip()
        
        # Prepare payload for OpenAI-compatible endpoint
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        
        # Measure response time
        start_time = time.time()
        try:
            response = requests.post(endpoint_url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()  # Raise error for bad status codes
            end_time = time.time()
            
            # Parse response
            resp_json = response.json()
            generated_answer = resp_json.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            
            # Evaluate correctness
            if model_evaluator:
                eval_payload = {
                    "model": model_evaluator,
                    "messages": [
                        {"role": "system", "content": "You are an evaluator. Compare the expected answer with the generated answer, ignore the tag <think> content, the generated answers may vary slightly in wording but should preserve the original meaning, and respond with only 'CORRECT' or 'INCORRECT'"},
                        {"role": "user", "content": f"Expected Answer: {expected_answer}\nGenerated Answer: {generated_answer}"}
                    ],
                    "stream": False
                }
                try:
                    eval_response = requests.post(endpoint_url, json=eval_payload, headers={"Content-Type": "application/json"})
                    eval_response.raise_for_status()
                    eval_resp_json = eval_response.json()
                    eval_result = eval_resp_json.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                    eval_result_lower = eval_result.lower()
                    if "incorrect" in eval_result_lower or "not correct" in eval_result_lower:
                        is_correct = False
                    elif "correct" in eval_result_lower:
                        is_correct = True
                    else:
                        is_correct = False  # Default to incorrect if unclear
                except requests.exceptions.RequestException as eval_e:
                    print(f"Evaluator error for {base_name}: {str(eval_e)}")
                    is_correct = False  # Consider incorrect if evaluator fails
            else:
                # Fallback to exact match if no evaluator is specified
                is_correct = generated_answer.lower() == expected_answer.lower()

            # Store results
            results.append({
                "model": model_name,
                "file": base_name,
                "correct": is_correct,
                "response_time": end_time - start_time
            })
            
            # Output results
            print(f"File: {base_name}")
            print(f"Response Time: {end_time - start_time:.2f} seconds")
            print(f"Correct: {is_correct}")
            if not is_correct:
                print(f"Expected: {expected_answer}")
                print(f"Generated: {generated_answer}")
            print("-" * 40)
        
        except requests.exceptions.RequestException as e:
            end_time = time.time()
            print(f"Error for {base_name}: {str(e)}")
            print(f"Time elapsed before error: {end_time - start_time:.2f} seconds")
            print("-" * 40)

# Prepare detailed table
detailed_table = []
for result in results:
    detailed_table.append([
        result["model"],
        result["file"],
        "Yes" if result["correct"] else "No",
        f"{result['response_time']:.2f}s"
    ])

# Print detailed table
print("\nDetailed Results")
print(tabulate(detailed_table, headers=["Model", "File", "Correct", "Response Time"], tablefmt="grid"))

# Aggregate results by model
model_summary = {}
for result in results:
    model = result["model"]
    if model not in model_summary:
        model_summary[model] = {
            "total": 0,
            "correct": 0,
            "total_time": 0
        }
    
    model_summary[model]["total"] += 1
    if result["correct"]:
        model_summary[model]["correct"] += 1
    model_summary[model]["total_time"] += result["response_time"]

# Prepare summary table
summary_table = []
for model, stats in model_summary.items():
    accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    avg_time = stats["total_time"] / stats["total"] if stats["total"] > 0 else 0
    summary_table.append([
        model,
        f"{stats['correct']}/{stats['total']} ({accuracy:.1f}%)",
        f"{avg_time:.2f}s"
    ])

# Print summary table
print("\nModel Performance Summary")
print(tabulate(summary_table, headers=["Model", "Correct", "Avg Response Time"], tablefmt="grid"))