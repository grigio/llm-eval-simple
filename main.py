import argparse
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
import re
from pathlib import Path

from dotenv import load_dotenv

from shared import GENERATED_ANSWERS_DIR, EVALUATED_REPORT_PATH, RAW_REPORT_PATH
from api_client import get_model_response, evaluate_correctness
from file_utils import get_prompt_files, read_file_content, save_raw_results, load_raw_results, save_evaluated_results, load_evaluated_results, ensure_directory_exists
from reporting import format_detailed_table, format_matrix_table, format_summary_table
from validation import ConfigValidation, FileOperationValidator, validate_glob_pattern, validate_model_list

# --- Constants ---

# Default configuration values
DEFAULT_PROMPT_DIR = "prompts"
DEFAULT_ANSWER_DIR = "answers"
DEFAULT_ENDPOINT_URL = "http://localhost:9292/v1/chat/completions"
DEFAULT_MODEL_EVALUATOR = "some-quite-powerful-model-8B"
DEFAULT_PATTERN = "*"
DEFAULT_ACTIONS = ["answer", "evaluate", "render", "serve"]
DEFAULT_THROTTLING_SECS = 0.1

# --- Configuration ---

@dataclass
class Config:
    """Holds the configuration for the evaluation script."""
    prompt_dir: str = DEFAULT_PROMPT_DIR
    answer_dir: str = DEFAULT_ANSWER_DIR
    endpoint_url: str = DEFAULT_ENDPOINT_URL
    model_names: List[str] = field(default_factory=list)
    model_evaluator: str = DEFAULT_MODEL_EVALUATOR
    pattern: str = DEFAULT_PATTERN
    actions: List[str] = field(default_factory=list)
    api_key: str = None
    throttling_secs: float = DEFAULT_THROTTLING_SECS
    custom_report_json: str = None
    
    def validate(self) -> List[str]:
        """Validate configuration using Pydantic models."""
        try:
            # Use Pydantic for comprehensive validation
            config_validation = ConfigValidation(
                endpoint_url=self.endpoint_url,
                model_names=self.model_names,
                model_evaluator=self.model_evaluator,
                pattern=self.pattern,
                actions=self.actions,
                api_key=self.api_key,
                throttling_secs=self.throttling_secs,
                prompt_dir=self.prompt_dir,
                answer_dir=self.answer_dir
            )
            # If validation passes, return empty errors list
            return []
        except Exception as e:
            return [str(e)]

def load_config() -> Config:
    """Loads configuration from environment variables and command-line arguments."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Test models on prompts with optional filtering.")
    parser.add_argument('--pattern', type=str, default="prompts/*", help="Glob pattern to filter prompt files (e.g., '*CODE*')")
    parser.add_argument('--actions', type=str, default="answer,evaluate,render,serve", help="Comma-separated list of actions to perform (answer,evaluate,render,serve)")
    parser.add_argument('--report-json', type=str, help="Path to custom JSON report file (overrides default)")
    args = parser.parse_args()

    try:
        # Validate and parse pattern
        pattern = validate_glob_pattern(args.pattern)
        
        # Validate and parse model names
        model_names_str = os.getenv("MODEL_NAMES", "gemma-3-270m-it-Q4_K_M,Qwen3-8B-Q4_K_M")
        model_names = validate_model_list(model_names_str)
        
        # Parse and validate actions
        actions_str = args.actions
        actions = [action.strip() for action in actions_str.split(',') if action.strip()]
        
        # Parse throttling with validation
        throttling_str = os.getenv("THROTTLING_SECS", str(DEFAULT_THROTTLING_SECS))
        try:
            throttling_secs = float(throttling_str)
            if throttling_secs < 0:
                raise ValueError("throttling_secs must be non-negative")
        except ValueError as e:
            raise ValueError(f"Invalid THROTTLING_SECS: {e}")

        config = Config(
            prompt_dir=DEFAULT_PROMPT_DIR,
            answer_dir=DEFAULT_ANSWER_DIR,
            endpoint_url=os.getenv("ENDPOINT_URL", DEFAULT_ENDPOINT_URL),
            model_names=model_names,
            model_evaluator=os.getenv("MODEL_EVALUATOR", DEFAULT_MODEL_EVALUATOR),
            pattern=pattern,
            actions=actions,
            api_key=os.getenv("API_KEY"),
            throttling_secs=throttling_secs
        )
        
        # Store custom report path if provided
        if args.report_json:
            config.custom_report_json = args.report_json
        
        # Validate configuration using Pydantic
        errors = config.validate()
        if errors:
            print("❌ Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            exit(1)
        
        # Validate directories exist
        prompt_path = Path(config.prompt_dir)
        if not prompt_path.exists():
            print(f"ℹ️  Info: Prompt directory '{config.prompt_dir}' does not exist - will be created if needed")
        
        answer_path = Path(config.answer_dir)
        if not answer_path.exists():
            print(f"ℹ️  Info: Answer directory '{config.answer_dir}' does not exist - will be created if needed")
        
        return config
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected configuration error: {e}")
        exit(1)

# --- Action Functions ---


# --- Action Functions ---

def answer_prompt(prompt_path: str, model_name: str, config: Config) -> Dict[str, Any]:
    """Processes a single prompt file and returns the generated answer."""
    try:
        # Validate file paths for security
        validated_prompt_path = FileOperationValidator.validate_file_path(prompt_path, config.prompt_dir)
        base_name = os.path.basename(prompt_path)
        answer_path = os.path.join(config.answer_dir, base_name)
        validated_answer_path = FileOperationValidator.validate_file_path(answer_path, config.answer_dir)

        if not validated_answer_path.exists():
            print(f"Skipping {base_name}: No matching answer file found.")
            return None

        # Validate file sizes
        FileOperationValidator.validate_file_size(validated_prompt_path)
        FileOperationValidator.validate_file_size(validated_answer_path)

        prompt = read_file_content(prompt_path)
        expected_answer = read_file_content(answer_path)
        
        # Validate content length
        prompt = FileOperationValidator.validate_content_length(prompt, 10000)
        expected_answer = FileOperationValidator.validate_content_length(expected_answer, 10000)
        
    except ValueError as e:
        print(f"Skipping {prompt_path}: {e}")
        return None
    except Exception as e:
        print(f"Error validating files for {prompt_path}: {e}")
        return None

    start_time = time.time()
    try:
        response_json = get_model_response(config.endpoint_url, model_name, prompt, config.api_key, throttling_secs=config.throttling_secs)
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

    except Exception as e:
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
    
    save_raw_results(results)
    print(f"\nGenerated {len(results)} answers. Report saved to {RAW_REPORT_PATH}")

def evaluate(config: Config):
    """Evaluates the generated answers."""
    print("\n--- Starting Evaluation ---")
    try:
        results = load_raw_results()
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
            config.api_key,
            config.throttling_secs
        )
        result["correct"] = is_correct
        result["evaluator_model"] = config.model_evaluator
        evaluated_results.append(result)
        print(f"Evaluated: {result['file']} for {result['model']} -> {'Correct' if is_correct else 'Incorrect'}")

    save_evaluated_results(evaluated_results)
    print(f"\nEvaluation complete. Report saved to {EVALUATED_REPORT_PATH}")

def render(config: Config):
    """Renders the final report."""
    print("\n--- Rendering Report ---")
    try:
        results = load_evaluated_results()
    except FileNotFoundError:
        print(f"Error: {EVALUATED_REPORT_PATH} not found. Please run the 'evaluate' action first.")
        return
    
    print_summary(results)

def serve(config: Config):
    """Starts a web server to display the report."""
    print("\n--- Starting Web Server ---")
    # Note: this will block the terminal until the server is stopped (e.g. with Ctrl+C)
    os.system("uv run python server.py")


# --- Reporting ---

def print_summary(results: List[Dict[str, Any]]):
    """Prints the detailed and summary tables of the results."""
    if not results:
        print("No results to display.")
        return

    # Detailed table
    print("\nDetailed Results")
    print(format_detailed_table(results))

    # Matrix table: prompts as columns, models as rows
    print("\nMatrix Results")
    print(format_matrix_table(results))

    # Summary table
    print("\nModel Performance Summary")
    print(format_summary_table(results))

# --- Main Execution ---

def main():
    """Main function to run the model evaluation."""
    config = load_config()
    ensure_directory_exists(GENERATED_ANSWERS_DIR)

    if "answer" in config.actions:
        answer(config)
    if "evaluate" in config.actions:
        evaluate(config)
    if "render" in config.actions:
        render(config)
    if "serve" in config.actions:
        serve(config)

if __name__ == "__main__":
    main()
