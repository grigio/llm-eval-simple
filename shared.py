import json
import os
from typing import Any, Dict, List, Tuple

# Constants
GENERATED_ANSWERS_DIR = "answers-generated"
RAW_REPORT_PATH = os.path.join(GENERATED_ANSWERS_DIR, "report.json")
EVALUATED_REPORT_PATH = os.path.join(GENERATED_ANSWERS_DIR, "report-evaluated.json")
HTML_REPORT_PATH = os.path.join(GENERATED_ANSWERS_DIR, "report-evaluated.html")

# Server constants
DEFAULT_SERVER_PORT = 4000
TEMPLATE_PATH = "report_template.html"

# Color constants for HTML output
GOLD_COLOR = (255, 215, 0)
GREEN_COLOR = (0, 247, 0)
LIGHT_GREEN_COLOR = (245, 255, 245)

# Response time constants
RESPONSE_TIME_DECIMAL_PLACES = 2
ACCURACY_DECIMAL_PLACES = 1
PERCENTAGE_MULTIPLIER = 100
BAR_LENGTH_DIVISOR = 10

# Color constants
RGB_MAX = 255
HSL_LIGHTNESS_MIN = 70
HSL_LIGHTNESS_RANGE = 30

# HTML/CSS constants
GOLD_RGB = (255, 215, 0)
GREEN_RGB = (0, 247, 0)
LIGHT_GREEN_RGB = (245, 255, 245)

# Time constants
INFINITE_TIME = float('inf')
DEFAULT_PROGRESS_BAR_LENGTH = 10


def calculate_model_summary(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Calculate summary statistics for each model."""
    model_summary = {}
    for r in results:
        model = r["model"]
        if model not in model_summary:
            model_summary[model] = {"total": 0, "correct": 0, "total_time": 0}
        
        model_summary[model]["total"] += 1
        if r["correct"]:
            model_summary[model]["correct"] += 1
        model_summary[model]["total_time"] += r["response_time"]
    
    return model_summary


def get_unique_prompts_and_models(results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Get sorted lists of unique prompts and models from results."""
    prompts = sorted(list(set(r["file"] for r in results)))
    models = sorted(list(set(r["model"] for r in results)))
    return prompts, models


def format_response_time(response_time: float) -> str:
    """Format response time with consistent decimal places."""
    return f"{response_time:.{RESPONSE_TIME_DECIMAL_PLACES}f}s"


def format_accuracy(correct: int, total: int) -> str:
    """Format accuracy as percentage with consistent decimal places."""
    accuracy = (correct / total) * PERCENTAGE_MULTIPLIER if total > 0 else 0
    return f"{accuracy:.{ACCURACY_DECIMAL_PLACES}f}%"


def create_progress_bar(accuracy: float, length: int = DEFAULT_PROGRESS_BAR_LENGTH) -> str:
    """Create a text-based progress bar."""
    bar_length = int(accuracy / BAR_LENGTH_DIVISOR)
    return "█" * bar_length + "░" * (length - bar_length)


def normalize_time_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize a time value between 0 and 1."""
    time_range = max_val - min_val if max_val != min_val else 1
    return (value - min_val) / time_range


def interpolate_color(color1: tuple, color2: tuple, factor: float) -> tuple:
    """Interpolate between two RGB colors."""
    r = int(color1[0] + (color2[0] - color1[0]) * factor)
    g = int(color1[1] + (color2[1] - color1[1]) * factor)
    b = int(color1[2] + (color2[2] - color1[2]) * factor)
    return (r, g, b)


def find_fastest_correct_per_prompt(results: List[Dict[str, Any]], prompts: List[str]) -> Dict[str, str]:
    """Find the fastest correct model for each prompt."""
    fastest_correct_per_prompt = {}
    for prompt in prompts:
        fastest_time = INFINITE_TIME
        fastest_model = None
        for r in results:
            if r["file"] == prompt and r["correct"] and r["response_time"] < fastest_time:
                fastest_time = r["response_time"]
                fastest_model = r["model"]
        if fastest_model:
            fastest_correct_per_prompt[prompt] = fastest_model
    return fastest_correct_per_prompt


def group_results_by_file(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group results by file for detailed display."""
    results_by_file = {}
    for r in results:
        if r['file'] not in results_by_file:
            results_by_file[r['file']] = {
                "prompt": r['prompt'],
                "expected": r['expected'],
                "models": []
            }
        results_by_file[r['file']]['models'].append(r)
    return results_by_file


def create_cell_data_dict(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create cell data dictionary for JavaScript interactions."""
    cell_data_dict = {}
    for r in results:
        cell_id = f"{r['model']}-{r['file']}"
        cell_data_dict[cell_id] = {
            "model": r["model"],
            "file": r["file"],
            "generated": r["generated"],
            "response_time": format_response_time(r['response_time']),
            "correct": r["correct"]
        }
    return cell_data_dict