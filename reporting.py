"""Reporting and formatting functions for LLM evaluation results."""

from typing import Any, Dict, List, Tuple
from tabulate import tabulate

from shared import (
    calculate_model_summary,
    create_progress_bar,
    format_accuracy,
    format_response_time,
    get_unique_prompts_and_models
)


def create_detailed_table_data(results: List[Dict[str, Any]]) -> List[List[str]]:
    """Create data for the detailed results table."""
    return [
        [r["model"], r["file"], "correct" if r["correct"] else "wrong", format_response_time(r["response_time"]), r.get("note", "")]
        for r in results
    ]


def create_matrix_table_data(results: List[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    """Create data for the matrix table (prompts as columns, models as rows)."""
    prompts, models = get_unique_prompts_and_models(results)
    
    header = ["Model"] + prompts
    table_data = []

    for model in models:
        row = [model]
        for prompt in prompts:
            found = False
            for r in results:
                if r["model"] == model and r["file"] == prompt:
                    row.append("correct" if r["correct"] else "wrong")
                    found = True
                    break
            if not found:
                row.append("unavailable")
        table_data.append(row)

    return header, table_data


def create_summary_table_data(results: List[Dict[str, Any]]) -> List[List[str]]:
    """Create data for the model performance summary table."""
    model_summary = calculate_model_summary(results)
    summary_table = []
    
    for model, stats in model_summary.items():
        total = stats["total"]
        correct = stats["correct"]
        total_time = stats["total_time"]
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0
        
        bar = create_progress_bar(accuracy)
        
        summary_table.append([
            model,
            f"{correct}/{total} ({format_accuracy(correct, total)}) [{bar}]",
            format_response_time(avg_time)
        ])
    
    return summary_table


def format_detailed_table(results: List[Dict[str, Any]]) -> str:
    """Format the detailed results table."""
    table_data = create_detailed_table_data(results)
    return tabulate(table_data, headers=["Model", "File", "Correct", "Response Time", "Note"], tablefmt="fancy_grid")


def format_matrix_table(results: List[Dict[str, Any]]) -> str:
    """Format the matrix table (prompts as columns, models as rows)."""
    header, table_data = create_matrix_table_data(results)
    return tabulate(table_data, headers=header, tablefmt="fancy_grid")


def format_summary_table(results: List[Dict[str, Any]]) -> str:
    """Format the model performance summary table."""
    table_data = create_summary_table_data(results)
    return tabulate(table_data, headers=["Model", "Correct", "Avg Response Time"], tablefmt="fancy_grid")


def calculate_all_summary_data(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate all summary data for results."""
    return {
        "detailed_table": create_detailed_table_data(results),
        "matrix_table": create_matrix_table_data(results),
        "summary_table": create_summary_table_data(results),
        "model_summary": calculate_model_summary(results),
        "unique_prompts_and_models": get_unique_prompts_and_models(results)
    }