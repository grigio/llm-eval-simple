"""File handling functions for LLM evaluation."""

import glob
import json
import os
from typing import List, Dict, Any

from shared import RAW_REPORT_PATH, EVALUATED_REPORT_PATH


def get_prompt_files(pattern: str) -> List[str]:
    """Gets a sorted list of prompt files matching the pattern."""
    files = glob.glob(pattern)
    return sorted([f for f in files if os.path.isfile(f)])


def read_file_content(file_path: str) -> str:
    """Read content from a file with UTF-8 encoding."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def write_json_report(data: List[Dict[str, Any]], file_path: str) -> None:
    """Write data to a JSON file with proper formatting."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def read_json_report(file_path: str) -> List[Dict[str, Any]]:
    """Read data from a JSON report file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_directory_exists(directory: str) -> None:
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)


def save_raw_results(results: List[Dict[str, Any]]) -> None:
    """Save raw results to the standard report path."""
    write_json_report(results, RAW_REPORT_PATH)


def load_raw_results() -> List[Dict[str, Any]]:
    """Load raw results from the standard report path."""
    return read_json_report(RAW_REPORT_PATH)


def save_evaluated_results(results: List[Dict[str, Any]]) -> None:
    """Save evaluated results to the standard report path."""
    write_json_report(results, EVALUATED_REPORT_PATH)


def load_evaluated_results() -> List[Dict[str, Any]]:
    """Load evaluated results from the standard report path."""
    return read_json_report(EVALUATED_REPORT_PATH)