"""File handling functions for LLM evaluation."""

import glob
import json
import os
from typing import List, Dict, Any
from pathlib import Path

from shared import RAW_REPORT_PATH, EVALUATED_REPORT_PATH
from validation import FileOperationValidator, validate_glob_pattern


def get_prompt_files(pattern: str) -> List[str]:
    """Gets a sorted list of prompt files matching the pattern."""
    try:
        # Validate pattern for security
        validated_pattern = validate_glob_pattern(pattern)
        
        files = glob.glob(validated_pattern)
        valid_files = []
        
        for file_path in files:
            try:
                # Validate each file path
                validated_path = FileOperationValidator.validate_file_path(file_path)
                
                # Check if it's a file and has allowed extension
                if validated_path.is_file():
                    FileOperationValidator.validate_file_size(validated_path)
                    valid_files.append(str(validated_path))
                    
            except ValueError:
                # Skip invalid files
                continue
        
        return sorted(valid_files)
        
    except ValueError as e:
        print(f"❌ Error in file pattern validation: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error getting prompt files: {e}")
        return []


def read_file_content(file_path: str) -> str:
    """Read content from a file with UTF-8 encoding and validation."""
    try:
        # Validate file path
        validated_path = FileOperationValidator.validate_file_path(file_path)
        
        # Validate file size
        FileOperationValidator.validate_file_size(validated_path)
        
        # Read content with size limit
        with open(validated_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Validate content length
        return FileOperationValidator.validate_content_length(content, 50000).strip()
        
    except UnicodeDecodeError:
        raise ValueError(f"File {file_path} is not valid UTF-8 text")
    except ValueError as e:
        raise ValueError(f"File validation error for {file_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {e}")


def write_json_report(data: List[Dict[str, Any]], file_path: str) -> None:
    """Write data to a JSON file with proper formatting and validation."""
    try:
        # Validate file path
        validated_path = FileOperationValidator.validate_file_path(file_path)
        
        # Ensure parent directory exists
        validated_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate data size
        json_str = json.dumps(data, indent=2)
        if len(json_str.encode('utf-8')) > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError("Data too large to write")
        
        # Write atomically to prevent corruption
        temp_path = validated_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # Atomic move
        temp_path.replace(validated_path)
        
    except ValueError as e:
        raise ValueError(f"Validation error writing {file_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error writing file {file_path}: {e}")


def read_json_report(file_path: str) -> List[Dict[str, Any]]:
    """Read data from a JSON report file with validation."""
    try:
        # Validate file path
        validated_path = FileOperationValidator.validate_file_path(file_path)
        
        # Validate file size
        FileOperationValidator.validate_file_size(validated_path)
        
        # Read and parse JSON
        with open(validated_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate data structure
        if not isinstance(data, list):
            raise ValueError("JSON data must be a list")
        
        # Validate each item
        validated_data = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} must be a dictionary")
            validated_data.append(item)
        
        return validated_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")
    except ValueError as e:
        raise ValueError(f"Validation error reading {file_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {e}")


def ensure_directory_exists(directory: str) -> None:
    """Ensure that a directory exists, creating it if necessary."""
    try:
        # Validate directory path
        validated_path = FileOperationValidator.validate_file_path(directory)
        
        # Create directory with parents
        validated_path.mkdir(parents=True, exist_ok=True)
        
    except ValueError as e:
        raise ValueError(f"Invalid directory path {directory}: {e}")
    except Exception as e:
        raise ValueError(f"Error creating directory {directory}: {e}")


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