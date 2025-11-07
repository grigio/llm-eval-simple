"""Test configuration and fixtures."""

import pytest
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, Any, List

from unittest.mock import Mock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_prompt_file(temp_dir):
    """Create a sample prompt file."""
    prompt_file = temp_dir / "test_prompt.txt"
    prompt_file.write_text("What is 2 + 2?", encoding='utf-8')
    return prompt_file


@pytest.fixture
def sample_answer_file(temp_dir):
    """Create a sample answer file."""
    answer_file = temp_dir / "test_answer.txt"
    answer_file.write_text("4", encoding='utf-8')
    return answer_file


@pytest.fixture
def sample_json_file(temp_dir):
    """Create a sample JSON file."""
    json_file = temp_dir / "test_data.json"
    data = [
        {
            "model": "test-model",
            "file": "test.txt",
            "prompt": "Test prompt",
            "expected": "Expected answer",
            "generated": "Generated answer",
            "correct": True,
            "response_time": 1.5
        }
    ]
    json_file.write_text(json.dumps(data), encoding='utf-8')
    return json_file


@pytest.fixture
def sample_results():
    """Sample evaluation results."""
    return [
        {
            "model": "model-a",
            "file": "test1.txt",
            "prompt": "What is 1+1?",
            "expected": "2",
            "generated": "2",
            "correct": True,
            "response_time": 1.0
        },
        {
            "model": "model-b",
            "file": "test1.txt",
            "prompt": "What is 1+1?",
            "expected": "2",
            "generated": "3",
            "correct": False,
            "response_time": 2.0
        },
        {
            "model": "model-a",
            "file": "test2.txt",
            "prompt": "What is 2+2?",
            "expected": "4",
            "generated": "4",
            "correct": True,
            "response_time": 1.5
        }
    ]


@pytest.fixture
def mock_api_response():
    """Mock OpenAI-compatible API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Test response",
                    "role": "assistant"
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        },
        "model": "test-model"
    }


@pytest.fixture
def mock_requests_response(mock_api_response):
    """Mock requests.Response object."""
    mock_resp = Mock()
    mock_resp.json.return_value = mock_api_response
    mock_resp.raise_for_status.return_value = None
    mock_resp.status_code = 200
    return mock_resp


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "endpoint_url": "http://localhost:9292/v1/chat/completions",
        "model_names": ["model-a", "model-b"],
        "model_evaluator": "evaluator-model",
        "pattern": "prompts/*",
        "actions": ["answer", "evaluate"],
        "api_key": "test-key",
        "throttling_secs": 0.1,
        "prompt_dir": "prompts",
        "answer_dir": "answers"
    }


@pytest.fixture
def large_content():
    """Large content for testing size limits."""
    return "x" * 100000  # 100KB


@pytest.fixture
def invalid_json_file(temp_dir):
    """Create an invalid JSON file."""
    invalid_file = temp_dir / "invalid.json"
    invalid_file.write_text("{ invalid json content", encoding='utf-8')
    return invalid_file


@pytest.fixture
def empty_file(temp_dir):
    """Create an empty file."""
    empty = temp_dir / "empty.txt"
    empty.write_text("", encoding='utf-8')
    return empty


@pytest.fixture
def oversized_file(temp_dir):
    """Create an oversized file for testing limits."""
    oversized = temp_dir / "oversized.txt"
    # Create a file larger than the limit (assuming 10MB limit)
    oversized.write_text("x" * (11 * 1024 * 1024), encoding='utf-8')
    return oversized