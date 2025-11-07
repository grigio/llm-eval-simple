"""Unit tests for main configuration and loading."""

import pytest
import os
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
from dataclasses import fields

from main import Config, load_config, answer_prompt


class TestConfig:
    """Test Config dataclass."""
    
    def test_config_default_values(self):
        """Test config default values."""
        config = Config()
        
        assert config.prompt_dir == "prompts"
        assert config.answer_dir == "answers"
        assert config.model_names == []
        assert config.actions == []
        assert config.throttling_secs == 0.1
        assert config.api_key is None
    
    def test_config_custom_values(self):
        """Test config with custom values."""
        config = Config(
            prompt_dir="custom_prompts",
            answer_dir="custom_answers",
            model_names=["model1", "model2"],
            actions=["answer"],
            throttling_secs=0.5,
            api_key="test-key"
        )
        
        assert config.prompt_dir == "custom_prompts"
        assert config.answer_dir == "custom_answers"
        assert config.model_names == ["model1", "model2"]
        assert config.actions == ["answer"]
        assert config.throttling_secs == 0.5
        assert config.api_key == "test-key"
    
    def test_config_validation_valid(self, sample_config):
        """Test config validation with valid data."""
        config = Config(**sample_config)
        errors = config.validate()
        
        assert errors == []
    
    def test_config_validation_invalid_url(self, sample_config):
        """Test config validation with invalid URL."""
        sample_config["endpoint_url"] = "invalid-url"
        config = Config(**sample_config)
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("URL" in error for error in errors)
    
    def test_config_validation_empty_model_names(self, sample_config):
        """Test config validation with empty model names."""
        sample_config["model_names"] = []
        config = Config(**sample_config)
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("model_names" in error for error in errors)
    
    def test_config_validation_invalid_actions(self, sample_config):
        """Test config validation with invalid actions."""
        sample_config["actions"] = ["invalid_action"]
        config = Config(**sample_config)
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("Invalid action" in error for error in errors)
    
    def test_config_validation_negative_throttling(self, sample_config):
        """Test config validation with negative throttling."""
        sample_config["throttling_secs"] = -1.0
        config = Config(**sample_config)
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("throttling_secs" in error for error in errors)


class TestLoadConfig:
    """Test load_config function."""
    
    @patch('main.os.getenv')
    @patch('main.argparse.ArgumentParser.parse_args')
    @patch('main.load_dotenv')
    def test_load_config_defaults(self, mock_load_dotenv, mock_args, mock_getenv):
        """Test loading config with defaults."""
        # Mock command line arguments
        mock_args.return_value = MagicMock(
            pattern="prompts/*",
            actions="answer,evaluate,render,serve"
        )
        
        # Mock environment variables
        def mock_getenv_side_effect(key, default=None):
            env_vars = {
                "ENDPOINT_URL": "http://localhost:9292/v1/chat/completions",
                "MODEL_NAMES": "model-a,model-b",
                "MODEL_EVALUATOR": "evaluator-model",
                "THROTTLING_SECS": "0.1"
            }
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = mock_getenv_side_effect
        
        config = load_config()
        
        assert config.endpoint_url == "http://localhost:9292/v1/chat/completions"
        assert config.model_names == ["model-a", "model-b"]
        assert config.model_evaluator == "evaluator-model"
        assert config.throttling_secs == 0.1
        assert config.pattern == "prompts/*"
        assert config.actions == ["answer", "evaluate", "render", "serve"]
    
    @patch('main.os.getenv')
    @patch('main.argparse.ArgumentParser.parse_args')
    @patch('main.load_dotenv')
    def test_load_config_validation_error(self, mock_load_dotenv, mock_args, mock_getenv):
        """Test loading config with validation errors."""
        # Mock command line arguments
        mock_args.return_value = MagicMock(
            pattern="prompts/*",
            actions="answer,evaluate,render,serve"
        )
        
        # Mock environment variables with invalid data
        def mock_getenv_side_effect(key, default=None):
            env_vars = {
                "ENDPOINT_URL": "invalid-url",  # Invalid URL
                "MODEL_NAMES": "model-a,model-b",
                "MODEL_EVALUATOR": "evaluator-model",
                "THROTTLING_SECS": "0.1"
            }
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = mock_getenv_side_effect
        
        with pytest.raises(SystemExit):
            load_config()
    
    @patch('main.os.getenv')
    @patch('main.argparse.ArgumentParser.parse_args')
    @patch('main.load_dotenv')
    @patch('main.Path')
    def test_load_config_directory_warnings(self, mock_path, mock_load_dotenv, mock_args, mock_getenv):
        """Test directory existence warnings."""
        # Mock command line arguments
        mock_args.return_value = MagicMock(
            pattern="prompts/*",
            actions="answer,evaluate,render,serve"
        )
        
        # Mock environment variables
        def mock_getenv_side_effect(key, default=None):
            env_vars = {
                "ENDPOINT_URL": "http://localhost:9292/v1/chat/completions",
                "MODEL_NAMES": "model-a,model-b",
                "MODEL_EVALUATOR": "evaluator-model",
                "THROTTLING_SECS": "0.1"
            }
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = mock_getenv_side_effect
        
        # Mock Path.exists() to return False
        mock_path.return_value.exists.return_value = False
        
        config = load_config()
        
        # Should still create config despite missing directories
        assert config is not None
    
    @patch('main.os.getenv')
    @patch('main.argparse.ArgumentParser.parse_args')
    @patch('main.load_dotenv')
    def test_load_config_custom_report_json(self, mock_load_dotenv, mock_args, mock_getenv):
        """Test loading config with custom report JSON."""
        # Mock command line arguments
        mock_args.return_value = MagicMock(
            pattern="prompts/*",
            actions="answer,evaluate,render,serve",
            report_json="/custom/path/report.json"
        )
        
        # Mock environment variables
        def mock_getenv_side_effect(key, default=None):
            env_vars = {
                "ENDPOINT_URL": "http://localhost:9292/v1/chat/completions",
                "MODEL_NAMES": "model-a,model-b",
                "MODEL_EVALUATOR": "evaluator-model",
                "THROTTLING_SECS": "0.1"
            }
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = mock_getenv_side_effect
        
        config = load_config()
        
        assert config.custom_report_json == "/custom/path/report.json"
    
    @patch('main.os.getenv')
    @patch('main.argparse.ArgumentParser.parse_args')
    @patch('main.load_dotenv')
    def test_load_config_invalid_throttling(self, mock_load_dotenv, mock_args, mock_getenv):
        """Test loading config with invalid throttling value."""
        # Mock command line arguments
        mock_args.return_value = MagicMock(
            pattern="prompts/*",
            actions="answer,evaluate,render,serve"
        )
        
        # Mock environment variables with invalid throttling
        def mock_getenv_side_effect(key, default=None):
            env_vars = {
                "ENDPOINT_URL": "http://localhost:9292/v1/chat/completions",
                "MODEL_NAMES": "model-a,model-b",
                "MODEL_EVALUATOR": "evaluator-model",
                "THROTTLING_SECS": "invalid"
            }
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = mock_getenv_side_effect
        
        with pytest.raises(SystemExit):
            load_config()


class TestAnswerPrompt:
    """Test answer_prompt function."""
    
    @patch('main.read_file_content')
    @patch('main.get_model_response')
    @patch('main.FileOperationValidator.validate_file_path')
    @patch('main.FileOperationValidator.validate_file_size')
    @patch('main.FileOperationValidator.validate_content_length')
    def test_answer_prompt_success(self, mock_validate_content, mock_validate_size, mock_validate_path, mock_get_response, mock_read_content, sample_config):
        """Test successful prompt answering."""
        # Mock file reading
        mock_read_content.side_effect = [
            "Test prompt",  # prompt content
            "Expected answer"  # answer content
        ]
        
        # Mock content validation to return content unchanged
        mock_validate_content.side_effect = lambda content, max_len: content
        
        # Mock API response
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Generated answer"
                    }
                }
            ]
        }
        mock_get_response.return_value = mock_response
        
        config = Config(**sample_config)
        
        # Mock path validation to return valid paths
        mock_prompt_path = MagicMock()
        mock_prompt_path.exists.return_value = True
        mock_answer_path = MagicMock()
        mock_answer_path.exists.return_value = True
        mock_validate_path.side_effect = [mock_prompt_path, mock_answer_path]
        
        result = answer_prompt(
            "test_prompt.txt",
            "test-model",
            config
        )
        
        assert result is not None
        assert result["model"] == "test-model"
        assert result["file"] == "test_prompt.txt"
        assert result["prompt"] == "Test prompt"
        assert result["expected"] == "Expected answer"
        assert result["generated"] == "Generated answer"
        assert "response_time" in result
    
    @patch('main.read_file_content')
    @patch('main.get_model_response')
    @patch('main.FileOperationValidator.validate_file_path')
    @patch('main.FileOperationValidator.validate_file_size')
    @patch('main.FileOperationValidator.validate_content_length')
    def test_answer_prompt_api_error(self, mock_validate_content, mock_validate_size, mock_validate_path, mock_get_response, mock_read_content, sample_config):
        """Test prompt answering with API error."""
        # Mock file reading
        mock_read_content.side_effect = [
            "Test prompt",
            "Expected answer"
        ]
        
        # Mock content validation to return content unchanged
        mock_validate_content.side_effect = lambda content, max_len: content
        
        # Mock path validation to return valid paths
        mock_prompt_path = MagicMock()
        mock_prompt_path.exists.return_value = True
        mock_answer_path = MagicMock()
        mock_answer_path.exists.return_value = True
        mock_validate_path.side_effect = [mock_prompt_path, mock_answer_path]
        
        # Mock API error
        mock_get_response.side_effect = ValueError("API error")
        
        result = answer_prompt(
            "test_prompt.txt",
            "test-model",
            sample_config
        )
        
        assert result is None
    
    @patch('main.read_file_content')
    @patch('main.FileOperationValidator.validate_file_path')
    @patch('main.FileOperationValidator.validate_file_size')
    @patch('main.FileOperationValidator.validate_content_length')
    def test_answer_prompt_missing_answer_file(self, mock_validate_content, mock_validate_size, mock_validate_path, mock_read_content, sample_config):
        """Test prompt answering with missing answer file."""
        # Mock content validation to return content unchanged
        mock_validate_content.side_effect = lambda content, max_len: content
        
        # Mock path validation - answer path doesn't exist
        mock_prompt_path = MagicMock()
        mock_prompt_path.exists.return_value = True
        mock_answer_path = MagicMock()
        mock_answer_path.exists.return_value = False  # Answer file doesn't exist
        mock_validate_path.side_effect = [mock_prompt_path, mock_answer_path]
        
        result = answer_prompt(
            "test_prompt.txt",
            "test-model",
            sample_config
        )
        
        assert result is None
    
    @patch('main.read_file_content')
    @patch('main.get_model_response')
    @patch('main.FileOperationValidator.validate_file_path')
    @patch('main.FileOperationValidator.validate_file_size')
    @patch('main.FileOperationValidator.validate_content_length')
    def test_answer_prompt_with_api_key(self, mock_validate_content, mock_validate_size, mock_validate_path, mock_get_response, mock_read_content, sample_config):
        """Test prompt answering with API key."""
        # Set API key in config
        config = Config(**sample_config)
        config.api_key = "test-api-key"
        
        # Mock file reading
        mock_read_content.side_effect = [
            "Test prompt",
            "Expected answer"
        ]
        
        # Mock content validation to return content unchanged
        mock_validate_content.side_effect = lambda content, max_len: content
        
        # Mock path validation to return valid paths
        mock_prompt_path = MagicMock()
        mock_prompt_path.exists.return_value = True
        mock_answer_path = MagicMock()
        mock_answer_path.exists.return_value = True
        mock_validate_path.side_effect = [mock_prompt_path, mock_answer_path]
        
        # Mock API response
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Generated answer"
                    }
                }
            ]
        }
        mock_get_response.return_value = mock_response
        
        answer_prompt(
            "test_prompt.txt",
            "test-model",
            config
        )
        
        # Verify API was called with key
        mock_get_response.assert_called_once()
        call_args = mock_get_response.call_args
        assert call_args[0][3] == "test-api-key"  # api_key parameter
    
    @patch('main.read_file_content')
    @patch('main.get_model_response')
    @patch('main.FileOperationValidator.validate_file_path')
    @patch('main.FileOperationValidator.validate_file_size')
    @patch('main.FileOperationValidator.validate_content_length')
    def test_answer_prompt_with_throttling(self, mock_validate_content, mock_validate_size, mock_validate_path, mock_get_response, mock_read_content, sample_config):
        """Test prompt answering with custom throttling."""
        # Set custom throttling
        config = Config(**sample_config)
        config.throttling_secs = 0.5
        
        # Mock file reading
        mock_read_content.side_effect = [
            "Test prompt",
            "Expected answer"
        ]
        
        # Mock content validation to return content unchanged
        mock_validate_content.side_effect = lambda content, max_len: content
        
        # Mock path validation to return valid paths
        mock_prompt_path = MagicMock()
        mock_prompt_path.exists.return_value = True
        mock_answer_path = MagicMock()
        mock_answer_path.exists.return_value = True
        mock_validate_path.side_effect = [mock_prompt_path, mock_answer_path]
        
        # Mock API response
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Generated answer"
                    }
                }
            ]
        }
        mock_get_response.return_value = mock_response
        
        answer_prompt(
            "test_prompt.txt",
            "test-model",
            config
        )
        
        # Verify API was called with throttling
        mock_get_response.assert_called_once()
        call_args = mock_get_response.call_args
        assert call_args[1]['throttling_secs'] == 0.5  # throttling_secs parameter


@pytest.mark.unit
class TestMainIntegration:
    """Integration tests for main module."""
    
    @patch('main.os.getenv')
    @patch('main.argparse.ArgumentParser.parse_args')
    @patch('main.load_dotenv')
    def test_complete_config_loading(self, mock_load_dotenv, mock_args, mock_getenv):
        """Test complete configuration loading process."""
        # Mock command line arguments
        mock_args.return_value = MagicMock(
            pattern="custom/*.txt",
            actions="answer,evaluate",
            report_json="/custom/report.json"
        )
        
        # Mock environment variables
        def mock_getenv_side_effect(key, default=None):
            env_vars = {
                "ENDPOINT_URL": "https://api.example.com/v1/chat/completions",
                "MODEL_NAMES": "gpt-4,claude-3",
                "MODEL_EVALUATOR": "gpt-3.5-turbo",
                "API_KEY": "sk-test-key",
                "THROTTLING_SECS": "0.2"
            }
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = mock_getenv_side_effect
        
        config = load_config()
        
        # Verify all configuration is loaded correctly
        assert config.endpoint_url == "https://api.example.com/v1/chat/completions"
        assert config.model_names == ["gpt-4", "claude-3"]
        assert config.model_evaluator == "gpt-3.5-turbo"
        assert config.api_key == "sk-test-key"
        assert config.throttling_secs == 0.2
        assert config.pattern == "custom/*.txt"
        assert config.actions == ["answer", "evaluate"]
        assert config.custom_report_json == "/custom/report.json"
        
        # Verify validation passes
        errors = config.validate()
        assert errors == []