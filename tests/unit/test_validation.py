"""Unit tests for validation module."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from validation import (
    APIRequest,
    EvaluationRequest,
    ConfigValidation,
    FileOperationValidator,
    APIResponseValidator,
    validate_glob_pattern,
    validate_model_list
)


class TestAPIRequest:
    """Test APIRequest validation."""
    
    def test_valid_api_request(self):
        """Test valid API request creation."""
        messages = [{"role": "user", "content": "Hello"}]
        request = APIRequest(model="test-model", messages=messages)
        
        assert request.model == "test-model"
        assert request.messages == messages
        assert request.stream is False
    
    def test_empty_messages_validation(self):
        """Test validation of empty messages."""
        with pytest.raises(ValidationError, match="List should have at least 1 item"):
            APIRequest(model="test", messages=[])
    
    def test_invalid_message_structure(self):
        """Test validation of invalid message structure."""
        invalid_messages = [{"invalid": "message"}]
        
        with pytest.raises(ValidationError, match="Message 0 must contain"):
            APIRequest(model="test", messages=invalid_messages)
    
    def test_invalid_role(self):
        """Test validation of invalid role."""
        invalid_messages = [{"role": "invalid", "content": "Hello"}]
        
        with pytest.raises(ValidationError, match="Message 0 role must be"):
            APIRequest(model="test", messages=invalid_messages)
    
    def test_empty_content(self):
        """Test validation of empty content."""
        invalid_messages = [{"role": "user", "content": ""}]
        
        with pytest.raises(ValidationError, match="Message 0 content cannot be empty"):
            APIRequest(model="test", messages=invalid_messages)


class TestEvaluationRequest:
    """Test EvaluationRequest validation."""
    
    def test_valid_evaluation_request(self):
        """Test valid evaluation request."""
        request = EvaluationRequest(
            expected_answer="Expected",
            generated_answer="Generated",
            evaluator_model="evaluator"
        )
        
        assert request.expected_answer == "Expected"
        assert request.generated_answer == "Generated"
        assert request.evaluator_model == "evaluator"
    
    def test_empty_expected_answer(self):
        """Test validation of empty expected answer."""
        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            EvaluationRequest(
                expected_answer="",
                generated_answer="Generated",
                evaluator_model="evaluator"
            )
    
    def test_too_long_answer(self):
        """Test validation of too long answer."""
        long_answer = "x" * 10001
        
        with pytest.raises(ValidationError, match="Answer too long"):
            EvaluationRequest(
                expected_answer=long_answer,
                generated_answer="Generated",
                evaluator_model="evaluator"
            )


class TestConfigValidation:
    """Test ConfigValidation."""
    
    def test_valid_config(self, sample_config):
        """Test valid configuration."""
        config = ConfigValidation(**sample_config)
        
        assert config.endpoint_url == sample_config["endpoint_url"]
        assert config.model_names == sample_config["model_names"]
        assert config.throttling_secs == sample_config["throttling_secs"]
    
    def test_invalid_url(self, sample_config):
        """Test invalid URL validation."""
        sample_config["endpoint_url"] = "invalid-url"
        
        with pytest.raises(ValidationError, match="Invalid URL format"):
            ConfigValidation(**sample_config)
    
    def test_invalid_model_names(self, sample_config):
        """Test invalid model names."""
        sample_config["model_names"] = ["invalid name!"]
        
        with pytest.raises(ValidationError, match="Invalid model name"):
            ConfigValidation(**sample_config)
    
    def test_invalid_actions(self, sample_config):
        """Test invalid actions."""
        sample_config["actions"] = ["invalid_action"]
        
        with pytest.raises(ValidationError, match="Invalid action"):
            ConfigValidation(**sample_config)
    
    def test_negative_throttling(self, sample_config):
        """Test negative throttling validation."""
        sample_config["throttling_secs"] = -1.0
        
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
            ConfigValidation(**sample_config)
    
    def test_path_traversal_in_pattern(self, sample_config):
        """Test path traversal in pattern."""
        sample_config["pattern"] = "../../../etc/passwd"
        
        with pytest.raises(ValidationError, match="Pattern cannot contain path traversal"):
            ConfigValidation(**sample_config)
    



class TestFileOperationValidator:
    """Test FileOperationValidator."""
    
    def test_valid_file_path(self, sample_json_file):
        """Test valid file path validation."""
        validated_path = FileOperationValidator.validate_file_path(str(sample_json_file))
        
        assert validated_path == sample_json_file.resolve()
    
    def test_invalid_file_extension(self, temp_dir):
        """Test invalid file extension."""
        invalid_file = temp_dir / "test.exe"
        invalid_file.write_text("content")
        
        with pytest.raises(ValueError, match="File type not allowed"):
            FileOperationValidator.validate_file_path(str(invalid_file))
    
    def test_path_traversal_protection(self, temp_dir):
        """Test path traversal protection."""
        with pytest.raises(ValueError, match="Path traversal detected"):
            FileOperationValidator.validate_file_path("../../../etc/passwd", str(temp_dir))
    
    def test_file_size_validation(self, temp_dir):
        """Test file size validation."""
        # Create a file larger than the limit
        large_file = temp_dir / "large.txt"
        large_file.write_text("x" * (11 * 1024 * 1024))  # 11MB
        
        with pytest.raises(ValueError, match="File too large"):
            FileOperationValidator.validate_file_size(large_file)
    
    def test_content_length_validation(self):
        """Test content length validation."""
        long_content = "x" * 60000  # Exceeds default 50000 limit
        
        with pytest.raises(ValueError, match="Content too long"):
            FileOperationValidator.validate_content_length(long_content)
    
    def test_allowed_extensions(self):
        """Test allowed file extensions."""
        allowed = {'.txt', '.md', '.py', '.js', '.jsx', '.ts', '.tsx', '.json'}
        assert FileOperationValidator.ALLOWED_EXTENSIONS == allowed


class TestAPIResponseValidator:
    """Test APIResponseValidator."""
    
    def test_valid_openai_response(self, mock_api_response):
        """Test valid OpenAI response validation."""
        validated = APIResponseValidator.validate_openai_response(mock_api_response)
        
        assert validated == mock_api_response
    
    def test_missing_choices(self):
        """Test response missing choices."""
        invalid_response = {"data": "test"}
        
        with pytest.raises(ValueError, match="Response missing required field: choices"):
            APIResponseValidator.validate_openai_response(invalid_response)
    
    def test_empty_choices(self):
        """Test empty choices list."""
        invalid_response = {"choices": []}
        
        with pytest.raises(ValueError, match="Choices must be a non-empty list"):
            APIResponseValidator.validate_openai_response(invalid_response)
    
    def test_missing_message(self):
        """Test choice missing message."""
        invalid_response = {"choices": [{"data": "test"}]}
        
        with pytest.raises(ValueError, match="Choice missing required field: message"):
            APIResponseValidator.validate_openai_response(invalid_response)
    
    def test_missing_content(self):
        """Test message missing content."""
        invalid_response = {"choices": [{"message": {"role": "assistant"}}]}
        
        with pytest.raises(ValueError, match="Message missing required field: content"):
            APIResponseValidator.validate_openai_response(invalid_response)
    
    def test_sanitize_content(self):
        """Test content sanitization."""
        content_with_script = "<script>alert('xss')</script>Hello"
        sanitized = APIResponseValidator.sanitize_content(content_with_script)
        
        assert "<script>" not in sanitized
        assert "Hello" in sanitized
    
    def test_content_length_limit(self):
        """Test content length limit."""
        long_content = "x" * 15000
        
        sanitized = APIResponseValidator.sanitize_content(long_content, max_length=100)
        
        assert len(sanitized) <= 103  # 100 + "..."
        assert sanitized.endswith("...")


class TestUtilityFunctions:
    """Test utility validation functions."""
    
    def test_validate_glob_pattern_valid(self):
        """Test valid glob pattern."""
        pattern = validate_glob_pattern("prompts/*.txt")
        assert pattern == "prompts/*.txt"
    
    def test_validate_glob_pattern_empty(self):
        """Test empty glob pattern."""
        with pytest.raises(ValueError, match="Pattern cannot be empty"):
            validate_glob_pattern("")
    
    def test_validate_glob_pattern_traversal(self):
        """Test glob pattern with path traversal."""
        with pytest.raises(ValueError, match="Pattern cannot contain path traversal"):
            validate_glob_pattern("../../../etc/*")
    
    def test_validate_glob_pattern_dangerous(self):
        """Test dangerous glob pattern."""
        with pytest.raises(ValueError, match="Dangerous pattern detected"):
            validate_glob_pattern("/*")
    
    def test_validate_model_list_valid(self):
        """Test valid model list."""
        models = validate_model_list("model-a,model-b,model-c")
        assert models == ["model-a", "model-b", "model-c"]
    
    def test_validate_model_list_empty(self):
        """Test empty model list."""
        with pytest.raises(ValueError, match="Model names cannot be empty"):
            validate_model_list("")
    
    def test_validate_model_list_invalid_characters(self):
        """Test model list with invalid characters."""
        with pytest.raises(ValueError, match="Invalid model name"):
            validate_model_list("model-a,invalid model!")
    
    def test_validate_model_list_whitespace(self):
        """Test model list with whitespace."""
        models = validate_model_list(" model-a , model-b ")
        assert models == ["model-a", "model-b"]


@pytest.mark.unit
class TestValidationIntegration:
    """Integration tests for validation."""
    
    def test_complete_validation_flow(self, sample_config):
        """Test complete validation flow."""
        # Validate configuration
        config = ConfigValidation(**sample_config)
        
        # Validate API request
        messages = [{"role": "user", "content": "Test"}]
        api_request = APIRequest(model=config.model_names[0], messages=messages)
        
        # Validate evaluation request
        eval_request = EvaluationRequest(
            expected_answer="Expected",
            generated_answer="Generated",
            evaluator_model=config.model_evaluator[0]  # Take first model
        )
        
        assert config is not None
        assert api_request is not None
        assert eval_request is not None