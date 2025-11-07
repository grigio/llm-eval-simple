"""Unit tests for API client module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from api_client import get_model_response, evaluate_correctness


class TestGetModelResponse:
    """Test get_model_response function."""
    
    @patch('api_client.requests.post')
    @patch('time.sleep')
    def test_successful_response(self, mock_sleep, mock_post, mock_requests_response):
        """Test successful API response."""
        mock_post.return_value = mock_requests_response
        
        result = get_model_response(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            model="test-model",
            prompt="Test prompt",
            api_key="test-key"
        )
        
        assert result == mock_requests_response.json.return_value
        mock_sleep.assert_called_once_with(0.1)
        mock_post.assert_called_once()
    
    @patch('api_client.requests.post')
    def test_request_timeout(self, mock_post):
        """Test request timeout handling."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timeout")
        
        with pytest.raises(ValueError, match="Request timeout"):
            get_model_response(
                endpoint_url="http://localhost:9292/v1/chat/completions",
                model="test-model",
                prompt="Test prompt"
            )
    
    @patch('api_client.requests.post')
    def test_connection_error(self, mock_post):
        """Test connection error handling."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with pytest.raises(ValueError, match="Connection error"):
            get_model_response(
                endpoint_url="http://localhost:9292/v1/chat/completions",
                model="test-model",
                prompt="Test prompt"
            )
    
    @patch('api_client.requests.post')
    def test_http_error(self, mock_post):
        """Test HTTP error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Too Many Requests"
        mock_post.return_value = mock_response
        mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Client Error")
        mock_post.return_value.response = mock_response
        
        with pytest.raises(ValueError, match="HTTP error: 429 Client Error"):
            get_model_response(
                endpoint_url="http://localhost:9292/v1/chat/completions",
                model="test-model",
                prompt="Test prompt"
            )
    
    def test_missing_required_parameters(self):
        """Test validation of required parameters."""
        with pytest.raises(ValueError, match="endpoint_url, model, and prompt are required"):
            get_model_response(
                endpoint_url="",
                model="test-model",
                prompt="Test prompt"
            )
    
    @patch('api_client.requests.post')
    @patch('time.sleep')
    def test_with_system_prompt(self, mock_sleep, mock_post, mock_requests_response):
        """Test API call with system prompt."""
        mock_post.return_value = mock_requests_response
        
        result = get_model_response(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            model="test-model",
            prompt="Test prompt",
            system_prompt="System instructions",
            api_key="test-key"
        )
        
        # Verify the call includes system prompt
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        
        assert len(payload['messages']) == 2
        assert payload['messages'][0]['role'] == 'system'
        assert payload['messages'][0]['content'] == 'System instructions'
        assert payload['messages'][1]['role'] == 'user'
        assert payload['messages'][1]['content'] == 'Test prompt'
    
    @patch('api_client.requests.post')
    @patch('time.sleep')
    def test_content_sanitization(self, mock_sleep, mock_post, mock_requests_response):
        """Test content sanitization."""
        mock_post.return_value = mock_requests_response
        
        prompt_with_script = "<script>alert('xss')</script>Hello"
        
        get_model_response(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            model="test-model",
            prompt=prompt_with_script
        )
        
        # Verify script tags are removed
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        content = payload['messages'][0]['content']
        
        assert '<script>' not in content
        assert 'Hello' in content
    
    @patch('api_client.requests.post')
    @patch('time.sleep')
    def test_custom_throttling(self, mock_sleep, mock_post, mock_requests_response):
        """Test custom throttling value."""
        mock_post.return_value = mock_requests_response
        
        get_model_response(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            model="test-model",
            prompt="Test prompt",
            throttling_secs=0.5
        )
        
        mock_sleep.assert_called_once_with(0.5)
    
    @patch('api_client.requests.post')
    @patch('time.sleep')
    def test_api_key_header(self, mock_sleep, mock_post, mock_requests_response):
        """Test API key is included in headers."""
        mock_post.return_value = mock_requests_response
        
        get_model_response(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            model="test-model",
            prompt="Test prompt",
            api_key="test-api-key"
        )
        
        call_args = mock_post.call_args
        headers = call_args[1]['headers']
        
        assert headers['Authorization'] == 'Bearer test-api-key'
    
    @patch('api_client.requests.post')
    @patch('time.sleep')
    def test_no_api_key(self, mock_sleep, mock_post, mock_requests_response):
        """Test request without API key."""
        mock_post.return_value = mock_requests_response
        
        get_model_response(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            model="test-model",
            prompt="Test prompt"
        )
        
        call_args = mock_post.call_args
        headers = call_args[1]['headers']
        
        assert 'Authorization' not in headers


class TestEvaluateCorrectness:
    """Test evaluate_correctness function."""
    
    def test_simple_string_comparison_no_evaluator(self):
        """Test simple string comparison when no evaluator model."""
        result = evaluate_correctness(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            evaluator_model="",
            expected_answer="Test",
            generated_answer="test"
        )
        
        assert result is True  # Case insensitive comparison
    
    def test_simple_string_comparison_false(self):
        """Test simple string comparison returning false."""
        result = evaluate_correctness(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            evaluator_model="",
            expected_answer="Test",
            generated_answer="Different"
        )
        
        assert result is False
    
    @patch('api_client.get_model_response')
    def test_evaluator_correct_response(self, mock_get_response):
        """Test evaluator model returning CORRECT."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "CORRECT"
                    }
                }
            ]
        }
        mock_get_response.return_value = mock_response
        
        result = evaluate_correctness(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            evaluator_model="evaluator-model",
            expected_answer="Expected answer",
            generated_answer="Generated answer"
        )
        
        assert result is True
        mock_get_response.assert_called_once()
    
    @patch('api_client.get_model_response')
    def test_evaluator_incorrect_response(self, mock_get_response):
        """Test evaluator model returning INCORRECT."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "INCORRECT"
                    }
                }
            ]
        }
        mock_get_response.return_value = mock_response
        
        result = evaluate_correctness(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            evaluator_model="evaluator-model",
            expected_answer="Expected answer",
            generated_answer="Generated answer"
        )
        
        assert result is False
    
    @patch('api_client.get_model_response')
    def test_evaluator_ambiguous_response(self, mock_get_response):
        """Test evaluator model returning ambiguous response."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "MAYBE CORRECT"
                    }
                }
            ]
        }
        mock_get_response.return_value = mock_response
        
        result = evaluate_correctness(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            evaluator_model="evaluator-model",
            expected_answer="Expected answer",
            generated_answer="Generated answer"
        )
        
        assert result is False  # Default to incorrect for ambiguous responses
    
    @patch('api_client.get_model_response')
    def test_evaluator_empty_response(self, mock_get_response):
        """Test evaluator model returning empty response."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": ""
                    }
                }
            ]
        }
        mock_get_response.return_value = mock_response
        
        result = evaluate_correctness(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            evaluator_model="evaluator-model",
            expected_answer="Expected answer",
            generated_answer="Generated answer"
        )
        
        assert result is False
    
    def test_missing_required_answers(self):
        """Test validation of required answer parameters."""
        with pytest.raises(ValueError, match="Both expected_answer and generated_answer are required"):
            evaluate_correctness(
                endpoint_url="http://localhost:9292/v1/chat/completions",
                evaluator_model="evaluator-model",
                expected_answer="",
                generated_answer="Generated answer"
            )
    
    @patch('api_client.get_model_response')
    def test_evaluation_with_api_key(self, mock_get_response):
        """Test evaluation with API key."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "CORRECT"
                    }
                }
            ]
        }
        mock_get_response.return_value = mock_response
        
        evaluate_correctness(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            evaluator_model="evaluator-model",
            expected_answer="Expected answer",
            generated_answer="Generated answer",
            api_key="test-key"
        )
        
        # Verify the API call includes the API key
        mock_get_response.assert_called_once_with(
            "http://localhost:9292/v1/chat/completions",
            "evaluator-model",
            "Expected Answer: Expected answer\nGenerated Answer: Generated answer",
            "test-key",
            "You are an evaluator. Compare the expected answer with the generated answer. Ignore the tag content. The generated answers may vary slightly in wording but should preserve the original meaning. If the answers are equivalent in meaning, mark as correct. Respond with only 'CORRECT' or 'INCORRECT'.",
            0.1
        )
    
    @patch('api_client.get_model_response')
    def test_evaluation_error_handling(self, mock_get_response):
        """Test evaluation error handling."""
        mock_get_response.side_effect = ValueError("API error")
        
        result = evaluate_correctness(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            evaluator_model="evaluator-model",
            expected_answer="Expected answer",
            generated_answer="Generated answer"
        )
        
        assert result is False
    
    @patch('api_client.get_model_response')
    def test_evaluation_with_custom_throttling(self, mock_get_response):
        """Test evaluation with custom throttling."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "CORRECT"
                    }
                }
            ]
        }
        mock_get_response.return_value = mock_response
        
        evaluate_correctness(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            evaluator_model="evaluator-model",
            expected_answer="Expected answer",
            generated_answer="Generated answer",
            throttling_secs=0.5
        )
        
        # Verify throttling is passed through
        call_args = mock_get_response.call_args
        assert call_args[0][5] == 0.5  # throttling_secs is 6th positional argument


@pytest.mark.unit
class TestAPIClientIntegration:
    """Integration tests for API client."""
    
    @patch('api_client.get_model_response')
    def test_complete_evaluation_flow(self, mock_get_response):
        """Test complete evaluation flow."""
        # Mock the evaluation response
        mock_eval_response = {
            "choices": [
                {
                    "message": {
                        "content": "CORRECT"
                    }
                }
            ]
        }
        mock_get_response.return_value = mock_eval_response
        
        result = evaluate_correctness(
            endpoint_url="http://localhost:9292/v1/chat/completions",
            evaluator_model="evaluator-model",
            expected_answer="The answer is 42",
            generated_answer="42",
            api_key="test-key",
            throttling_secs=0.2
        )
        
        assert result is True
        
        # Verify the correct prompt was sent
        call_args = mock_get_response.call_args
        prompt = call_args[0][2]  # prompt is 3rd positional argument
        
        assert "Expected Answer: The answer is 42" in prompt
        assert "Generated Answer: 42" in prompt
        assert call_args[0][3] == "test-key"  # api_key is 4th positional argument
        assert call_args[0][5] == 0.2  # throttling_secs is 6th positional argument