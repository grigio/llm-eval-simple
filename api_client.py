"""API interaction functions for LLM evaluation."""

import time
from typing import Any, Dict

import requests

from shared import RESPONSE_TIME_DECIMAL_PLACES
from validation import APIRequest, EvaluationRequest, APIResponseValidator


def get_model_response(endpoint_url: str, model: str, prompt: str, api_key: str = None, 
                      system_prompt: str = None, throttling_secs: float = 0.1) -> Dict[str, Any]:
    """Gets a response from the specified model with input validation."""
    try:
        # Validate inputs
        if not endpoint_url or not model or not prompt:
            raise ValueError("endpoint_url, model, and prompt are required")
        
        # Sanitize and validate prompt content
        prompt = APIResponseValidator.sanitize_content(prompt, 10000)
        if system_prompt:
            system_prompt = APIResponseValidator.sanitize_content(system_prompt, 2000)
        
        # Apply throttling
        time.sleep(throttling_secs)
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Validate request structure
        api_request = APIRequest(
            model=model,
            messages=messages,
            stream=False
        )
        
        # Prepare request
        payload = api_request.dict()
        headers = {"Content-Type": "application/json"}
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Make request with timeout
        response = requests.post(
            endpoint_url, 
            json=payload, 
            headers=headers,
            timeout=700  # 700 second timeout
        )
        response.raise_for_status()
        
        # Validate response structure
        response_data = response.json()
        APIResponseValidator.validate_openai_response(response_data)
        
        return response_data
        
    except requests.exceptions.Timeout:
        raise ValueError(f"Request timeout for model {model}")
    except requests.exceptions.ConnectionError:
        raise ValueError(f"Connection error for endpoint {endpoint_url}")
    except requests.exceptions.HTTPError as e:
        if hasattr(e, 'response') and e.response is not None:
            raise ValueError(f"HTTP error {e.response.status_code}: {e.response.text}")
        else:
            raise ValueError(f"HTTP error: {e}")
    except ValueError as e:
        raise ValueError(f"Validation error: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")


def evaluate_correctness(endpoint_url: str, evaluator_model: str, expected_answer: str, 
                        generated_answer: str, api_key: str = None, 
                        throttling_secs: float = 0.1) -> bool:
    """Evaluates the correctness of a generated answer using an evaluator model."""
    try:
        # Validate inputs
        if not expected_answer or not generated_answer:
            # Return False for empty answers instead of raising an error
            return False
        
        # Simple string comparison if no evaluator model
        if not evaluator_model:
            return generated_answer.lower().strip() == expected_answer.lower().strip()

        # Validate evaluation request
        eval_request = EvaluationRequest(
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            evaluator_model=evaluator_model
        )

        system_prompt = "You are an evaluator. Compare the expected answer with the generated answer. Ignore the tag content. The generated answers may vary slightly in wording but should preserve the original meaning. If the answers are equivalent in meaning, mark as correct. Respond with only 'CORRECT' or 'INCORRECT'."
        user_prompt = f"Expected Answer: {eval_request.expected_answer}\nGenerated Answer: {eval_request.generated_answer}"
        
        # Get evaluation from model
        eval_response = get_model_response(
            endpoint_url, 
            evaluator_model, 
            user_prompt, 
            api_key, 
            system_prompt, 
            throttling_secs
        )
        
        eval_result = eval_response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        # Validate evaluation result
        if not eval_result:
            raise ValueError("Empty evaluation response")
        
        # More flexible evaluation - look for clear indicators of correctness
        if "CORRECT" == eval_result:
            return True
        if "INCORRECT" == eval_result:
            return False

        # Default to incorrect if response is ambiguous
        print(f"⚠️  Ambiguous evaluation response: '{eval_result}', marking as incorrect")
        return False
        
    except ValueError as e:
        print(f"❌ Evaluation validation error: {e}")
        return False
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return False