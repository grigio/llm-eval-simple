"""API interaction functions for LLM evaluation."""

import time
from typing import Any, Dict

import requests

from shared import RESPONSE_TIME_DECIMAL_PLACES


def get_model_response(endpoint_url: str, model: str, prompt: str, api_key: str = None, 
                      system_prompt: str = None, throttling_secs: float = 0.1) -> Dict[str, Any]:
    """Gets a response from the specified model."""
    time.sleep(throttling_secs)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    response = requests.post(endpoint_url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def evaluate_correctness(endpoint_url: str, evaluator_model: str, expected_answer: str, 
                        generated_answer: str, api_key: str = None, 
                        throttling_secs: float = 0.1) -> bool:
    """Evaluates the correctness of a generated answer using an evaluator model."""
    if not evaluator_model:
        return generated_answer.lower() == expected_answer.lower()

    system_prompt = "You are an evaluator. Compare the expected answer with the generated answer. Ignore the tag  content. The generated answers may vary slightly in wording but should preserve the original meaning. If the answers are equivalent in meaning, mark as correct. Respond with only 'CORRECT' or 'INCORRECT'."
    user_prompt = f"Expected Answer: {expected_answer}\nGenerated Answer: {generated_answer}"
    
    try:
        eval_response = get_model_response(endpoint_url, evaluator_model, user_prompt, api_key, system_prompt, throttling_secs)
        eval_result = eval_response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        # More flexible evaluation - look for clear indicators of correctness
        if "CORRECT" == eval_result:
            return True
        if "INCORRECT" == eval_result:
            return False

        return False
    except requests.exceptions.RequestException as e:
        print(f"Evaluator error: {str(e)}")
        return False