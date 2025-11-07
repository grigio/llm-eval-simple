"""Input validation models and utilities for LLM evaluation."""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, validator, HttpUrl


class APIRequest(BaseModel):
    """Model for API request validation."""
    model: str = Field(..., min_length=1, max_length=100)
    messages: List[Dict[str, str]] = Field(..., min_items=1)
    stream: bool = Field(default=False)
    
    @validator('messages')
    def validate_messages(cls, v):
        """Validate message structure."""
        if not v:
            raise ValueError('Messages list cannot be empty')
        
        for i, message in enumerate(v):
            if not isinstance(message, dict):
                raise ValueError(f'Message {i} must be a dictionary')
            
            required_keys = {'role', 'content'}
            if not required_keys.issubset(message.keys()):
                raise ValueError(f'Message {i} must contain {required_keys}')
            
            valid_roles = {'system', 'user', 'assistant'}
            if message['role'] not in valid_roles:
                raise ValueError(f'Message {i} role must be one of {valid_roles}')
            
            if not message['content'].strip():
                raise ValueError(f'Message {i} content cannot be empty')
        
        return v


class EvaluationRequest(BaseModel):
    """Model for evaluation request validation."""
    expected_answer: str = Field(..., min_length=1)
    generated_answer: str = Field(..., min_length=1)
    evaluator_model: str = Field(..., min_length=1)
    
    @validator('expected_answer', 'generated_answer')
    def validate_answers(cls, v):
        """Validate answer content."""
        if not v or not v.strip():
            raise ValueError('Answer cannot be empty')
        if len(v) > 10000:  # Reasonable limit
            raise ValueError('Answer too long (max 10000 characters)')
        return v.strip()


class ConfigValidation(BaseModel):
    """Enhanced configuration validation model."""
    endpoint_url: str = Field(..., min_length=1)
    model_names: List[str] = Field(..., min_items=1)
    model_evaluator: str = Field(..., min_length=1)
    pattern: str = Field(..., min_length=1)
    actions: List[str] = Field(..., min_items=1)
    api_key: Optional[str] = None
    throttling_secs: float = Field(..., ge=0)
    prompt_dir: str = Field(..., min_length=1)
    answer_dir: str = Field(..., min_length=1)
    
    @validator('endpoint_url')
    def validate_endpoint_url(cls, v):
        """Validate endpoint URL format."""
        if not v:
            raise ValueError('Endpoint URL cannot be empty')
        
        try:
            parsed = urlparse(v)
            if parsed.scheme not in ('http', 'https'):
                raise ValueError('URL must use http or https protocol')
            if not parsed.netloc:
                raise ValueError('URL must have a valid host')
            return v.rstrip('/')
        except Exception:
            raise ValueError('Invalid URL format')
    
    @validator('model_names', 'model_evaluator')
    def validate_model_names(cls, v):
        """Validate model names."""
        if isinstance(v, str):
            v = [v]
        
        for name in v:
            if not name or not name.strip():
                raise ValueError('Model names cannot be empty')
            if len(name) > 100:
                raise ValueError('Model name too long (max 100 characters)')
            # Allow alphanumeric, hyphens, underscores, dots
            if not re.match(r'^[a-zA-Z0-9._-]+$', name):
                raise ValueError(f'Invalid model name: {name}')
        
        return v
    
    @validator('actions')
    def validate_actions(cls, v):
        """Validate actions list."""
        valid_actions = {'answer', 'evaluate', 'render', 'serve'}
        for action in v:
            if action not in valid_actions:
                raise ValueError(f'Invalid action: {action}. Valid actions: {valid_actions}')
        return v
    
    @validator('pattern')
    def validate_pattern(cls, v):
        """Validate glob pattern."""
        if not v or not v.strip():
            raise ValueError('Pattern cannot be empty')
        
        # Basic security check for path traversal
        if '..' in v:
            raise ValueError('Pattern cannot contain path traversal')
        
        return v.strip()
    

    
    @validator('prompt_dir', 'answer_dir')
    def validate_directories(cls, v):
        """Validate directory paths."""
        if not v or not v.strip():
            raise ValueError('Directory cannot be empty')
        
        # Security check for path traversal
        if '..' in v:
            raise ValueError('Directory cannot contain path traversal')
        
        return v.strip()


class FileOperationValidator:
    """Security utilities for file operations."""
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.jsx', '.ts', '.tsx', '.json'}
    BASE_DIRECTORIES = {'prompts', 'answers', 'generated_answers'}
    
    @staticmethod
    def validate_file_path(file_path: str, base_dir: str = None) -> Path:
        """Validate file path for security."""
        try:
            # Handle both relative and absolute paths
            if os.path.isabs(file_path):
                path = Path(file_path)
            else:
                # For relative paths, resolve from current directory
                path = Path(file_path).resolve()
            
            # Check for path traversal
            if base_dir:
                base_path = Path(base_dir).resolve()
                if not str(path).startswith(str(base_path)):
                    raise ValueError(f'Path traversal detected: {file_path}')
            
            # Check file extension - allow if no extension (for directories)
            if path.suffix and path.suffix.lower() not in FileOperationValidator.ALLOWED_EXTENSIONS:
                raise ValueError(f'File type not allowed: {path.suffix}')
            
            return path
        except Exception as e:
            raise ValueError(f'Invalid file path: {e}')
    
    @staticmethod
    def validate_file_size(file_path: Path) -> bool:
        """Validate file size."""
        if file_path.exists() and file_path.stat().st_size > FileOperationValidator.MAX_FILE_SIZE:
            raise ValueError(f'File too large: {file_path}')
        return True
    
    @staticmethod
    def validate_content_length(content: str, max_length: int = 50000) -> str:
        """Validate content length."""
        if len(content) > max_length:
            raise ValueError(f'Content too long: {len(content)} > {max_length}')
        return content


class APIResponseValidator:
    """Utilities for validating API responses."""
    
    @staticmethod
    def validate_openai_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate OpenAI-compatible API response."""
        if not isinstance(response, dict):
            raise ValueError('Response must be a dictionary')
        
        if 'choices' not in response:
            raise ValueError('Response missing required field: choices')
        
        choices = response['choices']
        if not isinstance(choices, list) or not choices:
            raise ValueError('Choices must be a non-empty list')
        
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError('Choice must be a dictionary')
        
        if 'message' not in choice:
            raise ValueError('Choice missing required field: message')
        
        message = choice['message']
        if not isinstance(message, dict):
            raise ValueError('Message must be a dictionary')
        
        if 'content' not in message:
            raise ValueError('Message missing required field: content')
        
        content = message['content']
        if not isinstance(content, str):
            raise ValueError('Content must be a string')
        
        return response
    
    @staticmethod
    def sanitize_content(content: str, max_length: int = 10000) -> str:
        """Sanitize and validate content."""
        if not isinstance(content, str):
            raise ValueError('Content must be a string')
        
        # Remove potential HTML/script tags for security
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'<[^>]+>', '', content)
        
        # Limit length
        if len(content) > max_length:
            content = content[:max_length] + '...'
        
        return content.strip()


def validate_glob_pattern(pattern: str) -> str:
    """Validate glob pattern for security."""
    if not pattern or not pattern.strip():
        raise ValueError('Pattern cannot be empty')
    
    pattern = pattern.strip()
    
    # Security checks
    if '..' in pattern:
        raise ValueError('Pattern cannot contain path traversal')
    
    # Check for dangerous patterns (more specific checks)
    if pattern == '/*' or pattern == '/.*':
        raise ValueError('Dangerous pattern detected: absolute root patterns not allowed')
    if pattern.startswith('~') or '$HOME' in pattern:
        raise ValueError('Dangerous pattern detected: home directory patterns not allowed')
    
    return pattern


def validate_model_list(model_names: str) -> List[str]:
    """Validate and parse model names list."""
    if not model_names or not model_names.strip():
        raise ValueError('Model names cannot be empty')
    
    models = [name.strip() for name in model_names.split(',') if name.strip()]
    
    if not models:
        raise ValueError('No valid model names found')
    
    for model in models:
        if not re.match(r'^[a-zA-Z0-9._-]+$', model):
            raise ValueError(f'Invalid model name: {model}')
    
    return models