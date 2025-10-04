"""Validation rules for authentication and LangGraph configuration.

Provides reusable validation functions for user registration, authentication,
and LangGraph agent configuration.
"""

import re
from typing import Tuple, Dict, Any


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


def validate_username(username: str) -> Tuple[bool, str]:
    """Validate username format and requirements.
    
    Args:
        username: The username to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
        
    Rules:
        - Must be between 3 and 30 characters
        - Can only contain letters, numbers, and underscores
        - Must start with a letter
    """
    if not username:
        return False, "Username cannot be empty"
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    
    if len(username) > 30:
        return False, "Username cannot exceed 30 characters"
    
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', username):
        return False, "Username must start with a letter and contain only letters, numbers, and underscores"
    
    return True, ""


def validate_password(password: str) -> Tuple[bool, str]:
    """Validate password strength and requirements.
    
    Args:
        password: The password to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
        
    Rules:
        - Must be more than 4 characters
        - Cannot contain spaces at the beginning or end
    """
    if not password:
        return False, "Password cannot be empty"
    
    if len(password) <= 4:
        return False, "Password must be longer than 4 characters"
    
    if password != password.strip():
        return False, "Password cannot start or end with spaces"
    
    return True, ""


def validate_password_match(password: str, confirm_password: str) -> Tuple[bool, str]:
    """Validate that two passwords match.
    
    Args:
        password: The original password.
        confirm_password: The confirmation password.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if password != confirm_password:
        return False, "Passwords do not match"
    
    return True, ""


def validate_admin_permission(requesting_user_is_admin: bool) -> Tuple[bool, str]:
    """Validate that a user has admin permissions for certain operations.
    
    Args:
        requesting_user_is_admin: Whether the requesting user is an admin.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not requesting_user_is_admin:
        return False, "Admin permission required for this operation"
    
    return True, ""


def validate_registration_data(username: str, password: str, confirm_password: str = None) -> Tuple[bool, str]:
    """Validate all registration data.
    
    Args:
        username: The username to register.
        password: The password for the account.
        confirm_password: Optional password confirmation.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    # Validate username
    is_valid, error_msg = validate_username(username)
    if not is_valid:
        return False, error_msg
    
    # Validate password
    is_valid, error_msg = validate_password(password)
    if not is_valid:
        return False, error_msg
    
    # Validate password match if confirmation provided
    if confirm_password is not None:
        is_valid, error_msg = validate_password_match(password, confirm_password)
        if not is_valid:
            return False, error_msg
    
    return True, ""


def validate_api_key_format(api_key: str) -> Tuple[bool, str]:
    """Validate API key format.
    
    Args:
        api_key: The API key to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
        
    Rules:
        - Must start with 'sk-'
        - Must be at least 20 characters long
        - Must contain only alphanumeric characters, hyphens, and underscores after prefix
    """
    if not api_key:
        return False, "API key cannot be empty"
    
    if not api_key.startswith("sk-"):
        return False, "API key must start with 'sk-'"
    
    if len(api_key) < 20:
        return False, "API key is too short"
    
    # Check characters after 'sk-' prefix
    key_body = api_key[3:]
    if not re.match(r'^[a-zA-Z0-9_-]+$', key_body):
        return False, "API key contains invalid characters"
    
    return True, ""


def validate_preferences(preferences: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate user preferences for LangGraph configuration.
    
    Args:
        preferences: Dictionary of preferences to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not isinstance(preferences, dict):
        return False, "Preferences must be a dictionary"
    
    # Validate temperature if present
    if "temperature" in preferences:
        temp = preferences["temperature"]
        if not isinstance(temp, (int, float)):
            return False, "Temperature must be a number"
        if temp < 0 or temp > 2:
            return False, "Temperature must be between 0 and 2"
    
    # Validate max_tokens if present
    if "max_tokens" in preferences:
        max_tokens = preferences["max_tokens"]
        if not isinstance(max_tokens, int):
            return False, "Max tokens must be an integer"
        if max_tokens < 1 or max_tokens > 100000:
            return False, "Max tokens must be between 1 and 100000"
    
    # Validate model if present
    if "model" in preferences:
        model = preferences["model"]
        if not isinstance(model, str):
            return False, "Model must be a string"
        # Add supported models validation if needed
        supported_models = [
            "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o",
            "claude-2", "claude-3-opus", "claude-3-sonnet",
            "llama-2", "mistral", "gemini-pro"
        ]
        if model not in supported_models:
            return False, f"Model '{model}' is not supported"
    
    # Validate boolean preferences
    boolean_prefs = ["auto_checkpoint", "enable_tool_rollback"]
    for pref in boolean_prefs:
        if pref in preferences and not isinstance(preferences[pref], bool):
            return False, f"{pref} must be a boolean"
    
    # Validate integer preferences
    integer_prefs = ["checkpoint_frequency", "max_checkpoints"]
    for pref in integer_prefs:
        if pref in preferences:
            value = preferences[pref]
            if not isinstance(value, int):
                return False, f"{pref} must be an integer"
            if value < 1:
                return False, f"{pref} must be positive"
    
    # Validate system_prompt if present
    if "system_prompt" in preferences:
        prompt = preferences["system_prompt"]
        if not isinstance(prompt, str):
            return False, "System prompt must be a string"
        if len(prompt) > 10000:
            return False, "System prompt is too long (max 10000 characters)"
    
    return True, ""


def validate_session_limit(session_limit: int) -> Tuple[bool, str]:
    """Validate session limit for a user.
    
    Args:
        session_limit: The session limit to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not isinstance(session_limit, int):
        return False, "Session limit must be an integer"
    
    if session_limit < 1:
        return False, "Session limit must be at least 1"
    
    if session_limit > 100:
        return False, "Session limit cannot exceed 100"
    
    return True, ""