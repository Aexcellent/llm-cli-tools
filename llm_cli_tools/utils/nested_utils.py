"""
Nested data structure utilities
"""

from typing import Any, Dict, List, Union


def get_nested_value(data: Dict[str, Any], key_path: Union[str, List[str]]) -> Any:
    """Get a value from a nested dictionary using a key path
    
    Args:
        data: The dictionary to search
        key_path: Either a dot-separated string or a list of keys
        
    Returns:
        The value at the specified path, or None if not found
    """
    if isinstance(key_path, str):
        keys = key_path.split('.')
    else:
        keys = key_path
    
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and key.isdigit():
            index = int(key)
            if 0 <= index < len(current):
                current = current[index]
            else:
                return None
        else:
            return None
        
        if current is None:
            return None
    
    return current


def set_nested_value(data: Dict[str, Any], key_path: Union[str, List[str]], value: Any) -> None:
    """Set a value in a nested dictionary using a key path
    
    Args:
        data: The dictionary to modify
        key_path: Either a dot-separated string or a list of keys
        value: The value to set
    """
    if isinstance(key_path, str):
        keys = key_path.split('.')
    else:
        keys = key_path
    
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
