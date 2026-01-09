"""
Data normalization utilities for different evaluation modes
"""

from typing import Any


def normalize_to_bool(value: Any) -> bool:
    """Normalize a value to boolean for binary evaluation
    
    Args:
        value: The value to normalize
        
    Returns:
        Boolean representation of the value
        
    Raises:
        ValueError: If the value cannot be converted to boolean
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "on", "correct", "right"):
            return True
        if v in ("false", "0", "no", "off", "incorrect", "wrong"):
            return False
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, float):
        return bool(value)
    raise ValueError(f"Cannot convert {value} to boolean")


def normalize_to_int(value: Any) -> int:
    """Normalize a value to integer for regression evaluation
    
    Args:
        value: The value to normalize
        
    Returns:
        Integer representation of the value
        
    Raises:
        ValueError: If the value cannot be converted to integer
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            pass
    raise ValueError(f"Cannot convert {value} to int")


def normalize_to_str(value: Any) -> str:
    """Normalize a value to string for multiclass evaluation
    
    Args:
        value: The value to normalize
        
    Returns:
        String representation of the value
    """
    if isinstance(value, str):
        return value.strip()
    return str(value)
