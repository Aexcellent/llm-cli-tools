"""
LLM CLI Tools - A comprehensive toolkit for LLM inference, evaluation, and data processing
"""

__version__ = "0.1.0"
__author__ = "Deyou Jiang"
__email__ = "jiangdeyou@inspur.com"

from llm_cli_tools.utils import (
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
    get_nested_value,
    set_nested_value,
    normalize_to_bool,
    normalize_to_int,
    normalize_to_str,
)

__all__ = [
    "load_json",
    "load_jsonl",
    "save_json",
    "save_jsonl",
    "get_nested_value",
    "set_nested_value",
    "normalize_to_bool",
    "normalize_to_int",
    "normalize_to_str",
]
