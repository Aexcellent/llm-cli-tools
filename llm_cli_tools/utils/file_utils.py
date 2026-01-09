"""
File utility functions for loading and saving JSON/JSONL files
"""

import json
import logging
from pathlib import Path
from typing import Any, List, Dict, Union

logger = logging.getLogger(__name__)


def load_json(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load data from a JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing the data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    return [data]


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load data from a JSONL file
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
                continue
    
    return data


def load_json_or_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load data from a JSON or JSONL file (auto-detect format)
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of dictionaries containing the data
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.jsonl':
        return load_jsonl(file_path)
    else:
        return load_json(file_path)


def save_json(data: List[Dict[str, Any]], file_path: Union[str, Path], indent: int = 2) -> None:
    """Save data to a JSON file
    
    Args:
        data: List of dictionaries to save
        file_path: Path to the output file
        indent: Number of spaces for indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    
    logger.info(f"Saved {len(data)} items to {file_path}")


def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """Save data to a JSONL file
    
    Args:
        data: List of dictionaries to save
        file_path: Path to the output file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(data)} items to {file_path}")
