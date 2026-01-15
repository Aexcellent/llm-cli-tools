"""
Data processing module for JSON/JSONL manipulation
"""

from llm_cli_tools.data.merge_jsonl import main as merge_main
from llm_cli_tools.data.convert2sftdata import main as convert_main
from llm_cli_tools.data.build_dpo import main as dpo_main
from llm_cli_tools.data.clean_failed_data import main as clean_main

__all__ = ["merge_main", "convert_main", "dpo_main", "clean_main"]
