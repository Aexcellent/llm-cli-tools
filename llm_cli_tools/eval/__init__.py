"""
Evaluation module for LLM model assessment
"""

from llm_cli_tools.eval.llm_eval import main as eval_main
from llm_cli_tools.eval.compare_models_metrics import main as compare_main

__all__ = ["eval_main", "compare_main"]
