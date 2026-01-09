import json
import time
import requests
import argparse
import logging
import os
from typing import List, Dict, Any, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_cli_tools.utils.nested_utils import get_nested_value
from llm_cli_tools.utils.normalize import normalize_to_bool, normalize_to_int, normalize_to_str

# 默认配置
DEFAULT_API_URL = "http://localhost:8101/v1/chat/completions"
DEFAULT_MODEL_NAME = "review-lora"
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.6
DEFAULT_RESULT_KEY = "auditresult"
DEFAULT_WORKERS = 160  # 并发线程数


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_key_list(key_str: str):
    """
    解析键列表参数，支持逗号分隔的字符串
    
    Args:
        key_str: 可以是 "field" 或 "field1,field2,field3"
    
    Returns:
        如果是单个字段，返回字符串；如果是多个字段，返回列表
    """
    if not key_str:
        return None
    
    keys = [k.strip() for k in key_str.split(",")]
    
    # 如果只有一个键，返回字符串；多个键返回列表
    if len(keys) == 1:
        return keys[0]
    else:
        return keys


def load_test_data(filepath: str, limit: int = None) -> List[Dict[str, Any]]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def get_completed_trace_ids(jsonl_path: str) -> Set[str]:
    """从 JSONL 中读取已完成的 trace_id"""
    if not os.path.exists(jsonl_path):
        return set()
    ids = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    ids.add(item["trace_id"])
                except Exception:
                    continue
    return ids


def call_llm_once(
    case: Dict[str, Any],
    api_url: str,
    headers: dict,
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    timeout: int,
    expected_key
) -> Dict[str, Any]:
    """单次 LLM 调用，返回结构化结果（无 latency）"""
    messages = case["messages"]
    expected = get_nested_value(case["output"], expected_key)
    trace_id = case["trace_id"]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
        else:
            content = ""
    except Exception as e:
        logging.debug(f"Error on {trace_id}: {e}")
        content = ""

    return {
        "trace_id": trace_id,
        "expected": expected,
        "raw_model_output": content
    }


def run_inference_parallel(
    test_cases: List[Dict],
    completed_ids: Set[str],
    jsonl_path: str,
    api_url: str,
    headers: dict,
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    timeout: int,
    workers: int,
    expected_key: str
):
    """并行推理，并追加写入 JSONL"""
    # 过滤未完成的 case
    pending_cases = [case for case in test_cases if case["trace_id"] not in completed_ids]
    skipped = len(test_cases) - len(pending_cases)

    if skipped > 0:
        logging.info(f"Skipped {skipped} already processed cases.")

    if not pending_cases:
        logging.info("No new cases to process.")
        return

    mode = "a" if os.path.exists(jsonl_path) else "w"

    with open(jsonl_path, mode, encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交任务
            future_to_case = {
                executor.submit(
                    call_llm_once,
                    case, api_url, headers, model,
                    temperature, max_tokens, json_mode, timeout, expected_key
                ): case for case in pending_cases
            }

            # 收集结果（带进度条）
            for future in tqdm(as_completed(future_to_case), total=len(pending_cases), desc="Inferencing"):
                result = future.result()
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()  # 实时落盘


def parse_output(text: str, result_key, eval_mode: str = "binary"):
    """解析模型输出，支持多种评估模式"""
    clean_text = text.strip()
    
    # 去除 ```json ... ```
    if clean_text.startswith("```"):
        parts = clean_text.split("```")
        if len(parts) >= 3:
            inner = parts[1].strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            clean_text = inner

    # 尝试 JSON 解析
    try:
        data = json.loads(clean_text)
        val = get_nested_value(data, result_key)
        if val is not None:
            return normalize_value(val, eval_mode)
    except (json.JSONDecodeError, TypeError):
        pass

    # 回退：根据评估模式处理
    if eval_mode == "binary":
        lower_text = clean_text.lower()
        if "true" in lower_text or "yes" in lower_text or "1" in lower_text:
            return True
        else:
            return False
    elif eval_mode == "multiclass":
        # 尝试提取类别标签
        lower_text = clean_text.lower()
        for label in ["a", "b", "c", "d", "positive", "negative", "neutral"]:
            if label in lower_text:
                return label
        return clean_text  # 返回原始文本作为类别
    elif eval_mode == "regression":
        # 尝试提取数字
        import re
        numbers = re.findall(r"-?\d+\.?\d*", clean_text)
        if numbers:
            return float(numbers[0])
        return None
    elif eval_mode == "exact":
        return clean_text
    
    return None


def normalize_value(value, eval_mode: str = "binary"):
    """将值标准化为指定评估模式的格式"""
    if eval_mode == "binary":
        return normalize_to_bool(value)
    elif eval_mode == "multiclass":
        return normalize_to_str(value)
    elif eval_mode == "regression":
        try:
            return float(normalize_to_int(value))
        except (ValueError, TypeError):
            return None
    elif eval_mode == "exact":
        return str(value).strip()
    return None


def evaluate_from_jsonl(jsonl_path: str, result_key: str, eval_mode: str = "binary"):
    """评估 JSONL 文件中的结果，支持多种评估模式"""
    total = 0
    correct = 0
    errors = 0
    
    if eval_mode == "binary":
        tp = fp = fn = tn = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                expected_bool = normalize_to_bool(item["expected"])
                predicted_bool = parse_output(item["raw_model_output"], result_key, eval_mode)
                
                if predicted_bool is None:
                    errors += 1
                    total += 1
                    continue
                
                if expected_bool and predicted_bool:
                    tp += 1
                elif not expected_bool and predicted_bool:
                    fp += 1
                elif expected_bool and not predicted_bool:
                    fn += 1
                else:
                    tn += 1
                total += 1
        
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print("\n" + "=" * 60)
        print("📊 FINAL EVALUATION RESULTS (Binary Classification)")
        print("=" * 60)
        print(f"Total samples:   {total}")
        print(f"True Positives:  {tp}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Negatives:  {tn}")
        print(f"Parse Errors:    {errors}")
        print("-" * 60)
        print(f"Accuracy:        {accuracy:.4f} ({accuracy:.2%})")
        print(f"Precision:       {precision:.4f} ({precision:.2%})")
        print(f"Recall:          {recall:.4f} ({recall:.2%})")
        print(f"F1 Score:        {f1:.4f} ({f1:.2%})")
        print("=" * 60)
    
    elif eval_mode == "multiclass":
        from collections import defaultdict
        class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                expected_class = normalize_value(item["expected"], eval_mode)
                predicted_class = parse_output(item["raw_model_output"], result_key, eval_mode)
                
                if predicted_class is None:
                    errors += 1
                    total += 1
                    continue
                
                if expected_class == predicted_class:
                    correct += 1
                    class_stats[expected_class]["tp"] += 1
                else:
                    class_stats[expected_class]["fn"] += 1
                    class_stats[predicted_class]["fp"] += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        precision_macro = sum(stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0 
                             for stats in class_stats.values()) / len(class_stats) if class_stats else 0.0
        recall_macro = sum(stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0 
                          for stats in class_stats.values()) / len(class_stats) if class_stats else 0.0
        f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro) if (precision_macro + recall_macro) > 0 else 0.0
        
        print("\n" + "=" * 60)
        print("📊 FINAL EVALUATION RESULTS (Multi-class Classification)")
        print("=" * 60)
        print(f"Total samples:   {total}")
        print(f"Correct:         {correct}")
        print(f"Parse Errors:    {errors}")
        print(f"Accuracy:        {accuracy:.4f} ({accuracy:.2%})")
        print(f"Precision (macro): {precision_macro:.4f} ({precision_macro:.2%})")
        print(f"Recall (macro):    {recall_macro:.4f} ({recall_macro:.2%})")
        print(f"F1 Score (macro):  {f1_macro:.4f} ({f1_macro:.2%})")
        print("\nPer-class statistics:")
        for cls, stats in sorted(class_stats.items()):
            cls_precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.0
            cls_recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0.0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0.0
            print(f"  {cls}: P={cls_precision:.3f}, R={cls_recall:.3f}, F1={cls_f1:.3f}")
        print("=" * 60)
    
    elif eval_mode == "regression":
        import math
        errors_list = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                expected_value = normalize_value(item["expected"], eval_mode)
                predicted_value = parse_output(item["raw_model_output"], result_key, eval_mode)
                
                if predicted_value is None or expected_value is None:
                    errors += 1
                    total += 1
                    continue
                
                error = abs(predicted_value - expected_value)
                errors_list.append(error)
                total += 1
        
        if errors_list:
            mae = sum(errors_list) / len(errors_list)
            mse = sum(e ** 2 for e in errors_list) / len(errors_list)
            rmse = math.sqrt(mse)
            max_error = max(errors_list)
            min_error = min(errors_list)
        else:
            mae = mse = rmse = max_error = min_error = 0.0
        
        print("\n" + "=" * 60)
        print("📊 FINAL EVALUATION RESULTS (Regression)")
        print("=" * 60)
        print(f"Total samples:   {total}")
        print(f"Parse Errors:    {errors}")
        print(f"MAE:             {mae:.4f}")
        print(f"MSE:             {mse:.4f}")
        print(f"RMSE:            {rmse:.4f}")
        print(f"Max Error:       {max_error:.4f}")
        print(f"Min Error:       {min_error:.4f}")
        print("=" * 60)
    
    elif eval_mode == "exact":
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                expected_value = normalize_value(item["expected"], eval_mode)
                predicted_value = parse_output(item["raw_model_output"], result_key, eval_mode)
                
                if predicted_value is None:
                    errors += 1
                    total += 1
                    continue
                
                if expected_value == predicted_value:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        print("\n" + "=" * 60)
        print("📊 FINAL EVALUATION RESULTS (Exact Match)")
        print("=" * 60)
        print(f"Total samples:   {total}")
        print(f"Correct:         {correct}")
        print(f"Incorrect:       {total - correct - errors}")
        print(f"Parse Errors:    {errors}")
        print(f"Accuracy:        {accuracy:.4f} ({accuracy:.2%})")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel LLM Evaluator with JSONL & Final Stats Only. Supports multiple evaluation modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 二分类评估（默认）
  python llm_eval.py --test-file test.json --output-jsonl results.jsonl --expected-key auditresult
  
  # 多分类评估
  python llm_eval.py --test-file test.json --output-jsonl results.jsonl --eval-mode multiclass --expected-key category
  
  # 回归评估
  python llm_eval.py --test-file test.json --output-jsonl results.jsonl --eval-mode regression --expected-key score
  
  # 精确匹配评估
  python llm_eval.py --test-file test.json --output-jsonl results.jsonl --eval-mode exact --expected-key answer
  
  # 使用嵌套字段（逗号分隔）
  python llm_eval.py --test-file test.json --output-jsonl results.jsonl --result-key prediction,class --expected-key output,label
  
  # 仅推理不评估
  python llm_eval.py --test-file test.json --output-jsonl results.jsonl --only-infer
  
  # 仅评估已有结果
  python llm_eval.py --output-jsonl results.jsonl --only-eval --eval-mode binary
        """
    )
    # I/O
    parser.add_argument("--test-file", type=str, default="/data/user/jdy/Qwen30B_doc_verifi/test_data/test_data.json", help="Input test file (JSON array)")
    parser.add_argument("--output-jsonl", type=str, default="outputs.jsonl", help="Raw outputs in JSONL")

    # API
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)

    # Generation
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--json-mode", action="store_true", help="Force JSON output")

    # Parsing
    parser.add_argument("--result-key", type=str, default=DEFAULT_RESULT_KEY, 
                        help="Result field name in model output (supports nested fields like 'prediction,class')")
    parser.add_argument("--expected-key", type=str, default=DEFAULT_RESULT_KEY, 
                        help="Expected value field name in test data (supports nested fields like 'output,label')")
    parser.add_argument("--eval-mode", type=str, default="binary", 
                        choices=["binary", "multiclass", "regression", "exact"],
                        help="Evaluation mode: binary, multiclass, regression, or exact match")

    # Control
    parser.add_argument("--only-infer", action="store_true")
    parser.add_argument("--only-eval", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip existing trace_id")
    parser.add_argument("--limit", type=int, help="Limit number of test cases")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel threads")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # 解析键参数，支持逗号分隔的列表形式
    args.result_key = parse_key_list(args.result_key)
    args.expected_key = parse_key_list(args.expected_key)

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    if args.only_eval:
        evaluate_from_jsonl(args.output_jsonl, args.result_key, args.eval_mode)
        return

    # Load and filter test cases
    test_cases = load_test_data(args.test_file, args.limit)
    completed_ids = get_completed_trace_ids(args.output_jsonl) if args.resume else set()

    # Run parallel inference
    run_inference_parallel(
        test_cases, completed_ids, args.output_jsonl,
        args.api_url, headers, args.model,
        args.temperature, args.max_tokens, args.json_mode, args.timeout,
        args.workers, args.expected_key
    )

    # Evaluate and print final stats (unless only-infer)
    if not args.only_infer:
        evaluate_from_jsonl(args.output_jsonl, args.result_key, args.eval_mode)


if __name__ == "__main__":
    main()