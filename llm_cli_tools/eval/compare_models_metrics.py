import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Any
from llm_cli_tools.utils.file_utils import load_json_or_jsonl
from llm_cli_tools.utils.nested_utils import get_nested_value
from llm_cli_tools.utils.normalize import normalize_to_bool, normalize_to_int, normalize_to_str

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

CURRENT_MODEL_NAME = "current_model"


def parse_model_output(raw_text: str, result_key: str = "auditresult", eval_mode: str = "binary"):
    """解析模型输出，根据评估模式返回相应类型的值"""
    clean = raw_text.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        if len(parts) >= 3:
            inner = parts[1].strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            clean = inner

    try:
        data = json.loads(clean)
        if result_key in data:
            value = data[result_key]
            if eval_mode == "binary":
                return normalize_to_bool(value)
            elif eval_mode == "multiclass":
                return normalize_to_str(value)
            elif eval_mode == "regression":
                return normalize_to_int(value)
            else:
                return value
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        pass

    # Fallback for binary mode
    if eval_mode == "binary":
        lower = clean.lower()
        if "true" in lower or "yes" in lower or "1" in lower:
            return True
        return False
    
    # For other modes, return the raw text
    if eval_mode == "multiclass":
        return clean
    elif eval_mode == "regression":
        try:
            return normalize_to_int(clean)
        except ValueError:
            return None
    
    return None


def load_evaluation_details(gt_files: List[str], trace_id_key: str = "trace_id", 
                           evaluations_key: str = "evaluations", 
                           ground_truth_key: str = "ground_truth", 
                           predicted_key: str = "predicted", 
                           eval_mode: str = "binary") -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """加载评估详情数据"""
    gt_map = {}
    model_preds = defaultdict(lambda: defaultdict(lambda: None))
    
    logger.info(f"Loading evaluation details from {len(gt_files)} file(s)...")

    for file_idx, file in enumerate(gt_files, 1):
        try:
            data = load_json_or_jsonl(file)
            logger.info(f"Processing file {file_idx}/{len(gt_files)}: {file}")
            
            processed_count = 0
            skipped_count = 0
            
            for item_idx, item in enumerate(data, 1):
                try:
                    trace_id = get_nested_value(item, trace_id_key)
                    if trace_id is None:
                        logger.debug(f"Item {item_idx}: Missing trace_id field '{trace_id_key}'")
                        skipped_count += 1
                        continue
                        
                    evals = get_nested_value(item, evaluations_key)
                    if evals is None:
                        logger.debug(f"Item {item_idx} (trace_id={trace_id}): Missing evaluations field '{evaluations_key}'")
                        skipped_count += 1
                        continue

                    # Extract ground truth
                    gt_val = None
                    for key, val in evals.items():
                        if ground_truth_key in val:
                            gt_val = val[ground_truth_key]
                            break
                    if gt_val is None:
                        logger.debug(f"Item {item_idx} (trace_id={trace_id}): Missing ground_truth field '{ground_truth_key}'")
                        skipped_count += 1
                        continue

                    # Normalize ground truth based on evaluation mode
                    try:
                        if eval_mode == "binary":
                            gt_normalized = normalize_to_bool(gt_val)
                        elif eval_mode == "multiclass":
                            gt_normalized = normalize_to_str(gt_val)
                        elif eval_mode == "regression":
                            gt_normalized = normalize_to_int(gt_val)
                        else:
                            gt_normalized = gt_val
                        
                        gt_map[trace_id] = gt_normalized
                    except Exception as e:
                        logger.warning(f"Item {item_idx} (trace_id={trace_id}): Failed to normalize ground truth: {e}")
                        skipped_count += 1
                        continue

                    # Extract predictions only if valid (skip null/empty)
                    pred_count = 0
                    for key, val in evals.items():
                        if key.startswith("output_"):
                            pred_val = val.get(predicted_key)
                            # Skip explicitly null-like values
                            if pred_val is None:
                                continue
                            if isinstance(pred_val, str):
                                stripped = pred_val.strip()
                                if stripped == "" or stripped.lower() == "null":
                                    continue
                            
                            # Normalize prediction based on evaluation mode
                            try:
                                if eval_mode == "binary":
                                    pred_normalized = normalize_to_bool(pred_val)
                                elif eval_mode == "multiclass":
                                    pred_normalized = normalize_to_str(pred_val)
                                elif eval_mode == "regression":
                                    pred_normalized = normalize_to_int(pred_val)
                                else:
                                    pred_normalized = pred_val
                                model_preds[key][trace_id] = pred_normalized
                                pred_count += 1
                            except Exception as e:
                                logger.debug(f"Item {item_idx} (trace_id={trace_id}, model={key}): Failed to normalize prediction: {e}")
                                continue  # skip if conversion fails
                    
                    if pred_count > 0:
                        processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Item {item_idx} in {file}: Failed to process: {e}")
                    skipped_count += 1
            
            logger.info(f"File {file}: Processed {processed_count} items, skipped {skipped_count} items")
            
        except Exception as e:
            logger.error(f"Failed to load file {file}: {e}")
            raise

    logger.info(f"Total: Loaded {len(gt_map)} ground truth samples and {len(model_preds)} model predictions")
    return gt_map, dict(model_preds)


def load_difficulty_map(diff_files: List[str], trace_id_key: str = "trace_id", 
                       difficulty_key: str = "difficulty") -> Dict[str, str]:
    """加载难度映射数据"""
    diff_map = {}
    logger.info(f"Loading difficulty map from {len(diff_files)} file(s)...")
    
    for file_idx, file in enumerate(diff_files, 1):
        try:
            data = load_json_or_jsonl(file)
            logger.info(f"Processing file {file_idx}/{len(diff_files)}: {file}")
            
            processed_count = 0
            skipped_count = 0
            
            for item_idx, item in enumerate(data, 1):
                try:
                    trace_id = get_nested_value(item, trace_id_key)
                    if trace_id is None:
                        logger.debug(f"Item {item_idx}: Missing trace_id field '{trace_id_key}'")
                        skipped_count += 1
                        continue
                    
                    difficulty = get_nested_value(item, difficulty_key)
                    if difficulty is None:
                        difficulty = "unknown"
                        logger.debug(f"Item {item_idx} (trace_id={trace_id}): Missing difficulty field '{difficulty_key}', using 'unknown'")
                    
                    diff_map[trace_id] = difficulty
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Item {item_idx} in {file}: Failed to parse difficulty: {e}")
                    skipped_count += 1
            
            logger.info(f"File {file}: Processed {processed_count} items, skipped {skipped_count} items")
            
        except Exception as e:
            logger.error(f"Failed to load difficulty map from {file}: {e}")
            raise
    
    logger.info(f"Total: Loaded difficulty for {len(diff_map)} samples")
    return diff_map


def load_overall_difficulty_distribution(diff_files: List[str], 
                                         difficulty_key: str = "difficulty") -> tuple[Dict[str, int], Dict[str, float]]:
    """从完整数据集加载难度分布（用于加权）"""
    diff_counts = defaultdict(int)
    total = 0
    logger.info(f"Loading overall difficulty distribution from {len(diff_files)} file(s)...")
    
    for file_idx, file in enumerate(diff_files, 1):
        try:
            data = load_json_or_jsonl(file)
            logger.info(f"Processing file {file_idx}/{len(diff_files)}: {file}")
            
            for item_idx, item in enumerate(data, 1):
                try:
                    difficulty = get_nested_value(item, difficulty_key)
                    if difficulty is None:
                        difficulty = "unknown"
                        logger.debug(f"Item {item_idx}: Missing difficulty field '{difficulty_key}', using 'unknown'")
                    diff_counts[difficulty] += 1
                    total += 1
                except Exception as e:
                    logger.warning(f"Item {item_idx} in {file}: Failed to parse item for distribution: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load distribution from {file}: {e}")
            raise
    
    weights = {d: cnt / total for d, cnt in diff_counts.items()} if total > 0 else {}
    logger.info(f"Total samples in distribution: {total}")
    logger.info(f"Difficulty distribution: {dict(diff_counts)}")
    
    return diff_counts, weights


def compute_metrics(y_true, y_pred, eval_mode="binary"):
    """根据评估模式计算相应的指标"""
    if eval_mode == "binary":
        return compute_binary_metrics(y_true, y_pred)
    elif eval_mode == "multiclass":
        return compute_multiclass_metrics(y_true, y_pred)
    elif eval_mode == "regression":
        return compute_regression_metrics(y_true, y_pred)
    else:
        return {"error": f"Unknown evaluation mode: {eval_mode}"}


def compute_binary_metrics(y_true, y_pred):
    """计算二分类指标"""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t and p:
            tp += 1
        elif not t and p:
            fp += 1
        elif t and not p:
            fn += 1
        else:
            tn += 1
    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "total": total}


def compute_multiclass_metrics(y_true, y_pred):
    """计算多分类指标（宏平均）"""
    from collections import Counter
    
    # 获取所有类别
    all_classes = set(y_true) | set(y_pred)
    
    # 计算准确率
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / len(y_true) if y_true else 0
    
    # 计算每个类别的 precision, recall, f1
    precisions = []
    recalls = []
    f1s = []
    
    for cls in all_classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    # 宏平均
    macro_prec = sum(precisions) / len(precisions) if precisions else 0
    macro_rec = sum(recalls) / len(recalls) if recalls else 0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0
    
    return {"acc": acc, "prec": macro_prec, "rec": macro_rec, "f1": macro_f1, "total": len(y_true)}


def compute_regression_metrics(y_true, y_pred):
    """计算回归指标"""
    n = len(y_true)
    if n == 0:
        return {"mae": 0, "mse": 0, "rmse": 0, "total": 0}
    
    errors = [t - p for t, p in zip(y_true, y_pred)]
    abs_errors = [abs(e) for e in errors]
    squared_errors = [e ** 2 for e in errors]
    
    mae = sum(abs_errors) / n
    mse = sum(squared_errors) / n
    rmse = mse ** 0.5
    
    return {"mae": mae, "mse": mse, "rmse": rmse, "total": n}


def main():
    parser = argparse.ArgumentParser(description="Compare model performance metrics across different difficulty levels")
    parser.add_argument("--current-model-output", required=True,
                        help="Path to current model outputs JSONL file")
    parser.add_argument("--evaluation-files", nargs="+", required=True,
                        help="Evaluation detail JSON/JSONL files containing ground truth and other model predictions")
    parser.add_argument("--difficulty-files", nargs="+", required=True,
                        help="Difficulty JSON/JSONL files (used for both mapping and weighting)")
    parser.add_argument("--result-key", default="auditresult",
                        help="Field name for result key in model output")
    parser.add_argument("--eval-mode", default="binary", choices=["binary", "multiclass", "regression"],
                        help="Evaluation mode: binary (default), multiclass, or regression")
    parser.add_argument("--trace-id-key", default="trace_id",
                        help="Field name for trace ID")
    parser.add_argument("--difficulty-key", default="difficulty",
                        help="Field name for difficulty")
    parser.add_argument("--evaluations-key", default="evaluations",
                        help="Field name for evaluations")
    parser.add_argument("--ground-truth-key", default="ground_truth",
                        help="Field name for ground truth in evaluations")
    parser.add_argument("--predicted-key", default="predicted",
                        help="Field name for predicted value in evaluations")
    parser.add_argument("--output-path", default=None,
                        help="Output file to save results (JSON format)")
    parser.add_argument("--model-name", default=CURRENT_MODEL_NAME,
                        help="Custom name for the current model")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    args = parser.parse_args()
    
    # Set logging level
    log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)
    
    logger.info(f"Starting model comparison with evaluation mode: {args.eval_mode}")
    logger.info(f"Current model name: {args.model_name}")
    logger.info(f"Evaluation files: {args.evaluation_files}")
    logger.info(f"Difficulty files: {args.difficulty_files}")
    logger.info(f"Current model output file: {args.current_model_output}")

    # Step 1: Load ground truth & other models' predictions (skip null preds)
    logger.info("Loading evaluation details...")
    gt_map, other_model_preds = load_evaluation_details(
        args.evaluation_files,
        trace_id_key=args.trace_id_key,
        evaluations_key=args.evaluations_key,
        ground_truth_key=args.ground_truth_key,
        predicted_key=args.predicted_key,
        eval_mode=args.eval_mode
    )
    all_trace_ids = set(gt_map.keys())
    logger.info(f"Loaded {len(all_trace_ids)} samples with ground truth.")

    # Step 2: Load difficulty map
    logger.info("Loading difficulty labels...")
    diff_map = load_difficulty_map(
        args.difficulty_files,
        trace_id_key=args.trace_id_key,
        difficulty_key=args.difficulty_key
    )
    logger.info(f"Loaded difficulty for {len(diff_map)} samples.")

    # Step 3: Load overall difficulty distribution (for weighting)
    logger.info("Loading overall difficulty distribution from diff files (for weighting)...")
    overall_diff_counts, overall_diff_weights = load_overall_difficulty_distribution(
        args.difficulty_files,
        difficulty_key=args.difficulty_key
    )
    total_overall = sum(overall_diff_counts.values())
    logger.info(f"Overall difficulty distribution (N={total_overall}):")
    for d, cnt in sorted(overall_diff_counts.items()):
        logger.info(f"  {d}: {cnt} ({overall_diff_weights[d]:.4f})")

    # Step 4: Load current model predictions (skip missing/empty outputs)
    logger.info(f"Loading current model predictions from {args.current_model_output}...")
    current_preds = {}
    
    try:
        with open(args.current_model_output, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    trace_id = get_nested_value(item, args.trace_id_key)
                    if trace_id is None:
                        logger.debug(f"Line {line_num}: Missing trace_id field '{args.trace_id_key}'")
                        continue
                    if trace_id not in all_trace_ids:
                        logger.debug(f"Line {line_num}: trace_id {trace_id} not in ground truth")
                        continue
                    raw_out = item.get("raw_model_output")
                    # Skip if output is missing or empty
                    if raw_out is None:
                        logger.debug(f"Line {line_num}: Missing raw_model_output for trace_id {trace_id}")
                        continue
                    if isinstance(raw_out, str) and raw_out.strip() == "":
                        logger.debug(f"Line {line_num}: Empty raw_model_output for trace_id {trace_id}")
                        continue
                    pred = parse_model_output(str(raw_out), args.result_key, args.eval_mode)
                    if pred is not None:
                        current_preds[trace_id] = pred
                    else:
                        logger.debug(f"Line {line_num}: Failed to parse prediction for trace_id {trace_id}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Failed to parse JSON: {e}")
                except Exception as e:
                    logger.warning(f"Line {line_num}: Failed to process: {e}")
        
        logger.info(f"Loaded current model predictions for {len(current_preds)} samples")
        
    except FileNotFoundError:
        logger.error(f"Outputs file not found: {args.outputs_jsonl}")
        raise
    except Exception as e:
        logger.error(f"Failed to load current model predictions: {e}")
        raise

    # Step 5: Find common trace_ids across ALL sources
    common_ids = all_trace_ids & set(current_preds.keys()) & set(diff_map.keys())
    for model, preds in other_model_preds.items():
        if preds:  # only intersect if model has predictions
            common_ids &= set(preds.keys())

    logger.info(f"Common trace_ids across all data (after excluding null/missing): {len(common_ids)}")
    
    if len(common_ids) == 0:
        logger.error("No common trace_ids found across all data sources!")
        logger.error(f"Ground truth samples: {len(all_trace_ids)}")
        logger.error(f"Current model predictions: {len(current_preds)}")
        logger.error(f"Difficulty samples: {len(diff_map)}")
        for model, preds in other_model_preds.items():
            logger.error(f"Other model '{model}' predictions: {len(preds)}")
        raise ValueError("No common trace_ids found across all data sources")

    # Build full records
    records = []
    for tid in common_ids:
        records.append({
            "trace_id": tid,
            "difficulty": diff_map[tid],
            "gt": gt_map[tid],
            "current_pred": current_preds[tid],
            "other_preds": {m: other_model_preds[m][tid] for m in other_model_preds}
        })

    # Group by difficulty
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["difficulty"]].append(rec)

    # Get all model names
    all_models = [args.model_name] + sorted(other_model_preds.keys())

    # Step 6: Compute metrics per difficulty
    results_by_diff = {}
    for difficulty, recs in grouped.items():
        y_true = [r["gt"] for r in recs]
        model_y_pred = {
            args.model_name: [r["current_pred"] for r in recs],
        }
        for model in other_model_preds:
            model_y_pred[model] = [r["other_preds"][model] for r in recs]

        results_by_diff[difficulty] = {}
        for model in all_models:
            if model in model_y_pred:
                results_by_diff[difficulty][model] = compute_metrics(y_true, model_y_pred[model], args.eval_mode)

    # Step 7: Print per-difficulty results
    difficulties_in_eval = sorted(results_by_diff.keys())
    logger.info(f"Found {len(difficulties_in_eval)} difficulty levels in evaluation data")
    
    for difficulty in difficulties_in_eval:
        n = len(grouped[difficulty])
        weight_in_eval = n / len(common_ids)
        weight_in_overall = overall_diff_weights.get(difficulty, 0)
        logger.info(f"Difficulty: {difficulty.upper()} (Eval N={n}, Eval Weight={weight_in_eval:.4f}, Overall Weight={weight_in_overall:.4f})")
        print(f"\n{'='*90}")
        print(f"Difficulty: {difficulty.upper()} "
              f"(Eval N={n}, Eval Weight={weight_in_eval:.4f}, Overall Weight={weight_in_overall:.4f})")
        
        # Print header based on evaluation mode
        if args.eval_mode == "binary":
            print(f"{'Model':<25} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Total'}")
            print("-" * 90)
            for model in all_models:
                if model in results_by_diff[difficulty]:
                    r = results_by_diff[difficulty][model]
                    logger.info(f"  {model}: Acc={r['acc']:.4f}, Prec={r['prec']:.4f}, Rec={r['rec']:.4f}, F1={r['f1']:.4f}, Total={r['total']}")
                    print(
                        f"{model:<25} "
                        f"{r['acc']:.4f} "
                        f"{r['prec']:.4f} "
                        f"{r['rec']:.4f} "
                        f"{r['f1']:.4f} "
                        f"{r['total']}"
                    )
        elif args.eval_mode == "multiclass":
            print(f"{'Model':<25} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Total'}")
            print("-" * 90)
            for model in all_models:
                if model in results_by_diff[difficulty]:
                    r = results_by_diff[difficulty][model]
                    logger.info(f"  {model}: Acc={r['acc']:.4f}, Prec={r['prec']:.4f}, Rec={r['rec']:.4f}, F1={r['f1']:.4f}, Total={r['total']}")
                    print(
                        f"{model:<25} "
                        f"{r['acc']:.4f} "
                        f"{r['prec']:.4f} "
                        f"{r['rec']:.4f} "
                        f"{r['f1']:.4f} "
                        f"{r['total']}"
                    )
        elif args.eval_mode == "regression":
            print(f"{'Model':<25} {'MAE':<8} {'MSE':<8} {'RMSE':<8} {'Total'}")
            print("-" * 90)
            for model in all_models:
                if model in results_by_diff[difficulty]:
                    r = results_by_diff[difficulty][model]
                    logger.info(f"  {model}: MAE={r['mae']:.4f}, MSE={r['mse']:.4f}, RMSE={r['rmse']:.4f}, Total={r['total']}")
                    print(
                        f"{model:<25} "
                        f"{r['mae']:.4f} "
                        f"{r['mse']:.4f} "
                        f"{r['rmse']:.4f} "
                        f"{r['total']}"
                    )

    # Step 8: Compute WEIGHTED METRICS using OVERALL difficulty distribution
    logger.info("Computing weighted overall metrics...")
    weighted_results = {}
    for model in all_models:
        if args.eval_mode == "regression":
            weighted_results[model] = {"mae": 0.0, "mse": 0.0, "rmse": 0.0}
        else:
            weighted_results[model] = {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}

    for difficulty, weight in overall_diff_weights.items():
        if difficulty not in results_by_diff:
            logger.warning(f"Difficulty '{difficulty}' exists in overall data but has no evaluation samples. Skipping.")
            continue
        for model in all_models:
            if model in results_by_diff[difficulty]:
                r = results_by_diff[difficulty][model]
                if args.eval_mode == "regression":
                    weighted_results[model]["mae"] += weight * r["mae"]
                    weighted_results[model]["mse"] += weight * r["mse"]
                    weighted_results[model]["rmse"] += weight * r["rmse"]
                else:
                    weighted_results[model]["acc"] += weight * r["acc"]
                    weighted_results[model]["prec"] += weight * r["prec"]
                    weighted_results[model]["rec"] += weight * r["rec"]
                    weighted_results[model]["f1"] += weight * r["f1"]

    # Step 9: Print weighted overall metrics
    logger.info("Printing weighted overall metrics...")
    print(f"\n{'='*90}")
    print(f"OVERALL WEIGHTED METRICS")
    print(f"Weighted by difficulty distribution in:")
    for df in args.diff_files:
        print(f"  - {df}")
    print(f"Total samples in distribution: {total_overall}")
    
    if args.eval_mode == "regression":
        print(f"{'Model':<25} {'MAE':<8} {'MSE':<8} {'RMSE':<8}")
        print("-" * 90)
        for model in all_models:
            r = weighted_results[model]
            logger.info(f"  {model}: MAE={r['mae']:.4f}, MSE={r['mse']:.4f}, RMSE={r['rmse']:.4f}")
            print(
                f"{model:<25} "
                f"{r['mae']:.4f} "
                f"{r['mse']:.4f} "
                f"{r['rmse']:.4f}"
            )
    else:
        print(f"{'Model':<25} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
        print("-" * 90)
        for model in all_models:
            r = weighted_results[model]
            logger.info(f"  {model}: Acc={r['acc']:.4f}, Prec={r['prec']:.4f}, Rec={r['rec']:.4f}, F1={r['f1']:.4f}")
            print(
                f"{model:<25} "
                f"{r['acc']:.4f} "
                f"{r['prec']:.4f} "
                f"{r['rec']:.4f} "
                f"{r['f1']:.4f}"
            )
    print(f"{'='*90}")

    # Step 10: Save results to file if specified
    if args.output_path:
        logger.info(f"Saving results to {args.output_path}...")
        output_data = {
            "eval_mode": args.eval_mode,
            "model_name": args.model_name,
            "total_samples": len(common_ids),
            "overall_difficulty_distribution": {
                "counts": overall_diff_counts,
                "weights": overall_diff_weights
            },
            "results_by_difficulty": results_by_diff,
            "weighted_overall_metrics": weighted_results
        }
        try:
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved successfully to: {args.output_path}")
            print(f"\nResults saved to: {args.output_path}")
        except Exception as e:
            logger.error(f"Failed to save results to {args.output_path}: {e}")
            raise
    
    logger.info("Model comparison completed successfully")


if __name__ == "__main__":
    main()