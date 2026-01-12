import json
import os
import threading
import argparse
import shutil
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# 线程安全写入锁
file_lock = threading.Lock()

# 默认评判 Prompt（多轮对话格式）
DEFAULT_JUDGE_PROMPT = """你是一个专业的评测专家，请对模型回答进行评分。

## 评分标准：
- 准确性（0-10分）
- 完整性（0-10分）
- 逻辑性（0-10分）
- 语言表达（0-10分）

请按照以下格式输出：
总分：XX/40
详细评分：
- 准确性：XX/10
- 完整性：XX/10
- 逻辑性：XX/10
- 语言表达：XX/10
评价：...
"""


def ensure_ids_and_save(data_path):
    """
    检查文件（JSON 或 JSONL）中的样本是否有 id 字段。
    如果没有，自动分配索引作为 id，并保存修改回源文件。
    返回修改后的数据列表。
    """
    print(f"\n正在检查文件: {data_path}")
    
    path = Path(data_path)
    is_jsonl = path.suffix.lower() == '.jsonl'
    
    data = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            if is_jsonl:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"  警告：第 {line_num + 1} 行 JSON 解析失败，已跳过")
                print(f"  格式检测: JSONL (共 {len(data)} 行)")
            else:
                data = json.load(f)
                print(f"  格式检测: JSON Array (共 {len(data)} 项)")
                
    except json.JSONDecodeError as e:
        raise ValueError(f"文件 {data_path} 不是有效的 JSON 格式: {e}")
    except FileNotFoundError:
        raise ValueError(f"文件不存在: {data_path}")

    if not isinstance(data, list):
        raise ValueError(f"文件内容必须是列表格式 (JSON Array 或 JSONL)")

    modified = False
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
            
        if "id" not in item:
            item["id"] = i
            modified = True
            
    if modified:
        print(f"  → 检测到缺失 ID，正在保存修改到源文件...")
        
        backup_path = str(path) + ".bak"
        shutil.copy(path, backup_path)
        print(f"  → 已备份原文件到: {backup_path}")
        
        with open(path, "w", encoding="utf-8") as f:
            if is_jsonl:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        print(f"  ✓ 源文件 {data_path} 已更新 (包含 ID)")
    else:
        print(f"  ✓ 所有样本均已有 ID，无需修改源文件")
    
    return data


def load_processed_ids(jsonl_path):
    """加载已处理的 ID 集合，用于断点续传"""
    ids = set()
    if Path(jsonl_path).exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "id" in data:
                            ids.add(data["id"])
                    except json.JSONDecodeError:
                        continue
    return ids


def extract_score_from_output(output):
    """从评判输出中提取分数"""
    if not output:
        return None
    
    output = output.replace('*','')
    
    match = re.search(r'总分[::：]\s*(\d+(?:\.\d+)?)', output)
    if match:
        return float(match.group(1))
    
    match = re.search(r'score[::：]\s*(\d+(?:\.\d+)?)', output, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    match = re.search(r'(总分|总分（100分制）)\s*[|=\-：:]\s*(\d+)\s*/\s*(\d+)', output)
    if match:
        return float(match.group(2))
    
    return None


def load_judge_prompt(prompt_file=None):
    """加载评判 Prompt"""
    if prompt_file and Path(prompt_file).exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        try:
            from prompts import LLM_JUDGE_PROMPT
            return LLM_JUDGE_PROMPT
        except ImportError:
            print("警告：未找到 prompts 模块，使用默认评判 Prompt")
            return DEFAULT_JUDGE_PROMPT


def process_inference(item_id, messages, client, model, save_path, temperature, max_tokens, round_num=None, preserved_fields=None):
    """处理单个推理请求"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens 
        )
        result = {
            "id": item_id,
            "messages": messages,
            "output": response.choices[0].message.content,
            "success": True,
        }
        if round_num is not None:
            result["round"] = round_num
        if preserved_fields:
            result.update(preserved_fields)
    except Exception as e:
        result = {
            "id": item_id,
            "messages": messages,
            "output": None,
            "success": False,
            "error": str(e),
        }
        if round_num is not None:
            result["round"] = round_num
        if preserved_fields:
            result.update(preserved_fields)
    
    with file_lock:
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def process_judge(item_id, messages, client, model, save_path, temperature, max_tokens, 
                  original_input=None, original_output=None, judge_round=None, preserved_fields=None):
    """处理单个评判请求"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens 
        )
        output_text = response.choices[0].message.content
        score = extract_score_from_output(output_text)
        
        result = {
            "id": item_id,
            "messages": messages,
            "output": output_text,
            "score": score,
            "success": True,
        }
        
        if judge_round is not None:
            result["judge_round"] = judge_round
        if original_input is not None:
            result["original_input"] = original_input
        if original_output is not None:
            result["original_output"] = original_output
        if preserved_fields:
            for field, value in preserved_fields.items():
                result[field] = value
            
    except Exception as e:
        result = {
            "id": item_id,
            "messages": messages,
            "output": None,
            "score": None,
            "success": False,
            "error": str(e),
        }
        if judge_round is not None:
            result["judge_round"] = judge_round
        if original_input is not None:
            result["original_input"] = original_input
        if original_output is not None:
            result["original_output"] = original_output
        if preserved_fields:
            for field, value in preserved_fields.items():
                result[field] = value
    
    with file_lock:
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def get_api_config(args):
    """获取API配置"""
    if args.api_key:
        api_key = args.api_key
    else:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("请通过 --api-key 参数或环境变量（DASHSCOPE_API_KEY/OPENAI_API_KEY）提供 API Key")
    
    if args.base_url:
        base_url = args.base_url
    else:
        model_lower = args.model.lower()
        if "qwen" in model_lower or "dashscope" in model_lower or "deepseek" in model_lower:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        else:
            base_url = "https://api.openai.com/v1"
    
    return api_key, base_url


def prepare_inference_tasks(data, preserve_fields=None):
    """准备推理任务
    
    支持两种数据格式：
    1. 包含 messages 字段的数据（直接使用）
    2. 包含 instruction 和 input 字段的数据（转换为 messages 格式）
    """
    test_items = []
    
    # 解析需要保留的字段
    preserve_fields_list = []
    if preserve_fields:
        preserve_fields_list = [field.strip() for field in preserve_fields.split(",")]
    
    for item in data:
        item_id = item.get("id")
        if item_id is None:
            continue
        
        # 检查是否包含 messages 字段
        if "messages" in item:
            # 直接使用现有的 messages
            messages = item["messages"]
        else:
            # 从 instruction 和 input 构造 messages
            messages = [
                {"role": "system", "content": item.get("instruction", "")},
                {"role": "user", "content": item.get("input", "")}
            ]
        
        task_item = {"id": item_id, "messages": messages}
        
        # 保留用户指定的字段
        for field in preserve_fields_list:
            if field in item:
                task_item[field] = item[field]
        
        test_items.append(task_item)
    
    return test_items


def prepare_judge_tasks(data, judge_prompt, skip_no_output=False, save_original=False, preserve_fields=None):
    """准备评判任务（使用多轮对话格式）"""
    test_items = []
    skipped_no_output = 0
    skipped_no_input = 0
    
    # 解析需要保留的字段
    preserve_fields_list = []
    if preserve_fields:
        preserve_fields_list = [field.strip() for field in preserve_fields.split(",")]
    
    for item in data:

        output_text = item.get("output")
        
        if skip_no_output and not output_text:
            skipped_no_output += 1
            continue
        
        # 构造多轮对话格式的 messages
        messages = [
            *item.get("messages", []),
            {"role": "assistant", "content": output_text or ""},
            {"role": "system", "content": judge_prompt},

        ]
        
        task_item = {
            "id": item.get("id"),
            "messages": messages,
            "original_input": json.dumps(item.get("messages", []), ensure_ascii=False) if save_original else None,
            "original_output": output_text if save_original else None
        }
        
        # 保留用户指定的字段
        for field in preserve_fields_list:
            if field in item:
                task_item[field] = item[field]
        
        test_items.append(task_item)
    
    return test_items, skipped_no_output, skipped_no_input


def run_inference(args, api_key, base_url):
    """运行推理模式"""
    data_path = args.input_path
    model_name = args.model
    max_workers = args.max_workers
    temperature = args.temperature
    max_tokens = args.max_tokens
    preserve_fields = args.preserve_fields
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.output_file:
        jsonl_file = output_dir / args.output_file
    else:
        safe_model_name = model_name.replace("-", "_").replace("/", "_")
        base_name = Path(data_path).stem
        jsonl_file = output_dir / f"{safe_model_name}_{base_name}.jsonl"

    print("=" * 60)
    print("=== 推理模式配置 ===")
    print(f"数据路径: {data_path}")
    print(f"模型名称: {model_name}")
    print(f"API Base URL: {base_url}")
    print(f"并发线程数: {max_workers}")
    print(f"温度参数: {temperature}")
    print(f"最大 Token: {max_tokens}")
    print(f"输出文件: {jsonl_file}")
    if preserve_fields:
        print(f"保留字段: {preserve_fields}")
    print("=" * 60)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_items = prepare_inference_tasks(data, preserve_fields=preserve_fields)
    print(f"Total items: {len(test_items)}")

    processed_ids = load_processed_ids(jsonl_file)
    items_to_process = [item for item in test_items if item["id"] not in processed_ids]

    if not items_to_process:
        print("\n✓ 所有项目已处理完成！")
        return

    print(f"待处理项目数: {len(items_to_process)}")
    print(f"已跳过项目数: {len(processed_ids)}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    print("\n开始处理...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_inference,
                item["id"],
                item["messages"],
                client,
                model_name,
                jsonl_file,
                temperature,
                max_tokens,
                None,
                {k: v for k, v in item.items() if k not in ["id", "messages"]}
            )
            for item in items_to_process
        ]

        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                status = "✓" if result["success"] else "✗"
                pbar.set_postfix({"id": result["id"], "status": status})

    print("\n" + "=" * 60)
    print("=== 最终统计 ===")
    
    final_results = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                final_results.append(json.loads(line))

    successful = sum(1 for r in final_results if r["success"])
    total = len(final_results)
    
    print(f"总处理数: {total}")
    print(f"成功数: {successful}")
    print(f"失败数: {total - successful}")
    if total > 0:
        print(f"成功率: {successful / total * 100:.2f}%")
    print("=" * 60)


def run_inference_round(args, api_key, base_url):
    """运行多轮推理模式 - 所有结果存储到单个文件，每条数据包含 round 字段"""
    data_path = args.input_path
    model_name = args.model
    max_workers = args.max_workers
    temperature = args.temperature
    max_tokens = args.max_tokens
    num_rounds = args.rounds
    preserve_fields = args.preserve_fields
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    safe_model_name = model_name.replace("-", "_").replace("/", "_")
    base_name = Path(data_path).stem
    
    # 使用单个输出文件
    if args.output_file:
        jsonl_file = output_dir / args.output_file
    else:
        jsonl_file = output_dir / f"{safe_model_name}_{base_name}_rounds.jsonl"

    print("=" * 60)
    print("=== 多轮推理模式配置 ===")
    print(f"数据路径: {data_path}")
    print(f"模型名称: {model_name}")
    print(f"API Base URL: {base_url}")
    print(f"并发线程数: {max_workers}")
    print(f"温度参数: {temperature}")
    print(f"最大 Token: {max_tokens}")
    print(f"推理轮数: {num_rounds}")
    print(f"输出文件: {jsonl_file} (所有轮次结果存储在单个文件中)")
    if preserve_fields:
        print(f"保留字段: {preserve_fields}")
    print("=" * 60)

    data = ensure_ids_and_save(data_path)
    test_items = prepare_inference_tasks(data, preserve_fields=preserve_fields)
    
    print(f"Total samples: {len(test_items)}")
    print(f"Total tasks: {len(test_items) * num_rounds} (每个样本 {num_rounds} 轮)")

    # 加载已处理的 (id, round) 组合
    processed_pairs = set()
    if Path(jsonl_file).exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "id" in data and "round" in data:
                            processed_pairs.add((data["id"], data["round"]))
                    except json.JSONDecodeError:
                        continue

    # 准备所有任务
    all_tasks = []
    for round_num in range(1, num_rounds + 1):
        for item in test_items:
            if (item["id"], round_num) not in processed_pairs:
                # 提取保留的字段（排除标准字段）
                preserved_fields_dict = {k: v for k, v in item.items() 
                                         if k not in ["id", "messages"]}
                all_tasks.append({
                    "id": item["id"],
                    "round": round_num,
                    "messages": item["messages"],
                    "preserved_fields": preserved_fields_dict
                })

    if not all_tasks:
        print("\n✓ 所有项目已处理完成！")
        return

    print(f"待处理任务数: {len(all_tasks)}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    print("\n开始处理...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_inference,
                task["id"],
                task["messages"],
                client,
                model_name,
                jsonl_file,
                temperature,
                max_tokens,
                task["round"],
                task["preserved_fields"]
            )
            for task in all_tasks
        ]

        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                status = "✓" if result["success"] else "✗"
                round_info = result.get("round", "N/A")
                pbar.set_postfix({"id": result["id"], "round": round_info, "status": status})

    print("\n" + "=" * 60)
    print("=== 最终统计 ===")
    
    # 统计所有结果
    final_results = []
    if Path(jsonl_file).exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    final_results.append(json.loads(line))

    total = len(final_results)
    successful = sum(1 for r in final_results if r["success"])
    
    print(f"总处理数: {total}")
    print(f"成功数: {successful}")
    print(f"失败数: {total - successful}")
    if total > 0:
        print(f"成功率: {successful / total * 100:.2f}%")
    
    # 按轮次统计
    print(f"\n=== 按轮次统计 ===")
    for round_num in range(1, num_rounds + 1):
        round_results = [r for r in final_results if r.get("judge_round") == round_num]
        round_success = sum(1 for r in round_results if r["success"])
        round_total = len(round_results)
        if round_total > 0:
            print(f"第 {round_num} 轮: 成功 {round_success}/{round_total} ({round_success/round_total*100:.2f}%)")
        else:
            print(f"第 {round_num} 轮: 无数据")
    
    # 样本完整性统计
    item_stats = {}
    for r in final_results:
        item_id = r["id"]
        if item_id not in item_stats:
            item_stats[item_id] = {"success": 0, "total": 0}
        item_stats[item_id]["total"] += 1
        if r["success"]:
            item_stats[item_id]["success"] += 1
    
    all_success_count = sum(1 for stats in item_stats.values() if stats["success"] == num_rounds)
    partial_success_count = sum(1 for stats in item_stats.values() if 0 < stats["success"] < num_rounds)
    all_failed_count = sum(1 for stats in item_stats.values() if stats["success"] == 0)
    
    print(f"\n=== 样本完整性统计 ===")
    print(f"所有轮次都成功: {all_success_count} 个样本")
    print(f"部分轮次成功: {partial_success_count} 个样本 (建议查看日志)")
    print(f"所有轮次都失败: {all_failed_count} 个样本")
    
    print("=" * 60)
    print(f"✓ 结果已保存到: {jsonl_file}")


def run_judge(args, api_key, base_url):
    """运行评判模式"""
    data_path = args.input_path
    model_name = args.model
    max_workers = args.max_workers
    temperature = args.temperature
    max_tokens = args.max_tokens
    skip_no_output = args.skip_no_output
    save_original = args.save_original
    preserve_fields = args.preserve_fields
    
    judge_prompt = load_judge_prompt(args.prompt_file)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.output_file:
        jsonl_file = output_dir / args.output_file
    else:
        input_name = Path(data_path).stem
        safe_model_name = model_name.replace("-", "_").replace("/", "_")
        jsonl_file = output_dir / f"{input_name}_{safe_model_name}_judge.jsonl"

    print("=" * 60)
    print("=== 评判模式配置 ===")
    print(f"输入文件: {data_path}")
    print(f"模型名称: {model_name}")
    print(f"API Base URL: {base_url}")
    print(f"并发线程数: {max_workers}")
    print(f"温度参数: {temperature}")
    print(f"最大 Token: {max_tokens}")
    print(f"输出文件: {jsonl_file}")
    print(f"跳过空输出: {skip_no_output}")
    print(f"保存原始数据: {save_original}")
    if preserve_fields:
        print(f"保留字段: {preserve_fields}")
    print("=" * 60)

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告：第 {line_num} 行解析失败，已跳过")
                continue

    print(f"读取数据行数: {len(data)}")

    test_items, skipped_no_output, skipped_no_input = prepare_judge_tasks(
        data, judge_prompt, skip_no_output, save_original, preserve_fields
    )

    print(f"有效评判任务数: {len(test_items)}")
    if skipped_no_input > 0:
        print(f"跳过无输入样本: {skipped_no_input}")
    if skipped_no_output > 0:
        print(f"跳过无输出样本: {skipped_no_output}")

    processed_ids = load_processed_ids(jsonl_file)
    items_to_process = [item for item in test_items if item["id"] not in processed_ids]

    if not items_to_process:
        print("\n✓ 所有项目已处理完成！")
        return

    print(f"待处理项目数: {len(items_to_process)}")
    print(f"已跳过项目数: {len(processed_ids)}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    print("\n开始评判...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_judge,
                item["id"],
                item["messages"],
                client,
                model_name,
                jsonl_file,
                temperature,
                max_tokens,
                item["original_input"],
                item["original_output"],
                None,
                {k: v for k, v in item.items() if k not in ["id", "messages", "original_input", "original_output"]}
            )
            for item in items_to_process
        ]

        with tqdm(total=len(futures), desc="Judging") as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                status = "✓" if result["success"] else "✗"
                score_info = result.get("score", "N/A")
                pbar.set_postfix({"id": result["id"], "status": status, "score": score_info})

    print("\n" + "=" * 60)
    print("=== 最终统计 ===")
    
    final_results = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                final_results.append(json.loads(line))

    successful = sum(1 for r in final_results if r["success"])
    failed = sum(1 for r in final_results if not r["success"])
    total = len(final_results)
    
    scores = [r["score"] for r in final_results if r["success"] and r.get("score") is not None]
    scores_none = sum(1 for r in final_results if r["success"] and r.get("score") is None)
    
    print(f"总处理数: {total}")
    print(f"成功数: {successful}")
    print(f"失败数: {failed}")
    if total > 0:
        print(f"成功率: {successful / total * 100:.2f}%")
    
    if scores:
        print(f"\n=== 分数统计 ===")
        print(f"有效分数样本数: {len(scores)}")
        print(f"未提取到分数的成功样本: {scores_none}")
        print(f"平均分: {sum(scores) / len(scores):.2f}")
        print(f"最高分: {max(scores):.2f}")
        print(f"最低分: {min(scores):.2f}")
        if len(scores) > 1:
            sorted_scores = sorted(scores)
            print(f"中位数: {sorted_scores[len(scores)//2]:.2f}")
            print(f"标准差: {(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5:.2f}")
    
    print("=" * 60)
    print(f"✓ 结果已保存到: {jsonl_file}")


def run_judge_round(args, api_key, base_url):
    """运行多轮评判模式 - 所有结果存储到单个文件，每条数据包含 judge_round 字段"""
    data_path = args.input_path
    model_name = args.model
    max_workers = args.max_workers
    temperature = args.temperature
    max_tokens = args.max_tokens
    num_rounds = args.rounds
    skip_no_output = args.skip_no_output
    save_original = args.save_original
    preserve_fields = args.preserve_fields
    
    judge_prompt = load_judge_prompt(args.prompt_file)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    input_name = Path(data_path).stem
    safe_model_name = model_name.replace("-", "_").replace("/", "_")
    
    # 使用单个输出文件
    if args.output_file:
        jsonl_file = output_dir / args.output_file
    else:
        jsonl_file = output_dir / f"{input_name}_{safe_model_name}_judge_rounds.jsonl"

    print("=" * 60)
    print("=== 多轮评判模式配置 ===")
    print(f"输入文件: {data_path}")
    print(f"模型名称: {model_name}")
    print(f"API Base URL: {base_url}")
    print(f"并发线程数: {max_workers}")
    print(f"温度参数: {temperature}")
    print(f"最大 Token: {max_tokens}")
    print(f"评判轮数: {num_rounds}")
    print(f"输出文件: {jsonl_file} (所有轮次结果存储在单个文件中)")
    print(f"跳过空输出: {skip_no_output}")
    print(f"保存原始数据: {save_original}")
    if preserve_fields:
        print(f"保留字段: {preserve_fields}")
    print("=" * 60)

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告：第 {line_num} 行解析失败，已跳过")
                continue

    print(f"读取数据行数: {len(data)}")

    test_items, skipped_no_output, skipped_no_input = prepare_judge_tasks(
        data, judge_prompt, skip_no_output, save_original, preserve_fields
    )

    print(f"有效评判任务数: {len(test_items)}")
    if skipped_no_input > 0:
        print(f"跳过无输入样本: {skipped_no_input}")
    if skipped_no_output > 0:
        print(f"跳过无输出样本: {skipped_no_output}")

    # 加载已处理的 (id, judge_round) 组合
    processed_pairs = set()
    if Path(jsonl_file).exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "id" in data and "judge_round" in data:
                            processed_pairs.add((data["id"], data["judge_round"]))
                    except json.JSONDecodeError:
                        continue

    # 准备所有任务
    all_tasks = []
    for round_num in range(1, num_rounds + 1):
        for item in test_items:
            if (item["id"], round_num) not in processed_pairs:
                # 提取保留的字段（排除标准字段）
                preserved_fields = {k: v for k, v in item.items() 
                                  if k not in ["id", "messages", "original_input", "original_output"]}
                all_tasks.append({
                    "id": item["id"],
                    "judge_round": round_num,
                    "messages": item["messages"],
                    "original_input": item["original_input"],
                    "original_output": item["original_output"],
                    "preserved_fields": preserved_fields
                })

    if not all_tasks:
        print("\n✓ 所有项目已处理完成！")
        return

    print(f"待处理任务数: {len(all_tasks)}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    print("\n开始评判...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_judge,
                task["id"],
                task["messages"],
                client,
                model_name,
                jsonl_file,
                temperature,
                max_tokens,
                task["original_input"],
                task["original_output"],
                task["judge_round"],
                task["preserved_fields"]
            )
            for task in all_tasks
        ]

        with tqdm(total=len(futures), desc="Judging") as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                status = "✓" if result["success"] else "✗"
                score_info = result.get("score", "N/A")
                round_info = result.get("judge_round", "N/A")
                pbar.set_postfix({
                    "id": result["id"],
                    "judge_round": round_info,
                    "status": status,
                    "score": score_info
                })

    print("\n" + "=" * 60)
    print("=== 最终统计 ===")
    
    # 统计所有结果
    final_results = []
    if Path(jsonl_file).exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    final_results.append(json.loads(line))

    successful = sum(1 for r in final_results if r["success"])
    failed = sum(1 for r in final_results if not r["success"])
    total = len(final_results)
    
    scores = [r["score"] for r in final_results if r["success"] and r.get("score") is not None]
    scores_none = sum(1 for r in final_results if r["success"] and r.get("score") is None)
    
    print(f"总处理数: {total}")
    print(f"成功数: {successful}")
    print(f"失败数: {failed}")
    if total > 0:
        print(f"成功率: {successful / total * 100:.2f}%")
    
    # 按轮次统计
    print(f"\n=== 按轮次统计 ===")
    for round_num in range(1, num_rounds + 1):
        round_results = [r for r in final_results if r.get("judge_round") == round_num]
        round_success = sum(1 for r in round_results if r["success"])
        round_total = len(round_results)
        if round_total > 0:
            print(f"第 {round_num} 轮: 成功 {round_success}/{round_total} ({round_success/round_total*100:.2f}%)")
        else:
            print(f"第 {round_num} 轮: 无数据")
    
    if scores:
        print(f"\n=== 分数统计 ===")
        print(f"有效分数样本数: {len(scores)}")
        print(f"未提取到分数的成功样本: {scores_none}")
        print(f"平均分: {sum(scores) / len(scores):.2f}")
        print(f"最高分: {max(scores):.2f}")
        print(f"最低分: {min(scores):.2f}")
        if len(scores) > 1:
            sorted_scores = sorted(scores)
            print(f"中位数: {sorted_scores[len(scores)//2]:.2f}")
            print(f"标准差: {(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5:.2f}")
    
    # 样本完整性统计
    item_stats = {}
    for r in final_results:
        item_id = r["id"]
        if item_id not in item_stats:
            item_stats[item_id] = {"success": 0, "total": 0}
        item_stats[item_id]["total"] += 1
        if r["success"]:
            item_stats[item_id]["success"] += 1
    
    all_success_count = sum(1 for stats in item_stats.values() if stats["success"] == num_rounds)
    partial_success_count = sum(1 for stats in item_stats.values() if 0 < stats["success"] < num_rounds)
    all_failed_count = sum(1 for stats in item_stats.values() if stats["success"] == 0)
    
    print(f"\n=== 样本完整性统计 ===")
    print(f"所有轮次都成功: {all_success_count} 个样本")
    print(f"部分轮次成功: {partial_success_count} 个样本 (建议查看日志)")
    print(f"所有轮次都失败: {all_failed_count} 个样本")
    
    print("=" * 60)
    print(f"✓ 结果已保存到: {jsonl_file}")


def main():
    parser = argparse.ArgumentParser(
        description="通用 LLM 命令行工具 - 支持推理、多轮推理、评判、多轮评判",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 推理模式
  python llm_cli_unified.py --mode inference --input-path data.json --model qwen-plus
  
  # 多轮推理模式
  python llm_cli_unified.py --mode inference-round --input-path data.json --model qwen-plus --rounds 3
  
  # 评判模式
  python llm_cli_unified.py --mode judge --input-path results.jsonl --model deepseek-v3.2
  
  # 多轮评判模式
  python llm_cli_unified.py --mode judge-round --input-path results.jsonl --model deepseek-v3.2
        """
    )
    
    parser.add_argument("--mode", type=str, required=True,
                        choices=["inference", "inference-round", "judge", "judge-round"],
                        help="运行模式: inference(推理), inference-round(多轮推理), judge(评判), judge-round(多轮评判)")
    
    # 通用参数
    parser.add_argument("--input-path", type=str, required=True,
                        help="输入文件路径（JSON 或 JSONL 格式）")
    parser.add_argument("--model", type=str, required=True,
                        help="模型名称，如: qwen-plus, gpt-4, deepseek-chat")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API Key（不指定则从环境变量获取）")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API Base URL（不指定则使用默认）")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="输出目录（默认: results）")
    parser.add_argument("--output-file", type=str, default=None,
                        help="输出文件名（不指定则自动生成）")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="并发线程数（默认: 10）")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="温度参数（默认: 0.6）")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="最大 token 数（默认: 4096）")
    
    # 多轮模式参数
    parser.add_argument("--rounds", type=int, default=1,
                        help="每个样本获取结果的轮数（默认: 1）")
    
    # 评判模式参数
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="自定义评判 Prompt 文件路径")
    parser.add_argument("--skip-no-output", action="store_true",
                        help="跳过 output 为 None 的样本")
    parser.add_argument("--save-original", action="store_true",
                        help="保存原始输入和输出到结果文件中（用于后续分析）")
    parser.add_argument("--preserve-fields", type=str, default=None,
                        help="从输入数据中保留到结果中的字段列表（逗号分隔），如: round,custom_field")
    
    args = parser.parse_args()
    
    # 获取API配置
    api_key, base_url = get_api_config(args)
    
    # 根据模式运行相应功能
    if args.mode == "inference":
        run_inference(args, api_key, base_url)
    elif args.mode == "inference-round":
        run_inference_round(args, api_key, base_url)
    elif args.mode == "judge":
        run_judge(args, api_key, base_url)
    elif args.mode == "judge-round":
        run_judge_round(args, api_key, base_url)


if __name__ == "__main__":
    main()
