import json
import os
from collections import defaultdict, Counter
import argparse
from pathlib import Path
from llm_cli_tools.utils.file_utils import load_json, load_jsonl, save_json, save_jsonl


def load_data(filepath):
    """自动判断文件类型并加载数据"""
    if filepath.endswith('.jsonl'):
        return load_jsonl(filepath)
    elif filepath.endswith('.json'):
        return load_json(filepath)
    else:
        print(f"⚠️ 不支持的文件格式: {filepath}")
        return []


def analyze_scores_detailed(data, verbose=False):
    """详细分数分析"""
    scores = [x.get('score', 0) for x in data if 'score' in x]
    
    if not scores:
        print("⚠️ 没有找到分数数据")
        return
    
    print(f"\n=== 数据概览 ===")
    print(f"总样本数: {len(data)}")
    print(f"有效分数样本数: {len(scores)}")
    print(f"平均分: {sum(scores)/len(scores):.2f}")
    print(f"最高分: {max(scores):.2f}")
    print(f"最低分: {min(scores):.2f}")
    
    if verbose:
        print(f"\n=== 分数分布 ===")
        score_counter = Counter(scores)
        for score in sorted(score_counter.keys(), reverse=True):
            print(f"分数 {score}: {score_counter[score]} 个样本")


def build_ref_map(ref_data, id_key='id', round_key='round'):
    """
    构建参考文件索引: (id, round) -> {messages, output}
    
    Args:
        ref_data: 参考数据列表
        id_key: ID 字段名
        round_key: 轮次字段名
    
    Returns:
        ref_map: 索引字典
    """
    print(f"正在构建参考文件索引 (源文件: {len(ref_data)} 条)...")
    ref_map = {}
    duplicates = 0
    
    for item in ref_data:
        key = (item.get(id_key), item.get(round_key))
        if key in ref_map:
            duplicates += 1
            continue
        
        ref_map[key] = {
            "messages": item.get('messages', []),
            "output": item.get('output', '')
        }
    
    if duplicates > 0:
        print(f"  警告: 发现 {duplicates} 个重复的，已跳过")
    print(f"  索引构建完成: {len(ref_map)} 个唯一键\n")
    return ref_map


def get_prompt_from_messages(messages):
    """
    从 messages 中提取 prompt。
    逻辑：提取 system 和 user 的内容
    """
    if not messages:
        return '', ''
    
    instruction = ''
    input_ = ''
    for item in messages:
        if item.get('role') == 'system':
            instruction = item.get('content', '')
        if item.get('role') == 'user':
            input_ = item.get('content', '')

    return instruction, input_


def build_dpo_dataset(score_data, ref_map, min_margin, min_chosen_score, 
                      id_key='id', round_key='round', verbose=False):
    """
    构建完整的 DPO 数据集 (包含文本内容)
    
    Args:
        score_data: 带分数的数据列表
        ref_map: 参考数据索引
        min_margin: 最小分差
        min_chosen_score: 正样本最低分
        id_key: ID 字段名
        round_key: 轮次字段名
        verbose: 是否显示详细信息
    
    Returns:
        dpo_list: DPO 数据列表
        filtered_list: 被过滤的样本列表
    """
    print(f"开始构建 DPO 数据 (Margin > {min_margin}, Chosen >= {min_chosen_score})...")
    
    # 1. 按 ID 分组
    groups = defaultdict(list)
    for item in score_data:
        id_val = item.get(id_key)
        if id_val is not None:
            groups[id_val].append(item)
    
    dpo_list = []
    filtered_list = []
    
    stats = {
        "total_groups": len(groups),
        "valid_pairs": 0,
        "filtered_small_margin": 0,
        "filtered_bad_chosen": 0,
        "filtered_missing_ref": 0,
        "filtered_single_item": 0
    }
    
    for id_val, items in groups.items():
        if len(items) < 2:
            stats["filtered_single_item"] += 1
            if verbose:
                filtered_list.append({
                    "id": id_val,
                    "reason": "Only one sample in group",
                    "count": len(items)
                })
            continue
        
        # 按分数排序
        items_sorted = sorted(items, key=lambda x: x.get('score', 0), reverse=True)
        
        chosen_meta = items_sorted[0]
        rejected_meta = items_sorted[-1]
        
        chosen_score = chosen_meta.get('score', 0)
        rejected_score = rejected_meta.get('score', 0)
        score_diff = chosen_score - rejected_score
        
        # --- 过滤逻辑 ---
        reason = None
        
        # 1. 检查分差
        if score_diff < min_margin:
            stats["filtered_small_margin"] += 1
            reason = f"Margin too small ({score_diff} < {min_margin})"
        
        # 2. 检查正样本质量
        elif chosen_score < min_chosen_score:
            stats["filtered_bad_chosen"] += 1
            reason = f"Chosen score too low ({chosen_score} < {min_chosen_score})"
        
        # 3. 检查参考文件中是否存在对应的文本数据
        else:
            chosen_key = (id_val, chosen_meta.get(round_key))
            rejected_key = (id_val, rejected_meta.get(round_key))
            if chosen_key not in ref_map or rejected_key not in ref_map:
                stats["filtered_missing_ref"] += 1
                reason = "Missing reference text (output/messages)"
        
        if reason:
            # 记录被过滤的样本以便调试
            filtered_list.append({
                "id": id_val,
                "reason": reason,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "rounds": [chosen_meta.get(round_key), rejected_meta.get(round_key)]
            })
            continue

        # --- 构建有效 DPO 对 ---
        chosen_ref = ref_map[chosen_key]
        rejected_ref = ref_map[rejected_key]
        
        instruction, input_ = get_prompt_from_messages(chosen_ref['messages'])
        
        dpo_item = {
            "instruction": instruction, 
            "input": input_,
            "chosen": chosen_ref['output'],
            "rejected": rejected_ref['output'],
            "id": id_val,
            "chosen_round": chosen_meta.get(round_key),
            "rejected_round": rejected_meta.get(round_key),
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "margin": score_diff
        }
        dpo_list.append(dpo_item)
        stats["valid_pairs"] += 1

    # 打印统计
    print(f"\n=== 构建结果 ===")
    print(f"总 ID 组数: {stats['total_groups']}")
    print(f"✅ 有效 DPO 对: {stats['valid_pairs']}")
    print(f"❌ 过滤样本总数: {len(filtered_list)}")
    print(f"   - 单样本组: {stats['filtered_single_item']}")
    print(f"   - 分差不足: {stats['filtered_small_margin']}")
    print(f"   - 正样本分低: {stats['filtered_bad_chosen']}")
    print(f"   - 缺失文本数据: {stats['filtered_missing_ref']}")
    
    return dpo_list, filtered_list


def save_data(filepath, data):
    """根据文件扩展名自动选择保存格式"""
    if filepath.endswith('.jsonl'):
        save_jsonl(data, filepath)
    elif filepath.endswith('.json'):
        save_json(data, filepath)
    else:
        print(f"⚠️ 不支持的输出格式: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="构建 DPO (Direct Preference Optimization) 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python build_dpo.py score_data.jsonl ref_data.jsonl -o dpo_output.jsonl
  
  # 自定义阈值
  python build_dpo.py score_data.jsonl ref_data.jsonl -o dpo_output.jsonl --min-margin 15 --min-chosen-score 70
  
  # 保存过滤日志
  python build_dpo.py score_data.jsonl ref_data.jsonl -o dpo_output.jsonl --save-filtered filtered_log.jsonl
  
  # 使用 JSON 格式输入
  python build_dpo.py score_data.json ref_data.json -o dpo_output.json
  
  # 显示详细统计信息
  python build_dpo.py score_data.jsonl ref_data.jsonl -o dpo_output.jsonl --verbose

数据格式说明:
  分数文件格式:
    {
      "id": "sample_001",
      "round": 1,
      "score": 85.5
    }
  
  参考文件格式:
    {
      "id": "sample_001",
      "round": 1,
      "messages": [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "用户输入"}
      ],
      "output": "模型输出"
    }
  
  输出文件格式:
    {
      "instruction": "系统提示",
      "input": "用户输入",
      "chosen": "优选输出",
      "rejected": "拒绝输出",
      "id": "sample_001",
      "chosen_round": 1,
      "rejected_round": 2,
      "chosen_score": 85.5,
      "rejected_score": 45.2,
      "margin": 40.3
    }
        """
    )
    
    parser.add_argument(
        'score_file',
        help='分数文件路径（JSON 或 JSONL 格式）'
    )
    
    parser.add_argument(
        'ref_file',
        help='参考文件路径（JSON 或 JSONL 格式，包含 messages 和 output）'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='输出文件路径（根据扩展名自动选择 JSON 或 JSONL 格式）'
    )
    
    parser.add_argument(
        '--min-margin',
        type=float,
        default=20.0,
        help='最小分差阈值（默认: 20.0）'
    )
    
    parser.add_argument(
        '--min-chosen-score',
        type=float,
        default=60.0,
        help='正样本最低分数阈值（默认: 60.0）'
    )
    
    parser.add_argument(
        '--save-filtered',
        type=str,
        metavar='FILE',
        help='保存被过滤的样本日志到指定文件'
    )
    
    parser.add_argument(
        '--id-key',
        type=str,
        default='id',
        help='ID 字段名（默认: id）'
    )
    
    parser.add_argument(
        '--round-key',
        type=str,
        default='round',
        help='轮次字段名（默认: round）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细统计信息'
    )
    
    args = parser.parse_args()
    
    # 1. 加载带分数的数据
    print("Step 1: 加载分数数据...")
    score_data = load_data(args.score_file)
    if not score_data:
        print("❌ 错误: 分数数据为空")
        return
        
    analyze_scores_detailed(score_data, verbose=args.verbose)
    
    # 2. 加载参考文本数据 (用于获取 output 和 messages)
    print("\nStep 2: 加载参考文本数据...")
    ref_data = load_data(args.ref_file)
    if not ref_data:
        print("❌ 错误: 参考文本数据为空")
        return
        
    ref_map = build_ref_map(ref_data, id_key=args.id_key, round_key=args.round_key)
    
    # 3. 构建 DPO 数据
    print("\nStep 3: 构建 DPO 数据集...")
    dpo_list, filtered_list = build_dpo_dataset(
        score_data, ref_map, 
        min_margin=args.min_margin, 
        min_chosen_score=args.min_chosen_score,
        id_key=args.id_key, 
        round_key=args.round_key,
        verbose=args.verbose
    )
    
    # 4. 保存结果
    print("\nStep 4: 保存结果...")
    
    if dpo_list:
        save_data(args.output, dpo_list)
    else:
        print("⚠️ 没有生成有效的 DPO 数据对")
    
    if args.save_filtered and filtered_list:
        save_data(args.save_filtered, filtered_list)
        print(f"\n📝 被过滤的样本日志已保存至: {args.save_filtered}")
        print("   你可以查看此文件调整阈值")

    print("\n🎉 处理完成!")


if __name__ == "__main__":
    main()
