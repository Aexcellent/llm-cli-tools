import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Set


def detect_file_type(filepath: str) -> str:
    """
    根据文件扩展名和内容判断文件类型
    
    Args:
        filepath: 文件路径
    
    Returns:
        "json" 或 "jsonl"
    """
    ext = Path(filepath).suffix.lower()
    
    # 先根据扩展名判断
    if ext == '.json':
        return 'json'
    elif ext == '.jsonl':
        return 'jsonl'
    
    # 如果扩展名不明确，尝试根据内容判断
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line.startswith('['):
                return 'json'
            else:
                return 'jsonl'
    except Exception:
        return 'jsonl'


def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """
    加载 JSON 文件
    
    Args:
        filepath: JSON 文件路径
    
    Returns:
        数据列表
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        raise ValueError(f"不支持的 JSON 数据类型: {type(data)}")


def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """
    加载 JSONL 文件
    
    Args:
        filepath: JSONL 文件路径
    
    Returns:
        数据列表
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_file(filepath: str) -> List[Dict[str, Any]]:
    """
    自动判断文件类型并加载
    
    Args:
        filepath: 文件路径
    
    Returns:
        数据列表
    """
    file_type = detect_file_type(filepath)
    
    if file_type == 'json':
        return load_json_file(filepath)
    else:
        return load_jsonl_file(filepath)


def save_json_file(data: List[Dict[str, Any]], filepath: str):
    """
    保存为 JSON 文件
    
    Args:
        data: 数据列表
        filepath: 输出文件路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl_file(data: List[Dict[str, Any]], filepath: str):
    """
    保存为 JSONL 文件
    
    Args:
        data: 数据列表
        filepath: 输出文件路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_file(data: List[Dict[str, Any]], filepath: str):
    """
    根据文件扩展名自动选择格式保存
    
    Args:
        data: 数据列表
        filepath: 输出文件路径
    """
    ext = Path(filepath).suffix.lower()
    
    if ext == '.json':
        save_json_file(data, filepath)
        print(f"检测到 .json 扩展名，已保存为 JSON 列表格式。")
    else:
        save_jsonl_file(data, filepath)
        print(f"检测到非 .json 扩展名，已保存为 JSONL 行格式。")


def deduplicate_by_key(data: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """
    根据指定键去重
    
    Args:
        data: 数据列表
        key: 用于去重的键名
    
    Returns:
        去重后的数据列表
    """
    seen: Set[Any] = set()
    deduplicated = []
    
    for item in data:
        if key in item:
            value = item[key]
            if value not in seen:
                seen.add(value)
                deduplicated.append(item)
        else:
            deduplicated.append(item)
    
    return deduplicated


def deduplicate_by_content(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    根据内容去重（将字典转为字符串比较）
    
    Args:
        data: 数据列表
    
    Returns:
        去重后的数据列表
    """
    seen: Set[str] = set()
    deduplicated = []
    
    for item in data:
        item_str = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if item_str not in seen:
            seen.add(item_str)
            deduplicated.append(item)
    
    return deduplicated


def main():
    parser = argparse.ArgumentParser(
        description="合并多个 JSON/JSONL 文件，支持自动判断文件类型和输出格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 合并两个 JSONL 文件
  python merge_jsonl.py file1.jsonl file2.jsonl -o merged.jsonl
  
  # 合并不同类型的文件（JSON 和 JSONL）
  python merge_jsonl.py data.json results.jsonl -o merged.json
  
  # 合并多个文件并去重（根据 id 字段）
  python merge_jsonl.py file1.jsonl file2.jsonl file3.jsonl -o merged.jsonl --dedupe id
  
  # 合并文件并完全去重（根据内容）
  python merge_jsonl.py file1.jsonl file2.jsonl -o merged.jsonl --dedupe-all
  
  # 合并文件并保留统计信息
  python merge_jsonl.py file1.jsonl file2.jsonl -o merged.jsonl --verbose
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='输入文件路径（支持多个文件，可以是 JSON 或 JSONL 格式）'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='输出文件路径（根据扩展名自动选择 JSON 或 JSONL 格式）'
    )
    
    parser.add_argument(
        '--dedupe',
        type=str,
        metavar='KEY',
        help='根据指定键去重（例如：--dedupe id）'
    )
    
    parser.add_argument(
        '--dedupe-all',
        action='store_true',
        help='根据内容完全去重'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细统计信息'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    for filepath in args.input_files:
        if not Path(filepath).exists():
            print(f"错误: 文件不存在: {filepath}", file=sys.stderr)
            sys.exit(1)
    
    # 加载所有文件
    merged_data = []
    file_stats = []
    
    for filepath in args.input_files:
        try:
            file_type = detect_file_type(filepath)
            data = load_file(filepath)
            
            if args.verbose:
                print(f"加载文件: {filepath} (类型: {file_type}, 条数: {len(data)})")
            
            merged_data.extend(data)
            file_stats.append({
                'filepath': filepath,
                'type': file_type,
                'count': len(data)
            })
        except Exception as e:
            print(f"错误: 加载文件失败 {filepath}: {e}", file=sys.stderr)
            sys.exit(1)
    
    original_count = len(merged_data)
    
    # 去重
    if args.dedupe:
        merged_data = deduplicate_by_key(merged_data, args.dedupe)
        if args.verbose:
            print(f"根据键 '{args.dedupe}' 去重: {original_count} -> {len(merged_data)}")
    elif args.dedupe_all:
        merged_data = deduplicate_by_content(merged_data)
        if args.verbose:
            print(f"根据内容完全去重: {original_count} -> {len(merged_data)}")
    
    # 保存合并后的数据
    try:
        save_file(merged_data, args.output)
    except Exception as e:
        print(f"错误: 保存文件失败 {args.output}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("📊 合并统计")
    print("=" * 60)
    
    for i, stat in enumerate(file_stats, 1):
        print(f"文件 {i}: {stat['filepath']}")
        print(f"  类型: {stat['type']}")
        print(f"  条数: {stat['count']}")
    
    print(f"\n合并前总条数: {original_count}")
    print(f"合并后总条数: {len(merged_data)}")
    
    if args.dedupe or args.dedupe_all:
        print(f"去重后减少: {original_count - len(merged_data)} 条")
    
    print(f"\n已写入: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
