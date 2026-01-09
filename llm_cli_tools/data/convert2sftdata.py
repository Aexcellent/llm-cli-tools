import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any


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


def process_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单条数据：转换messages，格式化output
    
    Args:
        item: 原始数据项
    
    Returns:
        处理后的数据项
    """
    new_item = {}
    
    # 1. 保留除了messages之外的所有字段，并特殊处理output
    for key, value in item.items():
        if key != 'messages':
            if key == 'output':
                # 无论 input 是 dict 还是 str，都尝试将其标准化为紧凑的 JSON 字符串
                if isinstance(value, dict):
                    # 是字典 -> 直接转字符串
                    new_item[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                elif isinstance(value, str):
                    # 是字符串 -> 尝试解析为字典后再转（标准化单引号等非标准格式）
                    try:
                        dict_value = json.loads(value)
                        new_item[key] = json.dumps(dict_value, ensure_ascii=False, separators=(",", ":"))
                    except (json.JSONDecodeError, TypeError):
                        # 解析失败说明就是普通文本 -> 直接使用
                        new_item[key] = value
                else:
                    # 其他类型（列表、数字等） -> 转为 JSON 字符串
                    new_item[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            else:
                # 其他字段直接保留
                new_item[key] = value

    # 2. 处理messages字段，提取为instruction和input
    if 'messages' in item:
        messages = item['messages']
        instruction = ""
        input_text = ""
        
        # 提取 system 和 user 的内容
        # 注意：这里以最后一次出现的为准，如果需要合并请修改逻辑
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content', '')
            if role == 'system':
                instruction = content
            elif role == 'user':
                input_text = content
        
        new_item['instruction'] = instruction
        new_item['input'] = input_text
    
    return new_item


def process_jsonl_file(input_file: str, output_file: str, verbose: bool = False) -> int:
    """
    处理JSONL文件（流式读取，节省内存）
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        verbose: 是否显示详细信息
    
    Returns:
        处理的数据条数
    """
    processed_count = 0
    error_count = 0
    
    if verbose:
        print(f"正在处理 (JSONL): {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    new_item = process_single_item(item)
                    
                    # 写入处理后的行
                    fout.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"跳过无效JSON行: {e}")
                    error_count += 1
                    continue
    except Exception as e:
        print(f"错误: 处理文件失败 {input_file}: {e}", file=sys.stderr)
        sys.exit(1)
    
    if verbose:
        print(f"完成！共处理 {processed_count} 条数据，跳过 {error_count} 条无效数据")
        print(f"输出文件: {output_file}")
    
    return processed_count


def process_json_file(input_file: str, output_file: str, verbose: bool = False) -> int:
    """
    处理JSON文件（列表格式）
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        verbose: 是否显示详细信息
    
    Returns:
        处理的数据条数
    """
    error_count = 0
    
    if verbose:
        print(f"正在处理 (JSON): {input_file}")
    
    try:
        # 读取原始JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理每条数据
        processed_data = []
        for item in data:
            try:
                new_item = process_single_item(item)
                processed_data.append(new_item)
            except Exception as e:
                if verbose:
                    print(f"跳过无效数据项: {e}")
                error_count += 1
                continue
        
        # 写入处理后的JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"错误: 处理文件失败 {input_file}: {e}", file=sys.stderr)
        sys.exit(1)
    
    if verbose:
        print(f"完成！共处理 {len(processed_data)} 条数据，跳过 {error_count} 条无效数据")
        print(f"输出文件: {output_file}")
    
    return len(processed_data)


def smart_process(input_file: str, output_file: str, verbose: bool = False) -> int:
    """
    智能判断输入输出格式并处理
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        verbose: 是否显示详细信息
    
    Returns:
        处理的数据条数
    """
    # 判断输出文件格式
    ext = Path(output_file).suffix.lower()
    
    if ext == '.json':
        return process_json_file(input_file, output_file, verbose)
    else:
        # 默认为 JSONL 处理
        return process_jsonl_file(input_file, output_file, verbose)


def main():
    parser = argparse.ArgumentParser(
        description="将包含 messages 字段的数据转换为 SFT（Supervised Fine-Tuning）格式，支持 JSON 和 JSONL 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 转换单个 JSONL 文件
  python convert2sftdata.py input.jsonl -o output.jsonl
  
  # 转换单个 JSON 文件
  python convert2sftdata.py input.json -o output.json
  
  # 转换多个文件
  python convert2sftdata.py file1.jsonl file2.jsonl -o output1.jsonl -o output2.jsonl
  
  # 转换文件并显示详细统计
  python convert2sftdata.py input.jsonl -o output.jsonl --verbose
  
  # 转换 JSONL 为 JSON 格式
  python convert2sftdata.py input.jsonl -o output.json

数据格式说明:
  输入格式:
    {
      "messages": [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "用户输入"}
      ],
      "output": {"key": "value"},
      "other_field": "other_value"
    }
  
  输出格式:
    {
      "instruction": "系统提示",
      "input": "用户输入",
      "output": "{\"key\":\"value\"}",
      "other_field": "other_value"
    }
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='输入文件路径（支持多个文件，可以是 JSON 或 JSONL 格式）'
    )
    
    parser.add_argument(
        '-o', '--output',
        action='append',
        required=True,
        help='输出文件路径（根据扩展名自动选择 JSON 或 JSONL 格式，可多次使用以指定多个输出文件）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细处理信息'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件数量和输出文件数量是否匹配
    if len(args.input_files) != len(args.output):
        print(f"错误: 输入文件数量 ({len(args.input_files)}) 和输出文件数量 ({len(args.output)}) 不匹配", file=sys.stderr)
        sys.exit(1)
    
    # 检查输入文件是否存在
    for filepath in args.input_files:
        if not Path(filepath).exists():
            print(f"错误: 文件不存在: {filepath}", file=sys.stderr)
            sys.exit(1)
    
    # 处理所有文件
    total_processed = 0
    file_stats = []
    
    for input_file, output_file in zip(args.input_files, args.output):
        try:
            input_type = detect_file_type(input_file)
            output_ext = Path(output_file).suffix.lower()
            
            if args.verbose:
                print(f"\n处理文件 {len(file_stats) + 1}/{len(args.input_files)}")
                print(f"输入: {input_file} (类型: {input_type})")
                print(f"输出: {output_file} (格式: {output_ext})")
            
            processed_count = smart_process(input_file, output_file, args.verbose)
            
            file_stats.append({
                'input_file': input_file,
                'output_file': output_file,
                'input_type': input_type,
                'output_format': output_ext,
                'count': processed_count
            })
            
            total_processed += processed_count
            
        except Exception as e:
            print(f"错误: 处理失败 {input_file}: {e}", file=sys.stderr)
            sys.exit(1)
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("📊 转换统计")
    print("=" * 60)
    
    for i, stat in enumerate(file_stats, 1):
        print(f"\n文件 {i}:")
        print(f"  输入: {stat['input_file']} ({stat['input_type']})")
        print(f"  输出: {stat['output_file']} ({stat['output_format']})")
        print(f"  条数: {stat['count']}")
    
    print(f"\n总计处理: {total_processed} 条数据")
    print("=" * 60)


if __name__ == "__main__":
    main()
