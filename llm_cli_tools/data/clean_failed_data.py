import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from llm_cli_tools.utils.file_utils import load_json_or_jsonl, save_json, save_jsonl


def get_file_type(filepath: str) -> str:
    """
    根据文件扩展名判断文件类型
    
    Args:
        filepath: 文件路径
    
    Returns:
        "json" 或 "jsonl"
    """
    ext = Path(filepath).suffix.lower()
    if ext == '.json':
        return 'json'
    else:
        return 'jsonl'


def clean_failed_data(data: List[Dict[str, Any]], check_fields: List[str], verbose: bool = False) -> Tuple[List[Dict[str, Any]], int]:
    """
    清理失败的数据
    
    Args:
        data: 原始数据列表
        check_fields: 要检查的字段列表
        verbose: 是否显示详细信息
    
    Returns:
        (清理后的数据列表, 删除的条数)
    """
    cleaned_data = []
    removed_count = 0
    
    for item in data:
        if not isinstance(item, dict):
            continue
        
        should_remove = False
        remove_reasons = []
        
        for field in check_fields:
            value = item.get(field)
            
            if value is None:
                should_remove = True
                remove_reasons.append(f'{field} 为 None')
            elif value is False:
                should_remove = True
                remove_reasons.append(f'{field} 为 False')
            elif value == 'null':
                should_remove = True
                remove_reasons.append(f'{field} 为 "null"')
        
        if should_remove:
            removed_count += 1
            if verbose:
                item_id = item.get('id', item.get('trace_id','N/A'))
                reason = ', '.join(remove_reasons)
                print(f"  删除: ID={item_id}, 原因: {reason}")
        else:
            cleaned_data.append(item)
    
    return cleaned_data, removed_count


def main():
    parser = argparse.ArgumentParser(
        description='清理 JSON 或 JSONL 文件中请求大模型失败的数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 清理 JSONL 文件（默认检查 output 字段）
  llm-clean input.jsonl --output-path cleaned.jsonl
  
  # 清理 JSON 文件
  llm-clean input.json --output-path cleaned.json
  
  # 检查多个字段
  llm-clean input.jsonl --check-fields output,score --output-path cleaned.jsonl
  
  # 覆盖原文件（会自动备份）
  llm-clean input.jsonl --overwrite
  
  # 显示详细信息
  llm-clean input.jsonl --output-path cleaned.jsonl --verbose
        '''
    )
    
    parser.add_argument(
        'input_file',
        help='输入文件路径（JSON 或 JSONL 格式）'
    )
    
    parser.add_argument(
        '--check-fields',
        type=str,
        default='output',
        help='要检查的字段列表（逗号分隔），默认为 output'
    )
    
    parser.add_argument(
        '--output-path',
        help='输出文件路径（不指定则使用原文件名添加 _cleaned 后缀）'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='覆盖原文件（会自动创建备份）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细信息'
    )
    
    args = parser.parse_args()
    
    check_fields = [field.strip() for field in args.check_fields.split(',')]
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"错误：文件不存在: {input_path}")
        return 1
    
    print(f"正在读取文件: {input_path}")
    
    try:
        data = load_json_or_jsonl(input_path)
        file_type = get_file_type(str(input_path))
        print(f"文件类型: {file_type.upper()}")
        print(f"原始数据条数: {len(data)}")
        print(f"检查字段: {', '.join(check_fields)}")
    except Exception as e:
        print(f"错误：读取文件失败: {e}")
        return 1
    
    print("\n正在清理失败数据...")
    cleaned_data, removed_count = clean_failed_data(data, check_fields, verbose=args.verbose)
    
    print(f"\n清理完成:")
    print(f"  删除条数: {removed_count}")
    print(f"  保留条数: {len(cleaned_data)}")
    
    if len(cleaned_data) == len(data):
        print("\n没有需要删除的数据，文件未修改")
        return 0
    
    if args.overwrite:
        output_path = input_path
        backup_path = str(input_path) + ".bak"
        shutil.copy(input_path, backup_path)
        print(f"\n已备份原文件到: {backup_path}")
    else:
        if args.output_path:
            output_path = Path(args.output_path)
        else:
            output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    
    try:
        if file_type == 'json':
            save_json(cleaned_data, output_path)
        else:
            save_jsonl(cleaned_data, output_path)
        print(f"\n已保存清理后的数据到: {output_path}")
    except Exception as e:
        print(f"错误：保存文件失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
