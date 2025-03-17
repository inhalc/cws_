import os
import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
from collections import Counter

def check_file_existence(file_path):
    """检查文件是否存在"""
    if not os.path.exists(file_path):
        return False, f"错误: 文件 '{file_path}' 不存在"
    return True, f"文件 '{file_path}' 存在"

def check_file_size(file_path, min_size=1):
    """检查文件大小是否合理"""
    size = os.path.getsize(file_path)
    if size < min_size:
        return False, f"警告: 文件大小为 {size} 字节，可能为空"
    return True, f"文件大小: {size} 字节"

def check_file(file_path, file_type=None):
    """检查文件的基本属性"""
    results = []
    
    # 检查文件是否存在
    exists, msg = check_file_existence(file_path)
    results.append(msg)
    if not exists:
        return results
    
    # 检查文件大小
    _, msg = check_file_size(file_path)
    results.append(msg)
    
    # 尝试读取文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = [f.readline() for _ in range(5)]
        
        # 统计行数
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
            
        results.append(f"文件行数: {line_count}")
        
        # 显示前几行内容
        results.append("文件前几行内容:")
        for i, line in enumerate(first_lines):
            if line.strip():  # 只显示非空行
                results.append(f"  行 {i+1}: {line.strip()}")
    
    except UnicodeDecodeError:
        results.append("警告: 文件编码不是UTF-8，尝试其他编码...")
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                first_lines = [f.readline() for _ in range(5)]
            results.append("成功以GBK编码读取文件")
            results.append("文件前几行内容:")
            for i, line in enumerate(first_lines):
                if line.strip():
                    results.append(f"  行 {i+1}: {line.strip()}")
        except:
            results.append("错误: 无法读取文件内容，请检查文件编码")
    except Exception as e:
        results.append(f"读取文件时出错: {str(e)}")
    
    return results

def find_potential_files():
    """查找当前目录及子目录中可能的文件"""
    potential_files = []
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".utf8") or file.endswith(".txt") or file.endswith(".json"):
                potential_files.append(os.path.join(root, file))
    return potential_files

def check_segmentation_file(file_path):
    """检查分词文件的特定特征"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        results = []
        line_count = len(lines)
        results.append(f"分词文件行数: {line_count}")
        
        # 统计基本信息
        total_words = 0
        total_chars = 0
        single_char_words = 0
        word_lengths = []
        
        for line in lines:
            words = line.strip().split()
            total_words += len(words)
            
            for word in words:
                total_chars += len(word)
                word_lengths.append(len(word))
                if len(word) == 1:
                    single_char_words += 1
        
        # 计算统计量
        avg_word_len = total_chars / total_words if total_words > 0 else 0
        single_word_ratio = single_char_words / total_words if total_words > 0 else 0
        
        results.append(f"总词数: {total_words}")
        results.append(f"总字符数: {total_chars}")
        results.append(f"平均词长: {avg_word_len:.2f}")
        results.append(f"单字词数: {single_char_words} ({single_word_ratio:.2%})")
        
        # 词长分布
        length_counter = Counter(word_lengths)
        results.append("词长分布:")
        for length, count in sorted(length_counter.items()):
            results.append(f"  {length}字词: {count} ({count/total_words:.2%})")
        
        # 检查特殊情况
        if single_word_ratio > 0.5:
            results.append("警告: 单字词比例过高 (>50%)，可能存在过度分词问题")
        if single_word_ratio < 0.1:
            results.append("警告: 单字词比例过低 (<10%)，可能存在分词不足问题")
        
        # 检查前几行的内容
        results.append("\n文件前3行内容:")
        for i, line in enumerate(lines[:3]):
            results.append(f"  {i+1}: {line.strip()}")
        
        return True, results
    except Exception as e:
        return False, [f"检查分词文件时出错: {str(e)}"]

def check_labeled_file(file_path):
    """检查带标签的训练文件的特定特征"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        results = []
        line_count = len(lines)
        results.append(f"带标签文件行数: {line_count}")
        
        # 统计基本信息
        valid_lines = 0
        total_chars = 0
        total_labels = 0
        label_counts = {0: 0, 1: 0}
        
        for line in lines:
            line = line.strip()
            if not line or '\t' not in line:
                continue
                
            parts = line.split('\t')
            if len(parts) != 2:
                continue
                
            valid_lines += 1
            sentence, label_str = parts
            labels = [int(l) for l in label_str.split()]
            
            if len(sentence) != len(labels):
                results.append(f"警告: 句子长度与标签数量不匹配: {len(sentence)} vs {len(labels)}")
                continue
                
            total_chars += len(sentence)
            total_labels += sum(labels)
            
            for label in labels:
                if label in label_counts:
                    label_counts[label] += 1
                    
        # 计算统计量
        label_ratio = total_labels / total_chars if total_chars > 0 else 0
        
        results.append(f"有效行数: {valid_lines}")
        results.append(f"总字符数: {total_chars}")
        results.append(f"标签0(非分词点)数量: {label_counts.get(0, 0)}")
        results.append(f"标签1(分词点)数量: {label_counts.get(1, 0)}")
        results.append(f"分词点比例: {label_ratio:.4f}")
        
        # 检验合理性
        if label_ratio < 0.1:
            results.append("警告: 分词点比例过低 (<10%)，可能存在分词不足问题")
        if label_ratio > 0.5:
            results.append("警告: 分词点比例过高 (>50%)，可能存在过度分词问题")
        
        # 检查前几行的内容
        results.append("\n文件前2行内容:")
        for i, line in enumerate(lines[:2]):
            results.append(f"  {i+1}: {line.strip()}")
        
        return True, results
    except Exception as e:
        return False, [f"检查标签文件时出错: {str(e)}"]

def main():
    parser = argparse.ArgumentParser(description="检查预处理文件")
    parser.add_argument('--file_path', help="要检查的文件路径")
    parser.add_argument('--type', choices=['csv', 'json', 'text', 'segmentation', 'labeled'], help="文件类型")
    
    try:
        args = parser.parse_args()
        
        # 如果未提供文件路径，则交互式选择
        if not args.file_path:
            files = find_potential_files()
            if not files:
                print("错误: 未找到可用的文件。")
                return
            
            print("\n找到以下文件:")
            for i, file in enumerate(files):
                print(f"{i+1}. {file}")
            
            choice = input("\n请选择要检查的文件编号，或直接输入文件路径: ")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    args.file_path = files[idx]
                else:
                    print("无效选择，退出")
                    return
            except ValueError:
                if os.path.exists(choice):
                    args.file_path = choice
                else:
                    print(f"文件不存在: {choice}")
                    return
        
        # 猜测文件类型
        if not args.type:
            if args.file_path.endswith('_words.utf8') or args.file_path.endswith('_gold.utf8'):
                args.type = 'segmentation'
            elif args.file_path.endswith('_labeled.utf8'):
                args.type = 'labeled'
            else:
                file_types = ['text', 'segmentation', 'labeled', 'csv', 'json']
                print("\n请选择文件类型:")
                for i, t in enumerate(file_types):
                    print(f"{i+1}. {t}")
                
                type_choice = input("请输入类型编号: ")
                try:
                    type_idx = int(type_choice) - 1
                    if 0 <= type_idx < len(file_types):
                        args.type = file_types[type_idx]
                    else:
                        args.type = 'text'  # 默认为文本类型
                except ValueError:
                    args.type = 'text'  # 默认为文本类型
        
        # 执行检查
        if args.type == 'segmentation':
            results = check_file(args.file_path, 'text')
            _, seg_results = check_segmentation_file(args.file_path)
            results.extend(seg_results)
        elif args.type == 'labeled':
            results = check_file(args.file_path, 'text')
            _, label_results = check_labeled_file(args.file_path)
            results.extend(label_results)
        else:
            results = check_file(args.file_path, args.type)
        
        # 输出结果
        print("\n===== 文件检查报告 =====")
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"文件路径: {args.file_path}")
        print(f"文件类型: {args.type}")
        print("========================\n")
        
        for result in results:
            print(result)
            
    except SystemExit:
        print("\n使用方法:")
        print("  python check_preprocessed_file.py [--file_path 文件路径] [--type 文件类型]")
        print("\n如果不提供参数，将会提示您交互式选择文件\n")
        
        # 重新调用main，而不是递归调用
        main()

if __name__ == "__main__":
    main()