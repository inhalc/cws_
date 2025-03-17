import os
import re
import random
import argparse
from collections import Counter

def ensure_dir(dir_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def convert_encoding(input_file, output_file, input_encoding='gbk', output_encoding='utf-8'):
    """转换文件编码"""
    try:
        with open(input_file, 'r', encoding=input_encoding) as f:
            content = f.read()
        
        with open(output_file, 'w', encoding=output_encoding) as f:
            f.write(content)
        
        print(f"成功转换编码: {input_file} → {output_file}")
        return True
    except Exception as e:
        print(f"转换编码失败: {e}")
        return False

def process_training_file(input_file, output_file):
    """处理训练文件，生成词级别的训练数据"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 清理数据
        line = re.sub(r'\s+', ' ', line)  # 规范化空白字符
        words = line.split()
        processed_lines.append(' '.join(words))
    
    # 写入处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_lines))
    
    print(f"已处理训练数据: {input_file} → {output_file}")
    print(f"处理后的行数: {len(processed_lines)}")
    
    # 统计基本信息
    total_words = sum(len(line.split()) for line in processed_lines)
    total_chars = sum(len(''.join(line.split())) for line in processed_lines)
    print(f"总词数: {total_words}")
    print(f"总字符数: {total_chars}")
    print(f"平均词长: {total_chars / total_words:.2f}")

def process_test_file(input_file, output_file):
    """处理测试文件，生成无空格的测试数据和对应的金标准分词结果"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 清理数据
        line = re.sub(r'\s+', ' ', line)  # 规范化空白字符
        words = line.split()
        processed_lines.append(' '.join(words))
    
    # 写入处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_lines))
    
    # 生成无空格的测试输入文件
    no_space_output = output_file.replace('_gold', '_input')
    with open(no_space_output, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(''.join(line.split()) + '\n')
    
    print(f"已处理测试数据: {input_file} → {output_file}")
    print(f"已生成无空格测试输入: {no_space_output}")
    print(f"处理后的行数: {len(processed_lines)}")

def split_train_dev(input_file, train_output, dev_output, dev_ratio=0.1):
    """将训练数据分割为训练集和开发集"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 随机打乱数据
    random.shuffle(lines)
    
    dev_size = int(len(lines) * dev_ratio)
    train_lines = lines[dev_size:]
    dev_lines = lines[:dev_size]
    
    # 写入训练集
    with open(train_output, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # 写入开发集
    with open(dev_output, 'w', encoding='utf-8') as f:
        f.writelines(dev_lines)
    
    print(f"数据集分割完成:")
    print(f"  - 训练集: {len(train_lines)} 行")
    print(f"  - 开发集: {len(dev_lines)} 行")

def generate_vocabulary(input_file, vocab_file, min_freq=2):
    """根据训练数据生成词表"""
    word_counter = Counter()
    char_counter = Counter()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                word_counter[word] += 1
                for char in word:
                    char_counter[char] += 1
    
    # 过滤低频词
    vocab = [(w, c) for w, c in word_counter.items() if c >= min_freq]
    vocab.sort(key=lambda x: x[1], reverse=True)
    
    # 写入词表
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for word, count in vocab:
            f.write(f"{word}\t{count}\n")
    
    # 写入字符表
    char_vocab_file = vocab_file.replace('vocab', 'char_vocab')
    char_vocab = sorted(char_counter.items(), key=lambda x: x[1], reverse=True)
    with open(char_vocab_file, 'w', encoding='utf-8') as f:
        for char, count in char_vocab:
            f.write(f"{char}\t{count}\n")
    
    print(f"词表生成完成: {vocab_file} ({len(vocab)} 个词)")
    print(f"字符表生成完成: {char_vocab_file} ({len(char_vocab)} 个字符)")

def process_sighan2005(data_dir, output_dir):
    """处理SIGHAN2005数据集"""
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # MSR数据集处理
    msr_train_src = os.path.join(data_dir, 'training', 'msr_training.utf8')
    msr_test_src = os.path.join(data_dir, 'gold', 'msr_test_gold.utf8')
    
    msr_train_out = os.path.join(output_dir, 'msr_training_words.utf8')
    msr_dev_out = os.path.join(output_dir, 'msr_dev_words.utf8')
    msr_train_filt_out = os.path.join(output_dir, 'msr_training_filtered_words.utf8')
    msr_test_out = os.path.join(output_dir, 'msr_test_gold.utf8')
    msr_vocab_out = os.path.join(output_dir, 'msr_vocab.utf8')
    
    # 处理训练数据
    if os.path.exists(msr_train_src):
        process_training_file(msr_train_src, msr_train_out)
        split_train_dev(msr_train_out, msr_train_filt_out, msr_dev_out)
        generate_vocabulary(msr_train_filt_out, msr_vocab_out)
    else:
        print(f"警告: 未找到MSR训练数据 {msr_train_src}")
    
    # 处理测试数据
    if os.path.exists(msr_test_src):
        process_test_file(msr_test_src, msr_test_out)
    else:
        print(f"警告: 未找到MSR测试数据 {msr_test_src}")
    
    # 处理PKU数据集（如果存在）
    pku_train_src = os.path.join(data_dir, 'training', 'pku_training.utf8')
    pku_test_src = os.path.join(data_dir, 'gold', 'pku_test_gold.utf8')
    
    if os.path.exists(pku_train_src) and os.path.exists(pku_test_src):
        pku_train_out = os.path.join(output_dir, 'pku_training_words.utf8')
        pku_dev_out = os.path.join(output_dir, 'pku_dev_words.utf8')
        pku_train_filt_out = os.path.join(output_dir, 'pku_training_filtered_words.utf8')
        pku_test_out = os.path.join(output_dir, 'pku_test_gold.utf8')
        pku_vocab_out = os.path.join(output_dir, 'pku_vocab.utf8')
        
        process_training_file(pku_train_src, pku_train_out)
        split_train_dev(pku_train_out, pku_train_filt_out, pku_dev_out)
        generate_vocabulary(pku_train_filt_out, pku_vocab_out)
        process_test_file(pku_test_src, pku_test_out)
    
    print("SIGHAN2005数据处理完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理SIGHAN2005中文分词数据集")
    parser.add_argument("--data_dir", default="../data/sighan2005", 
                        help="原始SIGHAN2005数据目录")
    parser.add_argument("--output_dir", default="../data/sighan2005", 
                        help="处理后数据的输出目录")
    args = parser.parse_args()
    
    process_sighan2005(args.data_dir, args.output_dir)