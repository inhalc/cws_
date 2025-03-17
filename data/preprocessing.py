import os
import re
import argparse
import random
from collections import Counter

def ensure_dir(dir_path):
    """确保目录存在"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def process_standard_segmentation_file(input_file, output_dir):
    """处理标准分词文件（已经分好词的文本，词之间用空格分隔）"""
    print(f"处理标准分词文件: {input_file}")
    
    # 获取文件基础名称
    basename = os.path.basename(input_file)
    name_part = os.path.splitext(basename)[0]
    
    # 定义输出文件路径
    words_output = os.path.join(output_dir, f"{name_part}_words.utf8")
    labeled_output = os.path.join(output_dir, f"{name_part}_labeled.utf8")
    
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # 读取和预处理输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # 尝试不同编码
        try:
            with open(input_file, 'r', encoding='gbk') as f:
                lines = f.readlines()
            print(f"注意: 使用GBK编码读取文件 {input_file}")
        except UnicodeDecodeError:
            print(f"错误: 无法读取文件 {input_file}，请检查文件编码")
            return None, None
    
    # 处理数据
    processed_words = []
    labeled_sentences = []
    total_words = 0
    total_chars = 0
    single_char_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 规范化空格
        line = re.sub(r'\s+', ' ', line)
        words = line.split()
        
        if not words:
            continue
            
        # 保存分词后的行
        processed_words.append(' '.join(words))
        
        # 生成无空格的句子
        sentence = ''.join(words)
        
        # 生成标签序列 (0/1标注，1表示词结束位置)
        labels = [0] * len(sentence)
        pos = 0
        for word in words:
            pos += len(word)
            if pos < len(sentence):
                labels[pos-1] = 1
                
            # 统计信息
            total_words += 1
            total_chars += len(word)
            if len(word) == 1:
                single_char_count += 1
        
        # 保存带标签的行
        labeled_sentences.append(f"{sentence}\t{' '.join(map(str, labels))}")
    
    # 写入处理后的分词文件
    with open(words_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_words))
    
    # 写入带标签的文件
    with open(labeled_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labeled_sentences))
    
    # 输出统计信息
    print(f"处理完成: {input_file}")
    print(f"  - 句子数: {len(processed_words)}")
    print(f"  - 总词数: {total_words}")
    print(f"  - 总字符数: {total_chars}")
    print(f"  - 平均词长: {total_chars/total_words:.2f}")
    print(f"  - 单字词比例: {single_char_count/total_words:.2f} ({single_char_count}/{total_words})")
    print(f"生成文件:")
    print(f"  - 分词文件: {words_output}")
    print(f"  - 标签文件: {labeled_output}")
    
    return words_output, labeled_output

def split_data(input_file, train_out, dev_out, dev_ratio=0.1):
    """将数据分割为训练集和开发集"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 随机打乱
    random.seed(42)  # 固定随机种子以便复现
    random.shuffle(lines)
    
    dev_size = int(len(lines) * dev_ratio)
    train_lines = lines[dev_size:]
    dev_lines = lines[:dev_size]
    
    # 写入训练集
    with open(train_out, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # 写入开发集
    with open(dev_out, 'w', encoding='utf-8') as f:
        f.writelines(dev_lines)
    
    print(f"数据分割: {input_file}")
    print(f"  - 训练集: {len(train_lines)} 行 -> {train_out}")
    print(f"  - 开发集: {len(dev_lines)} 行 -> {dev_out}")

def create_vocab(input_file, output_file, min_freq=2):
    """从分词文件创建词表"""
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
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in vocab:
            f.write(f"{word}\t{count}\n")
    
    # 写入字符表
    char_vocab_file = output_file.replace('vocab', 'char_vocab')
    char_vocab = sorted(char_counter.items(), key=lambda x: x[1], reverse=True)
    with open(char_vocab_file, 'w', encoding='utf-8') as f:
        for char, count in char_vocab:
            f.write(f"{char}\t{count}\n")
    
    print(f"词表生成: {output_file} ({len(vocab)} 词)")
    print(f"字表生成: {char_vocab_file} ({len(char_vocab)} 字)")

def create_test_file(input_file, output_dir):
    """
    处理测试文件，生成两个文件:
    1. 带空格的gold标准分词
    2. 不带空格的测试输入
    """
    basename = os.path.basename(input_file)
    name_part = os.path.splitext(basename)[0]
    
    # 定义输出文件路径
    gold_output = os.path.join(output_dir, f"{name_part}_gold.utf8")
    input_output = os.path.join(output_dir, f"{name_part}_input.utf8")
    
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(input_file, 'r', encoding='gbk') as f:
                lines = f.readlines()
            print(f"注意: 使用GBK编码读取文件 {input_file}")
        except:
            print(f"错误: 无法读取文件 {input_file}")
            return None, None
    
    # 处理数据
    gold_lines = []
    input_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 规范化空格
        line = re.sub(r'\s+', ' ', line)
        words = line.split()
        
        if not words:
            continue
            
        # 保存gold标准分词
        gold_lines.append(' '.join(words))
        
        # 保存无空格输入
        input_lines.append(''.join(words))
    
    # 写入gold标准文件
    with open(gold_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(gold_lines))
    
    # 写入测试输入文件
    with open(input_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(input_lines))
    
    print(f"测试文件处理: {input_file}")
    print(f"  - 标准分词: {gold_output}")
    print(f"  - 测试输入: {input_output}")
    
    return gold_output, input_output

# 修改 main 函数部分

def main():
    parser = argparse.ArgumentParser(description="处理中文分词数据")
    parser.add_argument("--train", help="训练文件路径 (分词格式)")
    parser.add_argument("--test", help="测试文件路径 (分词格式)")
    parser.add_argument("--output_dir", default="./processed", help="输出目录")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="开发集比例")
    
    # 解析参数
    try:
        args = parser.parse_args()
    except SystemExit:
        # 如果参数解析失败，提供交互式选择
        print("\n需要提供训练数据文件路径。您可以:")
        print("1. 使用命令行参数: python preprocessing.py --train 文件路径")
        print("2. 现在输入文件路径")
        
        train_path = input("\n请输入训练文件路径 (或按Enter取消): ")
        if not train_path:
            print("已取消。")
            return  # 退出函数
            
        class Args:
            pass
        args = Args()
        args.train = train_path
        args.test = input("请输入测试文件路径 (可选，按Enter跳过): ") or None
        args.output_dir = input("请输入输出目录 [./processed]: ") or "./processed"
        
        try:
            dev_ratio = input("请输入开发集比例 [0.1]: ") or "0.1"
            args.dev_ratio = float(dev_ratio)
        except ValueError:
            print("无效的比例值，使用默认值 0.1")
            args.dev_ratio = 0.1
    
    # 确保args.train有值
    if not args.train:
        print("错误: 未提供训练文件路径。")
        return  # 退出函数
    
    # 检查文件是否存在
    if not os.path.exists(args.train):
        print(f"错误: 训练文件不存在 '{args.train}'")
        
        # 尝试在当前目录和子目录中查找文件
        found_files = []
        for root, _, files in os.walk(".", topdown=True, followlinks=False):
            for file in files:
                if file.endswith(".utf8") or file.endswith(".txt"):
                    found_files.append(os.path.join(root, file))
        
        if found_files:
            print("\n找到以下可能的训练文件:")
            for i, file in enumerate(found_files):
                print(f"{i+1}. {file}")
            
            choice = input("\n请选择文件编号，或按Enter取消: ")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(found_files):
                    args.train = found_files[idx]
                    print(f"已选择: {args.train}")
                else:
                    print("无效选择，退出")
                    return  # 退出函数
            except (ValueError, IndexError):
                print("无效选择，退出")
                return  # 退出函数
        else:
            print("未找到可能的训练文件。")
            return  # 退出函数
    
    # 确保输出目录存在
    ensure_dir(args.output_dir)
    
    # 现在我们可以确保args.train是有效的文件路径
    train_words, train_labeled = process_standard_segmentation_file(args.train, args.output_dir)
    
    if train_words and train_labeled:
        # 分割训练/开发集
        train_base = os.path.splitext(train_words)[0]
        dev_words = train_base.replace("_words", "_dev_words") + ".utf8"
        train_filtered = train_base.replace("_words", "_filtered_words") + ".utf8"
        
        split_data(train_words, train_filtered, dev_words, args.dev_ratio)
        
        # 同样分割标签文件
        dev_labeled = train_base.replace("_words", "_dev_labeled") + ".utf8"
        train_filtered_labeled = train_base.replace("_words", "_filtered_labeled") + ".utf8"
        
        split_data(train_labeled, train_filtered_labeled, dev_labeled, args.dev_ratio)
        
        # 创建词表
        vocab_file = train_base.replace("_words", "_vocab") + ".utf8"
        create_vocab(train_filtered, vocab_file)
        
        # 处理测试文件
        if args.test:
            if os.path.exists(args.test):
                create_test_file(args.test, args.output_dir)
            else:
                print(f"警告: 测试文件不存在 '{args.test}'，跳过处理")
        
        print("\n处理完成!")
        print("可以使用以下命令训练模型:")
        print(f"python src/train.py --train_file {train_filtered_labeled}")
    else:
        print("处理训练文件失败，请检查文件格式是否正确。")

if __name__ == "__main__":
    main()