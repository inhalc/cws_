import argparse
import pickle
import os
import sys
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 添加项目根目录到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用绝对导入路径，与test_features.py保持一致
from src.model import Perceptron
from src.features import FeatureExtractor
from src.decoder import Decoder

def read_training_data(filename):
    """读取带标签的训练数据"""
    sentences = []
    labels = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否为带标签格式
            if '\t' in line:
                # 带标签格式："句子\t标签序列"
                parts = line.split('\t')
                if len(parts) == 2:
                    sentence, label_str = parts
                    labels.append([int(l) for l in label_str.split()])
                    sentences.append(sentence)
            else:
                # 旧格式处理，生成分词标签
                words = line.split()
                if not words:
                    continue
                    
                sentence = ''.join(words)
                sentences.append(sentence)
                
                # 生成分词标签
                label = [0] * len(sentence)
                pos = 0
                for word in words:
                    pos += len(word)
                    if pos < len(sentence):
                        label[pos-1] = 1  # 正确地标记词结束位置
                
                labels.append(label)
    
    return sentences, labels

def evaluate_f1(model, decoder, feature_extractor, sentences, gold_segmentations):
    """计算F1分数"""
    total_correct = 0
    total_predicted = 0
    total_gold = 0
    
    for sentence, gold_words in zip(sentences, gold_segmentations):
        # 模型预测
        predicted_words = decoder.decode(sentence, feature_extractor)
        
        # 计算F1
        gold_spans = set(get_word_spans(gold_words, sentence))
        pred_spans = set(get_word_spans(predicted_words, sentence))
        
        # 计数
        total_correct += len(gold_spans & pred_spans)
        total_predicted += len(pred_spans)
        total_gold += len(gold_spans)
    
    # 计算F1
    p = total_correct / total_predicted if total_predicted > 0 else 0
    r = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    return f1, p, r

def get_word_spans(words, sentence):
    """与测试函数相同的实现"""
    spans = []
    current_pos = 0
    
    for word in words:
        if not word:  # 跳过空词
            continue
            
        # 尝试在当前位置匹配
        if current_pos + len(word) <= len(sentence) and sentence[current_pos:current_pos+len(word)] == word:
            spans.append((current_pos, current_pos + len(word)))
            current_pos += len(word)
        else:
            # 如果无法在当前位置匹配，尝试在剩余文本中查找
            pos = sentence[current_pos:].find(word)
            if pos >= 0:
                current_pos = current_pos + pos
                spans.append((current_pos, current_pos + len(word)))
                current_pos += len(word)
    
    return spans

def extract_sentence_features(args):
    """为单个句子提取特征"""
    sentence, feature_extractor = args
    sentence_features = []
    for i in range(len(sentence)):
        features = feature_extractor.extract_features(sentence, i)
        sentence_features.append(features)
    return sentence_features

def analyze_segmentation(words):
    """分析分词结果，检测单字词比例"""
    single_char = sum(1 for w in words if len(w) == 1)
    total = len(words)
    return single_char / total if total > 0 else 0

def train(args):
    """训练模型"""
    print("开始训练模型...")
    
    # 加载外部词典(如果有指定)
    dictionary = set()
    if args.dict_file and os.path.exists(args.dict_file):
        print(f"加载外部词典: {args.dict_file}")
        with open(args.dict_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()[0] if args.dict_has_freq else line.strip()
                if word and len(word) > 1:  # 只添加多字词
                    dictionary.add(word)
        print(f"词典大小: {len(dictionary)} 个词")
    
    # 初始化模型和特征提取器
    model = Perceptron(learning_rate=args.learning_rate)
    feature_extractor = FeatureExtractor(
        window_size=args.window_size, 
        dictionary=dictionary
    )
    
    # 创建解码器，用于在验证集上评估
    decoder = Decoder(model, beam_size=args.beam_size) if args.beam_size > 1 else Decoder(model)
    
    # 读取训练数据
    print(f"从 {args.train_file} 读取训练数据...")
    sentences, gold_labels = read_training_data(args.train_file)
    print(f"读取了 {len(sentences)} 个训练样本")
    
    # 划分验证集
    if args.dev_ratio > 0:
        dev_size = int(len(sentences) * args.dev_ratio)
        indices = list(range(len(sentences)))
        random.shuffle(indices)
        train_indices = indices[dev_size:]
        dev_indices = indices[:dev_size]
        
        train_sentences = [sentences[i] for i in train_indices]
        train_labels = [gold_labels[i] for i in train_indices]
        dev_sentences = [sentences[i] for i in dev_indices]
        dev_labels = [gold_labels[i] for i in dev_indices]
        
        print(f"训练集大小: {len(train_sentences)}, 验证集大小: {len(dev_sentences)}")
    else:
        train_sentences = sentences
        train_labels = gold_labels
    
    # 替换多线程预计算特征部分
    print("预计算特征...")
    precomputed_features = []

    # 使用批处理而非全部放入内存
    batch_size = 1000  # 每批处理1000个句子
    total_batches = (len(train_sentences) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(total_batches), 
                         desc="特征预计算", 
                         ncols=80,  # 固定进度条宽度
                         position=0,  # 在第一行显示
                         mininterval=1.0):  # 最小更新间隔
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(train_sentences))
        
        batch_sentences = train_sentences[start_idx:end_idx]
        batch_features = []
        
        for sentence in tqdm(batch_sentences, 
                            desc=f"批次 {batch_idx+1}/{total_batches}", 
                            leave=False,  # 完成后删除进度条
                            position=1,  # 在第二行显示
                            ncols=80):  # 固定宽度
            sentence_features = []
            for i in range(len(sentence)):
                features = feature_extractor.extract_features(sentence, i)
                sentence_features.append(features)
            batch_features.append(sentence_features)
        
        precomputed_features.extend(batch_features)

    print(f"特征预计算完成，共 {len(precomputed_features)} 个句子的特征")
    
    # 初始化最佳模型记录
    best_acc = 0
    no_improve_epochs = 0
    
    single_char_ratios = []
    # 训练多轮
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        correct = 0
        total = 0
        
        # 打乱训练数据
        if args.shuffle:
            combined = list(zip(train_sentences, train_labels))
            random.shuffle(combined)
            train_sentences, train_labels = zip(*combined)
        
        for sentence, gold_label, sentence_features in zip(train_sentences, train_labels, precomputed_features):
            for i in range(len(sentence)):
                features = sentence_features[i]
                # 预测
                score = model.score(features)
                pred_label = 1 if score > 0 else 0
                
                # 更新
                if pred_label != gold_label[i]:
                    model.update(features, gold_label[i])
                else:
                    correct += 1
                total += 1
        
        train_acc = correct / total if total > 0 else 0
        print(f"训练准确率: {train_acc:.4f}")
        
        # 在验证集上评估
        if args.dev_ratio > 0:
            dev_correct = 0
            dev_total = 0
            
            for sentence, gold_label in zip(dev_sentences, dev_labels):
                for i in range(len(sentence)):
                    features = feature_extractor.extract_features(sentence, i)
                    score = model.score(features)
                    pred_label = 1 if score > 0 else 0
                    
                    if pred_label == gold_label[i]:
                        dev_correct += 1
                    dev_total += 1
            
            dev_acc = dev_correct / dev_total if dev_total > 0 else 0
            print(f"验证准确率: {dev_acc:.4f}")
            
            # 早停
            if args.early_stopping:
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    no_improve_epochs = 0
                    # 保存最佳模型
                    best_model_file = args.model_file.replace('.pkl', '_best.pkl')
                    os.makedirs(os.path.dirname(best_model_file), exist_ok=True)
                    with open(best_model_file, 'wb') as f:
                        pickle.dump(model, f)
                    print(f"保存最佳模型到: {best_model_file}")
                else:
                    no_improve_epochs += 1
                    print(f"验证准确率未提升, 已连续 {no_improve_epochs} 轮")
                    if no_improve_epochs >= args.patience:
                        print(f"早停! {args.patience} 轮未见改善")
                        break
        
        # 在每个epoch结束后添加F1评估
        if epoch % 3 == 0 or epoch == args.epochs - 1:  # 每3个epoch或最后一个epoch评估F1
            if dev_sentences:
                # 计算验证集F1
                # 先生成正确的分词结果
                dev_segmentations = []
                for sentence, label in zip(dev_sentences[:100], dev_labels[:100]):
                    # 从标签序列生成分词列表
                    words = []
                    start = 0
                    for i, is_boundary in enumerate(label):
                        if is_boundary == 1 or i == len(label)-1:
                            if i == len(label)-1:
                                words.append(sentence[start:])
                            else:
                                words.append(sentence[start:i+1])
                            start = i + 1
                    dev_segmentations.append(words)
                
                # 现在使用生成的分词结果进行评估
                dev_f1, dev_p, dev_r = evaluate_f1(model, decoder, feature_extractor, 
                                                  dev_sentences[:100], dev_segmentations)
                print(f"验证集 F1: {dev_f1:.4f} (P={dev_p:.4f}, R={dev_r:.4f})")
        
        # 监控分词结果
        if epoch % 3 == 0 or epoch == args.epochs - 1:
            sample_sentences = dev_sentences[:5] if dev_sentences else train_sentences[:5]
            print("\n分词结果示例:")
            for sentence in sample_sentences:
                predicted_words = decoder.decode(sentence, feature_extractor)
                ratio = analyze_segmentation(predicted_words)
                single_char_ratios.append(ratio)
                print(f"原文: {sentence}")
                print(f"分词: {' '.join(predicted_words)}")
                print(f"单字词比例: {ratio:.2f}")
            
            avg_ratio = sum(single_char_ratios) / len(single_char_ratios)
            print(f"平均单字词比例: {avg_ratio:.2f}")
            
            # 如果单字词比例过高，自动调整解码器阈值
            if avg_ratio > 0.7:  # 超过70%是单字词
                decoder.threshold += 0.1  # 提高阈值
                print(f"单字词比例过高，提高解码阈值至: {decoder.threshold:.2f}")
    
    # 完成训练，计算平均权重
    model.finalize()
    
    # 保存模型
    if args.model_file:
        # 确保目录存在
        os.makedirs(os.path.dirname(args.model_file), exist_ok=True)
        
        with open(args.model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存到: {args.model_file}")
    
    print("训练完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练中文分词模型")
    parser.add_argument("--train_file", default="data/processed/msr_training_filtered_labeled.utf8", 
                       help="训练文件路径")
    parser.add_argument("--model_file", default="models/perceptron_cws.pkl", 
                       help="模型保存路径")
    parser.add_argument("--epochs", type=int, default=10, 
                       help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1.0, 
                       help="学习率")
    parser.add_argument("--beam_size", type=int, default=1, 
                       help="beam search宽度")
    parser.add_argument("--window_size", type=int, default=3, 
                       help="特征窗口大小")
    parser.add_argument("--dict_file", type=str, default="", 
                       help="外部词典文件路径")
    parser.add_argument("--dict_has_freq", action="store_true", 
                       help="词典文件是否包含词频信息")
    parser.add_argument("--dev_ratio", type=float, default=0.1, 
                       help="验证集比例")
    parser.add_argument("--early_stopping", action="store_true", 
                       help="是否使用早停")
    parser.add_argument("--patience", type=int, default=3, 
                       help="早停耐心值")
    parser.add_argument("--shuffle", action="store_true", 
                       help="是否打乱训练数据")
    args = parser.parse_args()
    
    train(args)
