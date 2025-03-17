import argparse
import pickle
import os
import sys
import random

# 添加项目根目录到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用绝对导入路径，与test_features.py保持一致
from src.model import Perceptron
from src.features import FeatureExtractor
from src.decoder import Decoder

def read_training_data(filename):
    """读取训练数据"""
    sentences = []
    labels = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # 获取词列表
            words = line.split()
            if not words:
                continue
                
            # 连接成原始句子
            sentence = ''.join(words)
            sentences.append(sentence)
            
            # 生成分词标签 (1表示分词点，0表示非分词点)
            label = [0] * len(sentence)
            pos = 0
            for word in words[:-1]:  # 最后一个词后面不是分词点
                pos += len(word)
                if pos < len(sentence):
                    label[pos] = 1
            labels.append(label)
    
    return sentences, labels

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
    
    # 初始化最佳模型记录
    best_acc = 0
    no_improve_epochs = 0
    
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
        
        for sentence, gold_label in zip(train_sentences, train_labels):
            for i in range(len(sentence)):
                features = feature_extractor.extract_features(sentence, i)
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
    parser.add_argument("--train_file", default="data/sighan2005/msr_training_words.utf8", 
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
