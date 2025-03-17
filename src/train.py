import argparse
import pickle
import os
import sys

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
    
    # 初始化模型和特征提取器
    model = Perceptron()
    feature_extractor = FeatureExtractor()
    
    # 读取训练数据
    print(f"从 {args.train_file} 读取训练数据...")
    sentences, gold_labels = read_training_data(args.train_file)
    print(f"读取了 {len(sentences)} 个训练样本")
    
    # 训练多轮
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        correct = 0
        total = 0
        
        for sentence, gold_label in zip(sentences, gold_labels):
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
        
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.4f}")
    
    # 完成训练
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
    parser.add_argument("--train_file", default="data/sighan2005/msr_training_words.utf8", help="训练文件路径")
    parser.add_argument("--model_file", default="models/perceptron_cws.pkl", help="模型保存路径")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    args = parser.parse_args()
    
    train(args)
