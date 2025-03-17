import argparse
import pickle
import os
import sys

# 添加项目根目录到路径中，以便能够导入src模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import Perceptron
from src.features import FeatureExtractor
from src.decoder import Decoder

def read_test_data(filename):
    """读取测试数据
    
    Returns:
        sentences: 原始句子列表
        gold_segmentations: 对应的正确分词列表
    """
    sentences = []
    gold_segmentations = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # 获取分词结果
                words = line.split()
                if not words:  # 跳过空行
                    continue
                    
                gold_segmentations.append(words)
                
                # 连接成原始句子
                sentence = ''.join(words)
                sentences.append(sentence)
        
        return sentences, gold_segmentations
    except Exception as e:
        print(f"读取测试数据时出错: {e}")
        try:  # 尝试其他编码
            with open(filename, 'r', encoding='gbk') as f:
                print("尝试使用GBK编码读取文件...")
                # 处理逻辑相同，此处省略
        except:
            print("读取文件失败，请检查文件路径和编码")
        return [], []

def evaluate(args):
    """评估模型性能"""
    # 检查文件路径
    if not os.path.exists(args.model_file):
        print(f"错误: 模型文件不存在: {args.model_file}")
        return
        
    if not os.path.exists(args.test_file):
        print(f"错误: 测试文件不存在: {args.test_file}")
        return
    
    # 加载模型
    print(f"正在加载模型: {args.model_file}")
    try:
        with open(args.model_file, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 初始化特征提取器和解码器
    feature_extractor = FeatureExtractor()
    decoder = Decoder(model)
    
    # 读取测试数据
    print(f"正在读取测试数据: {args.test_file}")
    sentences, gold_segmentations = read_test_data(args.test_file)
    
    if not sentences:
        print("未能读取有效测试数据")
        return
    
    # 评估模型性能
    print("正在评估模型性能...")
    total_correct = 0
    total_predicted = 0
    total_gold = 0
    
    for i, (sentence, gold_words) in enumerate(zip(sentences, gold_segmentations)):
        try:
            # 使用模型预测分词
            predicted_words = decoder.decode(sentence, feature_extractor)
            
            # 计算正确预测的词数
            correct = len(set(get_word_spans(predicted_words, sentence)) & 
                          set(get_word_spans(gold_words, sentence)))
            
            # 更新统计信息
            total_correct += correct
            total_predicted += len(predicted_words)
            total_gold += len(gold_words)
            
            if i % 100 == 0 and i > 0:
                print(f"已处理 {i}/{len(sentences)} 个句子...")
                # 打印中间结果
                if total_predicted > 0 and total_gold > 0:
                    p = total_correct / total_predicted
                    r = total_correct / total_gold
                    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
                    print(f"  当前F1: {f1:.4f} (P={p:.4f}, R={r:.4f})")
        except Exception as e:
            print(f"处理句子 {i} 时出错: {e}")
            continue
    
    # 计算评估指标
    precision = total_correct / total_predicted if total_predicted > 0 else 0
    recall = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n最终评估结果:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"总句数: {len(sentences)}")
    print(f"总词数: 预测={total_predicted}, 标准={total_gold}")

def get_word_spans(words, sentence):
    """获取词在句子中的位置，用于准确比较分词结果"""
    spans = []
    start = 0
    for word in words:
        if not word:  # 忽略空词
            continue
            
        if start + len(word) <= len(sentence):
            if sentence[start:start+len(word)] == word:
                spans.append((start, start + len(word)))
                start += len(word)
            else:
                # 如果遇到不匹配的情况，尝试在句子中找到这个词
                pos = sentence[start:].find(word)
                if pos >= 0:
                    start = start + pos
                    spans.append((start, start + len(word)))
                    start += len(word)
                else:
                    # 如果找不到词，跳过并记录警告
                    # print(f"警告: 无法在句子中定位词: '{word}'")
                    pass
    return spans

def main():
    parser = argparse.ArgumentParser(description="评估中文分词模型")
    parser.add_argument("--test_file", default="data/sighan2005/msr_test_gold.utf8", help="测试文件路径")
    parser.add_argument("--model_file", default="models/default_model.pkl", help="模型文件路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细输出")
    
    args = parser.parse_args()
    
    # 如果没有指定模型文件或模型文件不存在，则提示用户
    if not os.path.exists(args.model_file):
        print(f"错误: 模型文件不存在: {args.model_file}")
        print("\n使用示例:")
        print("  python tests/test_features.py --model_file models/my_model.pkl")
        print("  python tests/test_features.py --model_file models/my_model.pkl --test_file data/sighan2005/msr_test_gold.utf8")
        return
    
    evaluate(args)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
        print("\n使用示例:")
        print("  python tests/test_features.py --model_file models/my_model.pkl")
        print("  python tests/test_features.py --model_file models/my_model.pkl --test_file data/sighan2005/msr_test_gold.utf8")