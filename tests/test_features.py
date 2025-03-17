import argparse
import pickle
import os
import sys
import glob

# 添加项目根目录到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import Perceptron
from src.features import FeatureExtractor
from src.decoder import Decoder

def read_gold_segmentations(filename):
    """读取标准分词结果文件"""
    segmentations = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            segmentations.append(words)
    return segmentations

def find_model_files():
    """查找所有可能的模型文件"""
    # 首先查找当前目录下的models文件夹
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    if not os.path.exists(model_dir):
        model_dir = "models"  # 尝试相对路径
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)  # 如果不存在，创建该目录
            print(f"创建了模型目录: {model_dir}")
            return []

    # 查找所有.pkl文件
    model_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    return model_files

def find_test_files():
    """查找所有可能的测试文件"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    # 查找processed目录
    processed_dir = os.path.join(data_dir, "processed")
    if os.path.exists(processed_dir):
        # 查找包含test的文件
        test_files = glob.glob(os.path.join(processed_dir, "*test*.utf8"))
        if test_files:
            return test_files
    
    # 查找sighan2005目录
    sighan_dir = os.path.join(data_dir, "sighan2005")
    if os.path.exists(sighan_dir):
        test_files = glob.glob(os.path.join(sighan_dir, "*test*.utf8"))
        if test_files:
            return test_files
    
    # 如果都找不到，在data目录本身查找
    test_files = glob.glob(os.path.join(data_dir, "*test*.utf8"))
    return test_files

def find_dict_files():
    """查找所有可能的词典文件"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    dict_files = glob.glob(os.path.join(data_dir, "**", "*vocab*.utf8"), recursive=True)
    dict_files.extend(glob.glob(os.path.join(data_dir, "**", "*dict*.txt"), recursive=True))
    return dict_files

def evaluate(model_file, test_file, dict_file=None, beam_size=1, window_size=3):
    """评估模型在测试集上的性能"""
    # 加载模型
    print(f"加载模型: {model_file}")
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return 0, 0, 0
    
    # 加载词典（如果有）
    dictionary = set()
    if dict_file and os.path.exists(dict_file):
        print(f"加载词典: {dict_file}")
        with open(dict_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()[0]
                if word and len(word) > 1:
                    dictionary.add(word)
        print(f"加载了 {len(dictionary)} 个词条")
    
    # 初始化特征提取器和解码器
    feature_extractor = FeatureExtractor(window_size=window_size, dictionary=dictionary)
    decoder = Decoder(model, beam_size=beam_size)
    
    # 读取测试数据
    print(f"读取测试文件: {test_file}")
    test_sentences = []
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                test_sentences.append(''.join(line.split()))
    except Exception as e:
        print(f"读取测试文件失败: {e}")
        return 0, 0, 0
    
    # 读取黄金标准分词结果
    gold_file = test_file.replace('_test.utf8', '_test_gold.utf8')
    if not os.path.exists(gold_file):
        gold_file = test_file.replace('_input.utf8', '_gold.utf8')
    
    if not os.path.exists(gold_file):
        print(f"找不到黄金标准文件，请提供正确的标准分词文件路径")
        gold_file = input("请输入标准分词文件路径（留空跳过评估，仅进行分词）: ")
        if not gold_file:
            # 仅进行分词，不评估
            print("\n仅进行分词演示：")
            for i, sentence in enumerate(test_sentences[:5]):
                predicted_words = decoder.decode(sentence, feature_extractor)
                print(f"\n原句 {i+1}: {sentence}")
                print(f"分词结果: {' '.join(predicted_words)}")
            return 0, 0, 0
    
    try:
        gold_segmentations = read_gold_segmentations(gold_file)
    except Exception as e:
        print(f"读取标准分词文件失败: {e}")
        return 0, 0, 0
    
    # 确保测试句子和黄金分词数量一致
    if len(test_sentences) != len(gold_segmentations):
        print(f"警告：测试句子数量 ({len(test_sentences)}) 与标准分词数量 ({len(gold_segmentations)}) 不匹配")
        # 取最小值
        min_len = min(len(test_sentences), len(gold_segmentations))
        test_sentences = test_sentences[:min_len]
        gold_segmentations = gold_segmentations[:min_len]
    
    # 计算F1分数
    total_correct = 0
    total_predicted = 0
    total_gold = 0
    
    print(f"\n评估中，共 {len(test_sentences)} 个句子...")
    for i, (sentence, gold_words) in enumerate(zip(test_sentences, gold_segmentations)):
        # 模型预测
        predicted_words = decoder.decode(sentence, feature_extractor)
        
        # 计算指标
        gold_set = set(''.join(gold_words[:i]) for i in range(1, len(gold_words) + 1))
        pred_set = set(''.join(predicted_words[:i]) for i in range(1, len(predicted_words) + 1))
        
        # 更新统计
        total_correct += len(gold_set & pred_set)
        total_predicted += len(pred_set)
        total_gold += len(gold_set)
        
        # 输出一些样例
        if i < 3 or i % 100 == 0:
            print(f"\n[{i+1}/{len(test_sentences)}] 原句: {sentence[:50]}..." if len(sentence) > 50 else f"\n[{i+1}/{len(test_sentences)}] 原句: {sentence}")
            print(f"预测: {' '.join(predicted_words[:15])}..." if len(predicted_words) > 15 else f"预测: {' '.join(predicted_words)}")
            print(f"标准: {' '.join(gold_words[:15])}..." if len(gold_words) > 15 else f"标准: {' '.join(gold_words)}")
    
    # 计算P/R/F1
    p = total_correct / total_predicted if total_predicted > 0 else 0
    r = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    print(f"\n评估结果:")
    print(f"精确率: {p:.4f}")
    print(f"召回率: {r:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    return f1, p, r

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估中文分词模型")
    parser.add_argument("--model_file", help="模型文件路径")
    parser.add_argument("--test_file", help="测试文件路径")
    parser.add_argument("--dict_file", default="", help="外部词典文件路径")
    parser.add_argument("--beam_size", type=int, default=1, help="beam search宽度")
    parser.add_argument("--window_size", type=int, default=3, help="特征窗口大小")
    
    try:
        args = parser.parse_args()
        
        # 如果未提供模型文件路径，进入交互模式
        if not args.model_file:
            model_files = find_model_files()
            
            if not model_files:
                print("错误: 未找到任何模型文件。请先训练模型或指定模型文件路径。")
                sys.exit(1)
            
            print("\n找到以下模型文件:")
            for i, file in enumerate(model_files):
                print(f"{i+1}. {file}")
            
            choice = input("\n请选择模型文件编号: ")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(model_files):
                    args.model_file = model_files[idx]
                else:
                    print("无效选择，退出")
                    sys.exit(1)
            except ValueError:
                print("请输入有效数字")
                sys.exit(1)
        
        # 如果未提供测试文件路径，进入交互模式
        if not args.test_file:
            test_files = find_test_files()
            
            if not test_files:
                print("错误: 未找到任何测试文件。")
                args.test_file = input("请手动输入测试文件路径: ")
                if not args.test_file or not os.path.exists(args.test_file):
                    print("无效的文件路径，退出")
                    sys.exit(1)
            else:
                print("\n找到以下测试文件:")
                for i, file in enumerate(test_files):
                    print(f"{i+1}. {file}")
                
                choice = input("\n请选择测试文件编号: ")
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(test_files):
                        args.test_file = test_files[idx]
                    else:
                        print("无效选择，退出")
                        sys.exit(1)
                except ValueError:
                    print("请输入有效数字")
                    sys.exit(1)
        
        # 如果提供了--dict_file但路径为空，进入交互模式
        if args.dict_file == "":
            use_dict = input("\n是否使用外部词典? (y/n): ")
            if use_dict.lower() == 'y':
                dict_files = find_dict_files()
                
                if not dict_files:
                    print("未找到任何词典文件。")
                    args.dict_file = input("请手动输入词典文件路径 (或按Enter跳过): ")
                else:
                    print("\n找到以下词典文件:")
                    for i, file in enumerate(dict_files):
                        print(f"{i+1}. {file}")
                    
                    choice = input("\n请选择词典文件编号 (或按Enter跳过): ")
                    if choice:
                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(dict_files):
                                args.dict_file = dict_files[idx]
                            else:
                                print("无效选择，不使用词典")
                                args.dict_file = ""
                        except ValueError:
                            print("无效输入，不使用词典")
                            args.dict_file = ""
        
        # 询问beam_size
        adjust_beam = input(f"\n当前beam search宽度为 {args.beam_size}，是否调整? (y/n): ")
        if adjust_beam.lower() == 'y':
            beam_size = input("请输入新的beam search宽度: ")
            try:
                args.beam_size = int(beam_size)
            except ValueError:
                print(f"无效输入，使用默认值 {args.beam_size}")
        
        # 询问window_size
        adjust_window = input(f"\n当前特征窗口大小为 {args.window_size}，是否调整? (y/n): ")
        if adjust_window.lower() == 'y':
            window_size = input("请输入新的特征窗口大小: ")
            try:
                args.window_size = int(window_size)
            except ValueError:
                print(f"无效输入，使用默认值 {args.window_size}")
        
        # 打印评估配置
        print("\n评估配置:")
        print(f"模型文件: {args.model_file}")
        print(f"测试文件: {args.test_file}")
        print(f"词典文件: {args.dict_file if args.dict_file else '不使用'}")
        print(f"Beam search宽度: {args.beam_size}")
        print(f"特征窗口大小: {args.window_size}")
        
        # 执行评估
        evaluate(args.model_file, args.test_file, args.dict_file, args.beam_size, args.window_size)
        
    except SystemExit:
        print("\n使用说明:")
        print("  python test_features.py [--model_file MODEL_FILE] [--test_file TEST_FILE] [--dict_file DICT_FILE]")
        print("如果不提供参数，将进入交互式选择模式\n")
        
        # 直接调用main函数（重新执行一遍，进入交互模式）
        if __name__ == "__main__":
            import sys
            sys.argv = [sys.argv[0]]  # 清空命令行参数
            exec(open(__file__).read())  # 重新执行脚本