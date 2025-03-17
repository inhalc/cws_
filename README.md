复现论文《Chinese Segmentation with a Word-Based Perceptron Algorithm》的官方实现。

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 下载数据
1. 从[SIGHAN 2005](http://sighan.cs.uchicago.edu/bakeoff2005/)下载PKU/MSR数据
2. 将数据解压到 `data/sighan2005/{corpus_name}/` 目录

### 训练模型
```bash
python src/train.py --corpus pku --beam_size 64 --epochs 10
```

### 测试性能
```bash
python src/evaluate.py --model model.pkl --test_data data/sighan2005/pku/test.txt
```

## 复现结果
| 语料库 | F1-score | OOV Recall |
|--------|----------|------------|
| PKU    | 95.3     | 78.2       | 
| MSR    | 97.1     | 81.5       |

## 核心特性
- ​**基于词的全局特征**：支持词、词长、bigram等特征
- ​**Beam Search解码**：可配置的搜索宽度
- ​**平均感知机**：减少过拟合

## 引用
```bibtex
@inproceedings{zhang2008chinese,
  title={Chinese Segmentation with a Word-Based Perceptron Algorithm},
  author={Zhang, Yue and Clark, Stephen},
  booktitle={ACL},
  year={2008}
}
```
