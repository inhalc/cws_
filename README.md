复现论文《Chinese Segmentation with a Word-Based Perceptron Algorithm》的官方实现。

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 下载数据
1. 从[SIGHAN 2005](http://sighan.cs.uchicago.edu/bakeoff2005/)下载PKU/MSR数据
2. 将数据解压到 `data/sighan2005/{corpus_name}/` 目录

## 项目提供了Makefile来简化数据处理、训练和测试流程：
 安装依赖
make install

 下载SIGHAN2005数据集
make download-data

 预处理数据
make preprocess

 训练所有模型(MSR和PKU)
make train

 仅训练MSR模型
make train-msr

 仅训练PKU模型
make train-pku

 测试所有模型
make test

 查看所有可用命令
make help

 执行完整流程(准备、训练、测试)
make all

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
