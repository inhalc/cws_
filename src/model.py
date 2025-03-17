import numpy as np
from collections import defaultdict

class AveragedPerceptron:
    def __init__(self):
        self.weights = defaultdict(float)
        self.total_weights = defaultdict(float)
        self.steps = defaultdict(int)
        self.n_samples = 0

    def update(self, truth_features, pred_features):
        # 标准感知机更新
        for f in truth_features:
            self.weights[f] += 1.0
            self.total_weights[f] += self.n_samples
        for f in pred_features:
            self.weights[f] -= 1.0
            self.total_weights[f] -= self.n_samples
        self.n_samples += 1

    def get_averaged_weights(self):
        # 计算平均权重
        avg = defaultdict(float)
        for f, total in self.total_weights.items():
            avg[f] = total / self.n_samples
        return avg

class Perceptron:
    def __init__(self, learning_rate=1.0):
        self.weights = {}
        self.total_weights = {}
        self.timestamps = {}
        self.current_time = 0
        self.learning_rate = learning_rate
        
    def score(self, features):
        """计算特征的分数"""
        score = 0.0
        for feat in features:
            if feat in self.weights:
                score += self.weights[feat]
        return score
    
    def update(self, features, label):
        """更新模型参数"""
        self.current_time += 1
        
        for feat in features:
            if feat not in self.weights:
                self.weights[feat] = 0.0
                self.total_weights[feat] = 0.0
                self.timestamps[feat] = self.current_time
            else:
                # 更新平均权重
                self.total_weights[feat] += (self.current_time - self.timestamps[feat]) * self.weights[feat]
                self.timestamps[feat] = self.current_time
            
            # 根据标签更新权重
            update = self.learning_rate if label == 1 else -self.learning_rate
            self.weights[feat] += update
    
    def finalize(self):
        """完成训练，计算平均权重"""
        for feat in self.weights:
            # 更新最后一次累积
            self.total_weights[feat] += (self.current_time - self.timestamps[feat]) * self.weights[feat]
            # 计算平均权重
            if self.current_time > 0:
                self.weights[feat] = self.total_weights[feat] / self.current_time