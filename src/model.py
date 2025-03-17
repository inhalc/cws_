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