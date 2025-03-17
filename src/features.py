import re

class FeatureExtractor:
    def __init__(self):
        # 可能需要设置特征窗口大小等参数
        self.window_size = 2
        
    def extract_features(self, sentence, index):
        """从句子中提取特征，针对指定索引位置
        
        Args:
            sentence: 输入的中文字符串
            index: 当前考虑的字符位置
            
        Returns:
            特征字典
        """
        features = {}
        
        # 单字特征
        for i in range(-self.window_size, self.window_size + 1):
            if 0 <= index + i < len(sentence):
                features[f'C{i}'] = sentence[index + i]
                
        # 双字特征
        for i in range(-self.window_size, self.window_size):
            if 0 <= index + i < len(sentence) - 1:
                features[f'B{i}'] = sentence[index + i:index + i + 2]
                
        # 可以添加更多特征，如论文中描述的
        # ...
        
        return features