import re

class FeatureExtractor:
    def __init__(self, window_size=3, dictionary=None, hash_size=1000000):
        self.window_size = window_size
        self.dictionary = dictionary or set()
        self.hash_size = hash_size
    
    def _hash_feature(self, feature):
        """对特征名进行哈希，减少内存占用"""
        return abs(hash(feature)) % self.hash_size
        
    def extract_features(self, sentence, index):
        """使用特征哈希的特征提取"""
        features = {}
        
        # 确保索引有效
        if index < 0 or index >= len(sentence):
            return features
        
        # 1. 单字特征 (更大窗口)
        for i in range(-self.window_size, self.window_size + 1):
            pos = index + i
            if 0 <= pos < len(sentence):
                features[self._hash_feature(f'C{i}')] = sentence[pos]
                
                # 字符类型特征
                char = sentence[pos]
                if '\u4e00' <= char <= '\u9fff':
                    features[self._hash_feature(f'TYPE{i}_HAN')] = 1
                elif '0' <= char <= '9':
                    features[self._hash_feature(f'TYPE{i}_DIGIT')] = 1
                elif ('a' <= char <= 'z') or ('A' <= char <= 'Z'):
                    features[self._hash_feature(f'TYPE{i}_LETTER')] = 1
                else:
                    features[self._hash_feature(f'TYPE{i}_PUNCT')] = 1
        
        # 2. 字符组合特征
        # 双字特征
        for i in range(-self.window_size+1, self.window_size):
            start = index + i
            end = start + 2
            if 0 <= start < end <= len(sentence):
                features[self._hash_feature(f'B{i}')] = sentence[start:end]
        
        # 三字特征
        for i in range(-self.window_size+1, self.window_size-1):
            start = index + i
            end = start + 3
            if 0 <= start < end <= len(sentence):
                features[self._hash_feature(f'T{i}')] = sentence[start:end]
        
        # 3. 词典特征 - 论文中的关键改进点
        if self.dictionary:
            # 当前位置开始的可能词
            for length in range(2, 11):  # 最大支持10字词
                if index + length <= len(sentence):
                    word = sentence[index:index+length]
                    if word in self.dictionary:
                        # 增加词典特征的权重，使分词更倾向于词典中的词
                        features[self._hash_feature(f'DICT_BEGIN_{length}')] = 3.0  # 增大权重
                        features[self._hash_feature('DICT_BEGIN')] = 3.0
            
            # 当前位置结束的可能词
            for length in range(2, 11):
                if index - length + 1 >= 0:
                    word = sentence[index-length+1:index+1]
                    if word in self.dictionary:
                        features[self._hash_feature(f'DICT_END_{length}')] = 1
                        features[self._hash_feature('DICT_END')] = 1
            
            # 当前位置是词中间
            for prev_len in range(1, 5):
                for next_len in range(1, 5):
                    if index - prev_len >= 0 and index + next_len < len(sentence):
                        word = sentence[index-prev_len:index+next_len+1]
                        if word in self.dictionary:
                            features[self._hash_feature(f'DICT_MID_{prev_len}_{next_len}')] = 1
                            features[self._hash_feature('DICT_MID')] = 1
        
        # 4. 位置特征
        if index == 0:
            features[self._hash_feature('BEGIN')] = 1
        if index == len(sentence) - 1:
            features[self._hash_feature('END')] = 1
            
        # 5. 字符转换特征 (捕获特定模式)
        if 0 < index < len(sentence)-1:
            c1 = self._get_char_type(sentence[index-1])
            c2 = self._get_char_type(sentence[index])
            c3 = self._get_char_type(sentence[index+1])
            features[self._hash_feature(f'TRANS_{c1}_{c2}_{c3}')] = 1
            features[self._hash_feature(f'TRANS_{c1}_{c2}')] = 1
            features[self._hash_feature(f'TRANS_{c2}_{c3}')] = 1
        
        return features
    
    def _get_char_type(self, char):
        """获取字符类型"""
        if '\u4e00' <= char <= '\u9fff':
            return 'C'  # 中文
        elif '0' <= char <= '9':
            return 'D'  # 数字
        elif ('a' <= char <= 'z') or ('A' <= char <= 'Z'):
            return 'L'  # 字母
        else:
            return 'P'  # 标点符号