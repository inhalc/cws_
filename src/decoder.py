class Decoder:
    def __init__(self, model):
        self.model = model
        
    def decode(self, sentence, feature_extractor):
        """使用当前模型解码句子（进行分词）
        
        Args:
            sentence: 输入的中文字符串
            feature_extractor: 特征提取器实例
            
        Returns:
            分词列表
        """
        scores = []
        for i in range(len(sentence)):
            features = feature_extractor.extract_features(sentence, i)
            score = self.model.score(features)
            scores.append(score)
            
        # 根据得分进行分词决策
        words = []
        start = 0
        for i in range(1, len(sentence)):
            # 如果当前位置适合切分（分数高于阈值或局部最大）
            if self._should_segment(scores, i):
                words.append(sentence[start:i])
                start = i
        
        # 添加最后一个词
        if start < len(sentence):
            words.append(sentence[start:])
            
        return words
    
    def _should_segment(self, scores, pos):
        """判断该位置是否应该进行分词"""
        # 可以使用简单阈值或更复杂的逻辑
        # 这里需要根据论文具体实现
        return scores[pos] > 0