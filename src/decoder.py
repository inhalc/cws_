class Decoder:
    def __init__(self, model, beam_size=8):
        self.model = model
        self.beam_size = beam_size
        
    def decode(self, sentence, feature_extractor):
        """使用beam search解码进行分词"""
        if self.beam_size <= 1:
            return self._greedy_decode(sentence, feature_extractor)
        else:
            return self._beam_decode(sentence, feature_extractor)
    
    def _greedy_decode(self, sentence, feature_extractor):
        """简单贪心解码"""
        # 对每个位置计算分数
        scores = [0] * (len(sentence)-1)
        for i in range(len(sentence)-1):
            features = feature_extractor.extract_features(sentence, i)
            scores[i] = self.model.score(features)
        
        # 根据分数进行分词
        words = []
        start = 0
        for i, score in enumerate(scores):
            if score > 0:  # 正分数表示应该分词
                words.append(sentence[start:i+1])
                start = i+1
        
        # 添加最后一个词
        if start < len(sentence):
            words.append(sentence[start:])
        
        return words
    
    def _beam_decode(self, sentence, feature_extractor):
        """Beam search解码 - 论文算法核心"""
        beam = [{'words': [], 'score': 0.0, 'last': 0}]
        
        for i in range(len(sentence)):
            new_beam = []
            
            for state in beam:
                last = state['last']
                
                # 选项1: 在当前位置分词
                if i > last:
                    word = sentence[last:i]
                    
                    # 计算这个词的特征分数
                    word_score = 0
                    for j in range(last, i-1):
                        features = feature_extractor.extract_features(sentence, j)
                        # 不分词
                        word_score -= max(0, self.model.score(features))
                    
                    if i-1 >= last:
                        # 这里应该分词
                        features = feature_extractor.extract_features(sentence, i-1)
                        word_score += max(0, self.model.score(features))
                    
                    # 创建新状态
                    new_state = {
                        'words': state['words'] + [word],
                        'score': state['score'] + word_score,
                        'last': i
                    }
                    new_beam.append(new_state)
                
                # 选项2: 继续当前词
                if i < len(sentence) - 1:
                    features = feature_extractor.extract_features(sentence, i)
                    no_sep_score = -max(0, self.model.score(features))
                    
                    # 创建新状态
                    new_state = {
                        'words': state['words'].copy(),
                        'score': state['score'] + no_sep_score,
                        'last': state['last']
                    }
                    new_beam.append(new_state)
            
            # 保留前beam_size个最佳状态
            new_beam.sort(key=lambda x: x['score'], reverse=True)
            beam = new_beam[:self.beam_size]
        
        # 处理最后一个词
        best_state = beam[0]
        if best_state['last'] < len(sentence):
            best_state['words'].append(sentence[best_state['last']:])
        
        return best_state['words']