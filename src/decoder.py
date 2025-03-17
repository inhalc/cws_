class Decoder:
    def __init__(self, model, beam_size=4):
        self.model = model
        self.beam_size = beam_size
        self.threshold = 0.8  # 默认阈值设为0，可根据需要调整
        
    def decode(self, sentence, feature_extractor):
        """分词解码，根据beam_size选择解码策略"""
        if self.beam_size <= 1:
            return self._greedy_decode(sentence, feature_extractor)
        else:
            return self._beam_search(sentence, feature_extractor)
    
    def _greedy_decode(self, sentence, feature_extractor):
        """使用贪心解码策略"""
        scores = []
        for i in range(len(sentence)):
            features = feature_extractor.extract_features(sentence, i)
            score = self.model.score(features)
            scores.append(score)
        
        # 使用阈值确定分词点
        words = []
        start = 0
        for i in range(len(sentence)):
            # 当score > threshold时被视为分词点(标签1)
            if i < len(sentence)-1 and scores[i] > self.threshold:  
                words.append(sentence[start:i+1])
                start = i+1
        
        # 添加最后一个词
        if start < len(sentence):
            words.append(sentence[start:])
            
        return words
    
    def _beam_search(self, sentence, feature_extractor):
        """Beam Search解码（论文核心算法）"""
        # 初始化beam
        beam = [{'words': [], 'score': 0.0, 'last_pos': 0}]
        
        # 对每个位置进行解码
        for i in range(len(sentence)):
            new_beam = []
            
            for item in beam:
                last_pos = item['last_pos']
                
                # 选项1: 在当前位置形成词
                if i >= last_pos:  # 有足够的字符形成词
                    word = sentence[last_pos:i+1]
                    
                    # 计算这个词的得分
                    word_score = 0.0
                    for j in range(last_pos, i):
                        # 非分词点应该得负分
                        features = feature_extractor.extract_features(sentence, j)
                        sep_score = self.model.score(features)
                        word_score -= max(0, sep_score)  # 惩罚内部分词点
                    
                    # 对每个位置分词的得分求和
                    features = feature_extractor.extract_features(sentence, i)
                    sep_score = self.model.score(features) 
                    # 末尾是词尾时应该得正分
                    word_score += max(0, sep_score) if i < len(sentence)-1 else 0
                    
                    # 添加新状态到beam
                    new_beam.append({
                        'words': item['words'] + [word],
                        'score': item['score'] + word_score,
                        'last_pos': i + 1
                    })
                
                # 选项2: 继续当前词
                if i < len(sentence) - 1:
                    features = feature_extractor.extract_features(sentence, i)
                    continue_score = -max(0, self.model.score(features))  # 不分词得分
                    
                    new_beam.append({
                        'words': item['words'].copy(),
                        'score': item['score'] + continue_score,
                        'last_pos': last_pos
                    })
            
            # 保留前beam_size个最佳结果
            new_beam.sort(key=lambda x: x['score'], reverse=True)
            beam = new_beam[:self.beam_size]
        
        # 处理最后一个词
        best_state = beam[0]
        if best_state['last_pos'] < len(sentence):
            best_state['words'].append(sentence[best_state['last_pos']:])
        
        # 修改beam search评分方法，添加对单字词的惩罚
        final_candidates = []
        for item in beam:
            # 惩罚单字词过多的候选
            single_char_penalty = sum(1 for w in item['words'] if len(w) == 1) * 0.5
            item['score'] -= single_char_penalty
            final_candidates.append(item)
        
        # 选择最高分候选
        final_candidates.sort(key=lambda x: x['score'], reverse=True)
        return final_candidates[0]['words']