def extract_word_features(segmented_sentence):
    """提取词级别全局特征"""
    features = []
    for i in range(len(segmented_sentence)):
        word = segmented_sentence[i]
        # 当前词特征
        features.append(f"WORD_{word}")
        features.append(f"WORD_LEN_{len(word)}")
        # 上下文特征
        if i > 0:
            features.append(f"BIGRAM_{segmented_sentence[i-1]}_{word}")
        if i < len(segmented_sentence)-1:
            features.append(f"BIGRAM_{word}_{segmented_sentence[i+1]}")
    return features