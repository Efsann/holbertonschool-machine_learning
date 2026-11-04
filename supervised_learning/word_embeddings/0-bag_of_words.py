#!/usr/bin/env python3
"""
Bag of Words embedding matrix module
"""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    parameters:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use for analysis

    returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: list of features used for embeddings
    """
    # 1. Cümlələri kiçik hərflərə çeviririk və yalnız hərf/rəqəm sözlərini çıxarırıq
    cleaned_sentences = []
    for sentence in sentences:
        # \w+ regex-i durğu işarələrini və apostrofları avtomatik ayırır
        words = re.findall(r'\w+', sentence.lower())
        cleaned_sentences.append(words)

    # 2. Lüğəti (features) müəyyən edirik
    if vocab is None:
        words_set = set()
        for words in cleaned_sentences:
            words_set.update(words)
        features = sorted(list(words_set))
    else:
        features = vocab

    # 3. Matrisi sıfırlarla doldurub hazırlayırıq
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # 4. Hər cümlədəki sözlərin sayını hesablayırıq
    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in features:
                j = features.index(word)
                embeddings[i, j] += 1

    return embeddings, features
