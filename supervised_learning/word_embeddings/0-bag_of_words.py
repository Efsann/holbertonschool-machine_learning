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
    cleaned_sentences = []
    for sentence in sentences:
        # Durğu işarələrini silirik ('s hissəsindəki apostrofu da silir)
        cleaned = re.sub(r'[^\w\s]', '', sentence.lower())
        # 's sözün sonundadırsa silirik (children's -> children)
        words = [re.sub(r"'s$", '', word) for word in cleaned.split()]
        cleaned_sentences.append(words)

    if vocab is None:
        words_set = set()
        for sentence_words in cleaned_sentences:
            words_set.update(sentence_words)
        features = sorted(list(words_set))
    else:
        features = vocab

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, sentence_words in enumerate(cleaned_sentences):
        for word in sentence_words:
            if word in features:
                j = features.index(word)
                embeddings[i, j] += 1

    return embeddings, features
