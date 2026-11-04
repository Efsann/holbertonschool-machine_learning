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
        features: numpy.ndarray of the features used for embeddings
    """
    cleaned_sentences = []
    for sentence in sentences:
        # Cümləni kiçik hərflərə çeviririk və durğu işarələrini təmizləyirik
        sent = sentence.lower()
        sent = sent.replace("'s", "")
        sent = re.sub(r'[^a-z0-9\s]', '', sent)
        cleaned_sentences.append(sent.split())

    if vocab is None:
        vocab_set = set()
        for words in cleaned_sentences:
            vocab_set.update(words)
        features = sorted(list(vocab_set))
    else:
        features = vocab

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Indeks tapmaq üçün features list formasına salınır
    feat_list = list(features)

    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in feat_list:
                j = feat_list.index(word)
                embeddings[i, j] += 1

    # Holberton checker-in gözlədiyi numpy.ndarray formatına çeviririk
    features = np.array(features)

    return embeddings, features
