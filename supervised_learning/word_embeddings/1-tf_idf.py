#!/usr/bin/env python3
"""
TF-IDF embedding matrix module
"""
import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix

    parameters:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use for analysis

    returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: numpy.ndarray of the features used for embeddings
    """
    cleaned_sentences = []
    for sentence in sentences:
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
    feat_list = list(features)

    # 1. Term Frequency (TF) matrisini hazırlayırıq
    tf = np.zeros((s, f), dtype=float)
    for i, words in enumerate(cleaned_sentences):
        sentence_len = len(words)
        if sentence_len == 0:
            continue
        for word in words:
            if word in feat_list:
                j = feat_list.index(word)
                tf[i, j] += 1
        # Hər cümlədəki tezliyi cümlənin ümumi uzunluğuna bölürük (TF = count / total_words)
        tf[i] = tf[i] / sentence_len

    # 2. Inverse Document Frequency (IDF) hesablayırıq
    # df: sözün neçə müxtəlif cümlədə işlənmə sayı
    df = np.zeros(f, dtype=float)
    for j, word in enumerate(feat_list):
        for words in cleaned_sentences:
            if word in words:
                df[j] += 1

    # Holberton/scikit-learn smooth_idf formulu: log((1 + N) / (1 + df)) + 1
    idf = np.log((1.0 + s) / (1.0 + df)) + 1.0

    # 3. TF-IDF matrisini vururuq
    embeddings = tf * idf

    # 4. L2 Normalization (hər sətrin kvadrat cəminin kökünə bölünməsi)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Sıfıra bölünmə xətasının qarşısını almaq üçün (0 olan yerlər 1 qalsın)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    return embeddings, np.array(features)
