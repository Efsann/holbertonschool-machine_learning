#!/usr/bin/env python3
"""
Semantic Search Module
"""
import os
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents

    parameters:
        corpus_path: path to the corpus of reference documents
        sentence: the sentence to perform semantic search on

    returns:
        reference text of the document most similar to sentence
    """
    documents = [sentence]
    file_contents = []

    for filename in sorted(os.listdir(corpus_path)):
        file_path = os.path.join(corpus_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(content)
                file_contents.append(content)

    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder/4"
    )

    embeddings = embed(documents)

    sentence_embedding = embeddings[0:1]
    doc_embeddings = embeddings[1:]

    similarity = sentence_embedding @ doc_embeddings.T
    closest_doc_index = similarity.numpy().argmax()

    return file_contents[closest_doc_index]
