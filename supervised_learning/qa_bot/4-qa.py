#!/usr/bin/env python3
"""
Multi-reference Question Answering Bot Module
"""

qa_bert = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts in a corpus

    parameters:
        corpus_path: path to the corpus of reference documents

    returns:
        None
    """
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        try:
            user_input = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("A: Goodbye")
            break

        if user_input.lower() in exit_words:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, user_input)
        answer = qa_bert(user_input, reference)

        if answer is None or answer.strip() == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))

