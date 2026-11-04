#!/usr/bin/env python3
"""
Question Answering Loop with Reference Text Module
"""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions from a reference text in a loop

    parameters:
        reference: string containing the reference document

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

        answer = question_answer(user_input, reference)

        if answer is None or answer.strip() == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
