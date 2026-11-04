#!/usr/bin/env python3
"""
Question Answering Loop Module
"""


def question_loop():
    """
    Continually prompts the user with Q: and prints A:
    Exits when user inputs exit, quit, goodbye, or bye (case-insensitive)
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
        else:
            print("A:")


if __name__ == '__main__':
    question_loop()
