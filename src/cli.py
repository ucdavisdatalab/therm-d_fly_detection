"""Command-line interface helper functions.
"""


def prompt_yes(prompt):
    """Prompt the user with a yes or no question and return True if they
    respond yes.
    """
    while True:
        match input(prompt).lower():
            case "y" | "yes":
                return True
            case "n" | "no":
                return False
