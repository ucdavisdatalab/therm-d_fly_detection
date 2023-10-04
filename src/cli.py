"""Command-line interface helper functions.
"""

import sys


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


def check_dir_exists_empty(directory, name):
    """Check that a directory exists and is empty. If it doesn't exist, create
    it. If it isn't empty, prompt the user.

    Arguments
    ---------
    directory: Path
        Path to a directory to check.

    name: str
        Name for the directory to use in messages.
    """
    print(f"{name}: '{directory}'")

    if directory.is_dir() and next(directory.iterdir(), None):
        msg = (f"{name} contains files. "
               "Continue and possibly overwrite (y/n)? ")
        if not prompt_yes(msg):
            sys.exit(1)
    else:
        directory.mkdir(parents = True, exist_ok = True)
