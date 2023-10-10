"""Command-line interface helper functions.
"""

from pathlib import Path
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


def prompt_overwrite(path, label, mkdir = False):
    """Check whether a file exists or directory is empty and prompt the user to
    overwrite if not.

    Arguments
    ---------
    path: str or Path
        Path to a file or directory.

    label: str
        A label for the file or directory, to use in printouts.

    mkdir: bool
        If path is a directory and doesn't exist, create it?
    """
    path = Path(path)
    print(f"{label}: '{path}'")

    # If the file/directory doesn't exist, no need to check further.
    if not path.exists():
        if mkdir:
            path.mkdir(parents = True, exist_ok = True)
        return

    # Otherwise...
    if path.is_dir() and next(path.iterdir(), None) is not None:
        msg = (f"Directory '{path}' contains files.\n"
               "  Continue and possibly overwrite? [y/n] ")
    else:
        msg = (f"File '{path}' exists.\n"
               "  Continue and possibly overwrite? [y/n] ")

    if not prompt_yes(msg):
        sys.exit(1)
