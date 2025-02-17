import sys
import subprocess
from typing import Optional
from contextlib import nullcontext
from rich.console import Console

def run_command(cmd: str, verbose=False, console: Console=None) -> Optional[str]:
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        console.rule("[bold red] :skull: :skull: :skull: ERROR :skull: :skull: :skull: ", style="red")
        console.print(f"[bold red]Error running command: {cmd}")
        console.rule(style="red")
        console.print(out.stderr.decode("utf-8"))
        sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out

def status(msg: str, spinner: str = "bouncingBall", verbose: bool = False, console: Console = None):
    """A context manager that does nothing is verbose is True. Otherwise it hides logs under a message.

    Args:
        msg: The message to log.
        spinner: The spinner to use.
        verbose: If True, print all logs, else hide them.
    """
    if verbose:
        return nullcontext()
    return console.status(msg, spinner=spinner)

def prompt_user_command(command_name: str, console: Console):
    choice = input(f"Do you want to run {command_name}? [y]/n\n")
    if choice.lower() == "n":
        console.log("[bold red]:x: Exiting...")   
        sys.exit(0)