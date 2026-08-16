import os
import signal
import sys
from typing import NoReturn


def flush() -> None:
    """
    Flushes standard output and error streams.
    """
    sys.stdout.flush()
    sys.stderr.flush()


def kill() -> None:
    """
    Forcefully kills the current process.
    """
    os.kill(os.getpid(), signal.SIGKILL)


def stop() -> NoReturn:
    """
    Exits the program gracefully.
    """
    sys.exit(0)
