import os
import signal
import sys


def flush():
    """
    Flushes standard output and error streams.
    """
    sys.stdout.flush()
    sys.stderr.flush()


def kill():
    """
    Forcefully kills the current process.
    """
    os.kill(os.getpid(), signal.SIGKILL)


def stop():
    """
    Exits the program gracefully.
    """
    sys.exit(0)
