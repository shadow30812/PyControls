import os
import signal
import sys


def stop():
    sys.exit(0)


def flush():
    sys.stdout.flush()


def kill():
    os.kill(os.getppid(), signal.SIGHUP)
