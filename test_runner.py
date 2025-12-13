"""
An automation for
```
python -m unittest discover tests
```
to be run from the project's root directory
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

result = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "tests"],
    cwd=PROJECT_ROOT,
    text=True,
    capture_output=True,
)

print(result.stdout)
print(result.stderr)
