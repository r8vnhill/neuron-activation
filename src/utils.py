#  Copyright (c) 2023, Roäc Ravenhill.
#  BSD Zero Clause License.
import os
from pathlib import Path

__all__ = ["DATA_PATH"]

DATA_PATH = (Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data").resolve()
