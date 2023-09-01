from os import PathLike
from typing import TypeAlias

from numpy.typing import NDArray

__all__ = ["FilePath", "NDArray"]


FilePath: TypeAlias = str | PathLike[str]
