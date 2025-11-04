"""Type definitions for generativepoetry."""

from typing import List, Tuple, Union

# Type aliases
WordList = List[str]
TextOrWordList = Union[str, List[str]]
Coordinate = Tuple[float, float]
ColorRGB = Tuple[float, float, float]
