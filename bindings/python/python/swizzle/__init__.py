"""Swizzle Runtime: open engine for running AI locally.

`swizzle.Model` loads a .gguf or .ktc file; `swizzle.llama_cpp` offers the
`Llama` class of llama-cpp-python on top of it.
"""

from ._swizzle import Model, TextStream, __version__

__all__ = ["Model", "TextStream", "__version__"]
