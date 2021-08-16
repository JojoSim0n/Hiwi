
import itertools

from typing import (
    Iterable,
    Iterator,
    TypeVar,
)


T = TypeVar("T")

def endless_iter(iterable: Iterable[T]) -> Iterator[T]:
    """Generator that endlessly yields elements from iterable.
    If any call to `iter(iterable)` has no elements, then this function raises
    ValueError.
    >>> x = range(2)
    >>> it = endless_iter(x)
    >>> next(it)
    0
    >>> next(it)
    1
    >>> next(it)
    0
    """
    try:
        next(iter(iterable))
    except StopIteration:
        err = ValueError(f"iterable {iterable} had no elements to iterate over.")
        raise err

    return itertools.chain.from_iterable(itertools.repeat(iterable))