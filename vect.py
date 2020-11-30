"""
Vectorize() with support for decorating methods; for example::

    from scipy.stats import rv_continuous
    from scipy_ext import vectorize

    class dist(rv_continuous):
        @vectorize(excluded=('n',), otypes=(float,))
        def _cdf(self, x, n):
            if n < 5:
                return f(x)  # One expensive calculation.
            else:
                return g(x)  # A different expensive calculation.

"""
from __future__ import division

from numpy import arange, stack, vectorize as numpy_vectorize


class _vectorize(numpy_vectorize):
    """
    Method decorator, working just like `numpy.vectorize()`.
    """
    def __get__(self, instance, owner):
        # Vectorize stores the decorated function (former "unbound method")
        # as pyfunc. Bound method's __get__ returns the method itself.
        self.pyfunc = self.pyfunc.__get__(instance, owner)
        return self


def vectorize(*args, **kwargs):
    """
    Allows using `@vectorize` as well as `@vectorize()`.
    """
    if args and callable(args[0]):
        # Guessing the argument is the method.
        return _vectorize(args[0])
    else:
        # Wait for the second call.
        return lambda m: _vectorize(m, *args, **kwargs)


def varange(starts, count):
    """
    Vectorized `arange()` taking a sequence of starts and a count of elements.

    For example::

        >>> varange(1, 5)
        array([1, 2, 3, 4, 5])

        >>> varange((1, 3), 5)
        array([[1, 2, 3, 4, 5],
               [3, 4, 5, 6, 7]])
    """
    try:
        return stack(arange(s, s + count) for s in starts)
    except TypeError:
        return arange(starts, starts + count)
