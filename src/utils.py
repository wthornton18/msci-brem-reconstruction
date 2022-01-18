from itertools import compress, islice, chain
from collections import namedtuple
from typing import Generator, Iterable, Type, TypeVar
import numpy as np
from pylorentz import Momentum4 as M4


class Momentum4(M4):
    @staticmethod
    def m_px_py_pz(m, px, py, pz):
        e = np.sqrt(m ** 2 + px ** 2 + py ** 2 + pz ** 2)
        return Momentum4(e, px, py, pz)


data = namedtuple(
    "data",
    [
        "event_number",
        "e_px",
        "e_py",
        "e_pz",
        "e_m",
        "e_x",
        "e_y",
        "e_z",
        "j_px",
        "j_py",
        "j_pz",
        "j_m",
        "b_px",
        "b_py",
        "b_pz",
        "b_m",
        "k_px",
        "k_py",
        "k_pz",
        "k_m",
        "std_e_px",
        "std_e_py",
        "std_e_pz",
        "std_e_e",
        "brem_x",
        "brem_y",
        "brem_z",
        "brem_px",
        "brem_py",
        "brem_pz",
        "brem_e",
        "nbrems",
        "nelectrons",
        "etrack_type",
    ],
)


def ichunked(seq, chunksize):
    """Yields items from an iterator in iterable chunks."""
    it = iter(seq)
    while True:
        try:
            yield chain([next(it)], islice(it, chunksize - 1))
        except StopIteration:
            return


_T = TypeVar("_T")


def chunked(seq: Iterable[_T], chunksize=2):
    """Yields items from an iterator in list chunks."""
    for chunk in ichunked(seq, chunksize):
        yield list(chunk)


_Named_Tuple = TypeVar("_Named_Tuple", bound=namedtuple)


def named_zip(named_tuple: _Named_Tuple, *iterables) -> Generator[_Named_Tuple, None, None]:
    def _namedzip_gen(*iterables):
        zipped = zip(*iterables)
        return _named_zip_generator(zipped, named_tuple)

    if iterables:
        return _namedzip_gen(*iterables)
    else:
        return _namedzip_gen


def _named_zip_generator(zipped, named_tuple):
    for vals in zipped:
        yield named_tuple(*vals)


class Vector3:
    def __init__(self, *x):
        if len(x) == 1 and isinstance(x[0], Vector3):
            x = x[0]
            self._values = np.array(x.components)
        elif len(x) == 3:
            self._values = np.array(list(x))
        else:
            raise TypeError("3-vector expects 3 values")

    def __repr__(self):
        if self._values.ndim == 1:
            pattern = "%g"
        else:
            pattern = "%r"

        return "%s(%s)" % (self.__class__.__name__, ", ".join([pattern % _ for _ in self._values]))

    def __sub__(self, other):

        vector = self.__class__(self)
        vector -= other
        return vector

    def __isub__(self, other):
        self._values = self.components - Vector3(*other).components
        return self

    def __iter__(self):
        return iter(self._values)

    @property
    def components(self):
        return self._values

    @property
    def trans(self):
        return np.sqrt(self._values[0] ** 2 + self._values[1] ** 2)

    @property
    def phi(self):
        return np.arctan2(self._values[1], self._values[0])

    @property
    def theta(self):
        return np.arctan2(self.trans, self._values[2])


class Position3(Vector3):
    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]
