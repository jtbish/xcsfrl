import abc


class IntervalABC(metaclass=abc.ABCMeta):
    def __init__(self, lower, upper):
        assert lower <= upper
        self._lower = lower
        self._upper = upper
        self._span = self._calc_span(self._lower, self._upper)

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def span(self):
        return self._span

    @abc.abstractmethod
    def _calc_span(self, lower, upper):
        raise NotImplementedError

    def contains_val(self, val):
        return self._lower <= val <= self._upper

    def does_subsume(self, other):
        """Does this interval subsume other interval?"""
        return (self._lower <= other._lower and self._upper >= other._upper)

    def __str__(self):
        return f"[{self._lower}, {self._upper}]"


class IntegerInterval(IntervalABC):
    def _calc_span(self, lower, upper):
        return upper - lower + 1


class RealInterval(IntervalABC):
    def _calc_span(self, lower, upper):
        return upper - lower
