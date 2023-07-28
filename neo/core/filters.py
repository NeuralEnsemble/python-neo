"""
This module implements :class:`FilterCondition`, which enables use of different filter conditions for
neo.core.container.filter.
"""


class FilterCondition:
    """
        FilterCondition object is given as parameter to container.filter():

        segment.filter(my_annotation=<FilterCondition>) or
        segment=filter({'my_annotation': <FilterCondition>})
    """

    def __init__(self, z):
        pass

    def evaluate(self, x):
        raise NotImplementedError()


class Equals(FilterCondition):

    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x == self.control


class IsNot(FilterCondition):

    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x != self.control


class LessThanEquals(FilterCondition):

    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x <= self.control


class GreaterThanEquals(FilterCondition):

    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x >= self.control


class LessThan(FilterCondition):

    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x < self.control


class GreaterThan(FilterCondition):

    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x > self.control


class IsIn(FilterCondition):

    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        if isinstance(self.control, list):
            return x in self.control
        if isinstance(self.control, int):
            return x == self.control

        raise SyntaxError('parameter not of type list or int')


class InRange(FilterCondition):

    def __init__(self, a, b, left_closed=False, right_closed=False):
        if not isinstance(a, int) or not isinstance(b, int):
            raise SyntaxError("parameters not of type int")

        self.a = a
        self.b = b
        self.left_closed = left_closed
        self.right_closed = right_closed

    def evaluate(self, x):
        if not self.left_closed and not self.right_closed:
            return self.a <= x <= self.b
        if not self.left_closed and self.right_closed:
            return self.a <= x < self.b
        if self.left_closed and not self.right_closed:
            return self.a < x <= self.b
        return self.a < x < self.b
