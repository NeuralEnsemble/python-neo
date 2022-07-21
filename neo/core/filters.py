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


class Equal(FilterCondition):

    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x == self.control


class IsNot(FilterCondition):

    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x != self.control


class LessThanEqual(FilterCondition):

    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x <= self.control


class GreaterThanEqual(FilterCondition):

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
        if type(self.control) == list:
            return x in self.control
        elif type(self.control) == int:
            return x == self.control
        else:
            raise SyntaxError('parameter not of type list or int')


class InRange(FilterCondition):

    def __init__(self, a, b, left_closed=False, right_closed=False):
        if type(a) != int or type(b) != int:
            raise SyntaxError("parameters not of type int")
        else:
            self.a = a
            self.b = b
            self.left_closed = left_closed
            self.right_closed = right_closed

    def evaluate(self, x):
        if not self.left_closed and not self.right_closed:
            return self.a <= x <= self.b
        elif not self.left_closed and self.right_closed:
            return self.a <= x < self.b
        elif self.left_closed and not self.right_closed:
            return self.a < x <= self.b
        else:
            return self.a < x < self.b
