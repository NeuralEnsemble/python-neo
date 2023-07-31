"""
This module implements :class:`FilterCondition`, which enables use of different filter conditions
for neo.core.container.filter.

Classes:
    - :class:`FilterCondition`: Abstract base class for defining filter conditions.
    - :class:`Equals`: Filter condition to check if a value is equal to the control value.
    - :class:`IsNot`: Filter condition to check if a value is not equal to the control value.
    - :class:`LessThanOrEquals`: Filter condition to check if a value is less than or equal to the
    control value.
    - :class:`GreaterThanOrEquals`: Filter condition to check if a value is greater than or equal to
    the control value.
    - :class:`LessThan`: Filter condition to check if a value is less than the control value.
    - :class:`GreaterThan`: Filter condition to check if a value is greater than the control value.
    - :class:`IsIn`: Filter condition to check if a value is in a list or equal to the control
    value.
    - :class:`InRange`: Filter condition to check if a value is in a specified range.

The provided classes allow users to select filter conditions and use them with
:func:`neo.core.container.filter()` to perform specific filtering operations on data.
"""
from abc import ABC, abstractmethod
from numbers import Number

class FilterCondition(ABC):
    """
    FilterCondition object is given as parameter to container.filter():

    Usage:
        segment.filter(my_annotation=<FilterCondition>) or
        segment=filter({'my_annotation': <FilterCondition>})
    """
    @abstractmethod
    def __init__(self, z):
        """
        Initialize new FilterCondition object.

        Parameters:
            z: Any - The control value to be used for filtering.

        This is an abstract base class and should not be instantiated directly.
        """

    @abstractmethod
    def evaluate(self, x):
        """
        Evaluate the filter condition for given value.

        Parameters:
            x: Any - The value to be compared with the control value.

        Returns:
            bool: True if the condition is satisfied, False otherwise.

        This method should be implemented in subclasses.
        """


class Equals(FilterCondition):
    """
    Filter condition to check if target value is equal to the control value.
    """
    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x == self.control


class IsNot(FilterCondition):
    """
    Filter condition to check if target value is not equal to the control value.
    """
    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x != self.control


class LessThanOrEquals(FilterCondition):
    """
    Filter condition to check if target value is less than or equal to the control value.
    """
    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x <= self.control


class GreaterThanOrEquals(FilterCondition):
    """
    Filter condition to check if target value is greater than or equal to the control value.
    """
    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x >= self.control


class LessThan(FilterCondition):
    """
    Filter condition to check if target value is less than the control value.
    """
    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x < self.control


class GreaterThan(FilterCondition):
    """
    Filter condition to check if target value is greater than the control value.
    """
    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        return x > self.control


class IsIn(FilterCondition):
    """
    Filter condition to check if target is in control.
    """
    def __init__(self, z):
        self.control = z

    def evaluate(self, x):
        if isinstance(self.control, list):
            return x in self.control
        if isinstance(self.control, tuple):
            return x in self.control
        if isinstance(self.control, set):
            return x in self.control
        if isinstance(self.control, int):
            return x == self.control

        raise SyntaxError('parameter not of type list or int')


class InRange(FilterCondition):
    """
    Filter condition to check if a value is in a specified range.

    Usage:
        InRange(upper_bound, upper_bound, left_closed=False, right_closed=False)

    Parameters:
        lower_bound: int - The lower bound of the range.
        upper_bound: int - The upper bound of the range.
        left_closed: bool - If True, the range includes the lower bound (lower_bound <= x).
        right_closed: bool - If True, the range includes the upper bound (x <= upper_bound).
    """
    def __init__(self, lower_bound, upper_bound, left_closed=False, right_closed=False):
        if not isinstance(lower_bound, Number) or not isinstance(upper_bound, Number):
            raise ValueError("parameter is not a number")

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.left_closed = left_closed
        self.right_closed = right_closed

    def evaluate(self, x):
        if not self.left_closed and not self.right_closed:
            return self.lower_bound <= x <= self.upper_bound
        if not self.left_closed and self.right_closed:
            return self.lower_bound <= x < self.upper_bound
        if self.left_closed and not self.right_closed:
            return self.lower_bound < x <= self.upper_bound
        return self.lower_bound < x < self.upper_bound
