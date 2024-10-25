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
from typing import Union, Any


class FilterCondition(ABC):
    """
    FilterCondition object is given as parameter to container.filter():

    Usage:
        segment.filter(my_annotation=<FilterCondition>) or
        segment=filter({'my_annotation': <FilterCondition>})
    """

    @abstractmethod
    def __init__(self, control: Any) -> None:
        """
        Initialize new FilterCondition object.

        Parameters
        ----------
        control: Any
            The control value to be used for filtering.

        Notes
        -----
        This is an abstract base class and should not be instantiated directly.
        """

    @abstractmethod
    def evaluate(self, compare: Any) -> bool:
        """
        Evaluate the filter condition for given value.

        Parameters:
        -----------
        compare: Any
        The value to be compared with the control value.

        Returns
        -------
        bool: True if the condition is satisfied, False otherwise.

        Notes
        -----
        This method should be implemented in subclasses.
        """


class Equals(FilterCondition):
    """
    Filter condition to check if target value is equal to the control value.
    """

    def __init__(self, control: Any) -> None:
        self.control = control

    def evaluate(self, compare: Any) -> bool:
        return compare == self.control


class IsNot(FilterCondition):
    """
    Filter condition to check if target value is not equal to the control value.
    """

    def __init__(self, control: Any) -> None:
        self.control = control

    def evaluate(self, compare: Any) -> bool:
        return compare != self.control


class LessThanOrEquals(FilterCondition):
    """
    Filter condition to check if target value is less than or equal to the control value.
    """

    def __init__(self, control: Number) -> None:
        self.control = control

    def evaluate(self, compare: Number) -> bool:
        return compare <= self.control


class GreaterThanOrEquals(FilterCondition):
    """
    Filter condition to check if target value is greater than or equal to the control value.
    """

    def __init__(self, control: Number) -> None:
        self.control = control

    def evaluate(self, compare: Number) -> bool:
        return compare >= self.control


class LessThan(FilterCondition):
    """
    Filter condition to check if target value is less than the control value.
    """

    def __init__(self, control: Number) -> None:
        self.control = control

    def evaluate(self, compare: Number) -> bool:
        return compare < self.control


class GreaterThan(FilterCondition):
    """
    Filter condition to check if target value is greater than the control value.
    """

    def __init__(self, control: Number) -> None:
        self.control = control

    def evaluate(self, compare: Number) -> bool:
        return compare > self.control


class IsIn(FilterCondition):
    """
    Filter condition to check if target is in control.
    """

    def __init__(self, control: Union[list, tuple, set, int]) -> None:
        self.control = control

    def evaluate(self, compare: Any) -> bool:
        if isinstance(self.control, (list, tuple, set)):
            return compare in self.control
        if isinstance(self.control, int):
            return compare == self.control

        raise SyntaxError("parameter not of type list, tuple, set or int")


class InRange(FilterCondition):
    """
    Filter condition to check if a value is in a specified range.

    Parameters
    -----------
    lower_bound: int
        The lower bound of the range.
    upper_bound: int
        The upper bound of the range.
    left_closed: bool
        If True, the range includes the lower bound (lower_bound <= compare).
    right_closed: bool
        If True, the range includes the upper bound (compare <= upper_bound).

    Returns
    -------
    bool:
    whether the values are in range

    Examples
    --------

    >>> InRange(lower_bound, upper_bound, left_closed=False, right_closed=False)
    """

    def __init__(
        self, lower_bound: Number, upper_bound: Number, left_closed: bool = False, right_closed: bool = False
    ) -> None:
        if not isinstance(lower_bound, Number) or not isinstance(upper_bound, Number):
            raise ValueError("parameter is not a number")

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.left_closed = left_closed
        self.right_closed = right_closed

    def evaluate(self, compare: Number) -> bool:
        if not self.left_closed and not self.right_closed:
            return self.lower_bound <= compare <= self.upper_bound
        if not self.left_closed and self.right_closed:
            return self.lower_bound <= compare < self.upper_bound
        if self.left_closed and not self.right_closed:
            return self.lower_bound < compare <= self.upper_bound
        return self.lower_bound < compare < self.upper_bound
