# -*- coding: utf-8 -*-
'''
This module defines :class:`BaseNeo`, the abstract base class
used by all :module:`neo.core` classes.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

from datetime import datetime, date, time, timedelta
from decimal import Decimal
import logging
from numbers import Number

import numpy as np

ALLOWED_ANNOTATION_TYPES = (int, float, complex,
                            str, bytes,
                            type(None),
                            datetime, date, time, timedelta,
                            Number, Decimal,
                            np.number, np.bool_)

# handle both Python 2 and Python 3
try:
    ALLOWED_ANNOTATION_TYPES += (long, unicode)
except NameError:
    pass

try:
    basestring
except NameError:
    basestring = str

logger = logging.getLogger("Neo")


def _check_annotations(value):
    '''
    Recursively check that value is either of a "simple" type (number, string,
    date/time) or is a (possibly nested) dict, list or numpy array containing
    only simple types.
    '''
    if isinstance(value, np.ndarray):
        if not issubclass(value.dtype.type, ALLOWED_ANNOTATION_TYPES):
            raise ValueError("Invalid annotation. NumPy arrays with dtype %s"
                             "are not allowed" % value.dtype.type)
    elif isinstance(value, dict):
        for element in value.values():
            _check_annotations(element)
    elif isinstance(value, (list, tuple)):
        for element in value:
            _check_annotations(element)
    elif not isinstance(value, ALLOWED_ANNOTATION_TYPES):
        raise ValueError("Invalid annotation. Annotations of type %s are not"
                         "allowed" % type(value))


def merge_annotation(a, b):
    '''
    First attempt at a policy for merging annotations (intended for use with
    parallel computations using MPI). This policy needs to be discussed
    further, or we could allow the user to specify a policy.

    Current policy:
        for arrays: concatenate the two arrays
        otherwise: fail if the annotations are not equal
    '''
    assert type(a) == type(b)
    if isinstance(a, dict):
        return merge_annotations(a, b)
    elif isinstance(a, np.ndarray):  # concatenate b to a
        return np.append(a, b)
    elif isinstance(a, basestring):
        if a == b:
            return a
        else:
            return a + ";" + b
    else:
        assert a == b
        return a


def merge_annotations(A, B):
    '''
    Merge two sets of annotations.
    '''
    merged = {}
    for name in A:
        if name in B:
            merged[name] = merge_annotation(A[name], B[name])
        else:
            merged[name] = A[name]
    for name in B:
        if name not in merged:
            merged[name] = B[name]
    logger.debug("Merging annotations: A=%s B=%s merged=%s", A, B, merged)
    return merged


class BaseNeo(object):
    '''This is the base class from which all Neo objects inherit.

    This class implements support for universally recommended arguments,
    and also sets up the :attr:`annotations` dict for additional arguments.

    The following "universal" methods are available:
        :__init__: Grabs the universally recommended arguments :attr:`name`,
            :attr:`file_origin`, and :attr:`description` and stores them as
            attributes.

            Also takes every additional argument (that is, every argument
            that is not handled by :class:`BaseNeo` or the child class), and
            puts in the dict :attr:`annotations`.

        :annotate(**args): Updates :attr:`annotations` with keyword/value
                           pairs.

    Each child class should:
        0) call BaseNeo.__init__(self, name=name, file_origin=file_origin,
                                 description=description, **annotations)
           with the universal recommended arguments, plus optional annotations
        1) process its required arguments in its __new__ or __init__ method
        2) process its non-universal recommended arguments (in its __new__ or
           __init__ method

    Non-keyword arguments should only be used for required arguments.

    The required and recommended arguments for each child class (Neo object)
    are specified in :module:`neo.description` and the documentation for the
    child object.
    '''

    def __init__(self, name=None, file_origin=None, description=None,
                 **annotations):
        '''
        This is the base constructor for all Neo objects.

        Stores universally recommended attributes and creates
        :attr:`annotations` from additional arguments not processed by
        :class:`BaseNeo` or the child class.
        '''
        # create `annotations` for additional arguments
        _check_annotations(annotations)
        self.annotations = annotations

        # these attributes are recommended for all objects.
        self.name = name
        self.description = description
        self.file_origin = file_origin

    def annotate(self, **annotations):
        '''
        Add annotations (non-standardized metadata) to a Neo object.

        Example:

        >>> obj.annotate(key1=value0, key2=value1)
        >>> obj.key2
        value2
        '''
        _check_annotations(annotations)
        self.annotations.update(annotations)

    _repr_pretty_attrs_keys_ = ["name", "description", "annotations"]

    def _has_repr_pretty_attrs_(self):
        return any(getattr(self, k) for k in self._repr_pretty_attrs_keys_)

    def _repr_pretty_attrs_(self, pp, cycle):
        first = True
        for key in self._repr_pretty_attrs_keys_:
            value = getattr(self, key)
            if value:
                if first:
                    first = False
                else:
                    pp.breakable()
                with pp.group(indent=1):
                    pp.text("{0}: ".format(key))
                    pp.pretty(value)

    def _repr_pretty_(self, pp, cycle):
        '''
        Handle pretty-printing the :class:`BaseNeo`.
        '''
        pp.text(self.__class__.__name__)
        if self._has_repr_pretty_attrs_():
            pp.breakable()
            self._repr_pretty_attrs_(pp, cycle)
