# -*- coding: utf-8 -*-
"""
This module defines :class:`BaseNeo`, the abstract base class
used by all :module:`neo.core` classes.
"""

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
    """
    Recursively check that value is either of a "simple" type (number, string,
    date/time) or is a (possibly nested) dict, list or numpy array containing
    only simple types.
    """
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
    """
    First attempt at a policy for merging annotations (intended for use with
    parallel computations using MPI). This policy needs to be discussed
    further, or we could allow the user to specify a policy.

    Current policy:
        For arrays or lists: concatenate
        For dicts: merge recursively
        For strings: concatenate with ';'
        Otherwise: fail if the annotations are not equal
    """
    assert type(a) == type(b), 'type(%s) %s != type(%s) %s' % (a, type(a),
                                                               b, type(b))
    if isinstance(a, dict):
        return merge_annotations(a, b)
    elif isinstance(a, np.ndarray):  # concatenate b to a
        return np.append(a, b)
    elif isinstance(a, list):  # concatenate b to a
        return a + b
    elif isinstance(a, basestring):
        if a == b:
            return a
        else:
            return a + ";" + b
    else:
        assert a == b, '%s != %s' % (a, b)
        return a


def merge_annotations(A, B):
    """
    Merge two sets of annotations.

    Merging follows these rules:
    All keys that are in A or B, but not both, are kept.
    For keys that are present in both:
        For arrays or lists: concatenate
        For dicts: merge recursively
        For strings: concatenate with ';'
        Otherwise: fail if the annotations are not equal
    """
    merged = {}
    for name in A:
        if name in B:
            try:
                merged[name] = merge_annotation(A[name], B[name])
            except BaseException as exc:
                exc.args += ('key %s' % name,)
                raise
        else:
            merged[name] = A[name]
    for name in B:
        if name not in merged:
            merged[name] = B[name]
    logger.debug("Merging annotations: A=%s B=%s merged=%s", A, B, merged)
    return merged


class BaseNeo(object):
    """
    This is the base class from which all Neo objects inherit.

    This class implements support for universally recommended arguments,
    and also sets up the :attr:`annotations` dict for additional arguments.

    Each class can define one or more of the following class attributes:
        :_single_parent_objects: Neo objects that can be parents of this
                                 object. This attribute is used in cases where
                                 only one parent of this class is allowed.
                                 An instance attribute named
                                 class.__name__.lower() will be automatically
                                 defined to hold this parent and will be
                                 initialized to None.
        :_multi_parent_objects: Neo objects that can be parents of this
                                object. This attribute is used in cases where
                                multiple parents of this class is allowed.
                                An instance attribute named
                                class.__name__.lower()+'s' will be
                                automatically defined to hold this parent and
                                will be initialized to an empty list.
        :_necessary_attrs: A list of tuples containing the attributes that the
                           class must have. The tuple can have 2-4 elements.
                           The first element is the attribute name.
                           The second element is the attribute type.
                           The third element is the number of  dimensions
                           (only for numpy arrays and quantities).
                           The fourth element is the dtype of array
                           (only for numpy arrays and quantities).
                           This does NOT include the attributes holding the
                           parents or children of the object.
        :_recommended_attrs: A list of tuples containing the attributes that
                             the class may optionally have.  It uses the same
                             structure as :_necessary_attrs:
        :_repr_pretty_attrs_keys_: The names of attributes printed when
                                   pretty-printing using iPython.

    The following helper properties are available:
        :_parent_objects: All parent objects.
                         :_single_parent_objects: + :_multi_parent_objects:
        :_single_parent_containers: The names of the container attributes used
                                   to store :_single_parent_objects:
        :_multi_parent_containers: The names of the container attributes used
                                   to store :_multi_parent_objects:
        :_parent_containers: All parent container attributes.
                            :_single_parent_containers: +
                            :_multi_parent_containers:
        :parents: All objects that are parents of the current object.
        :_all_attrs: All required and optional attributes.
                     :_necessary_attrs: + :_recommended_attrs:

    The following "universal" methods are available:
        :__init__: Grabs the universally recommended arguments :attr:`name`,
            :attr:`file_origin`, and :attr:`description` and stores them as
            attributes.

            Also takes every additional argument (that is, every argument
            that is not handled by :class:`BaseNeo` or the child class), and
            puts in the dict :attr:`annotations`.

        :annotate(**args): Updates :attr:`annotations` with keyword/value
                           pairs.

        :merge(**args): Merge the contents of another object into this one.
                        The merge method implemented here only merges
                        annotations (see :merge_annotations:).
                        Subclasses should implementt their own merge rules.

        :merge_annotations(**args): Merge the :attr:`annotations` of another
                                    object into this one.

    Each child class should:
        0) describe its parents (if any) and attributes in the relevant
           class attributes. :_recommended_attrs: should append
           BaseNeo._recommended_attrs to the end.
        1) call BaseNeo.__init__(self, name=name, description=description,
                                 file_origin=file_origin, **annotations)
           with the universal recommended arguments, plus optional annotations
        2) process its required arguments in its __new__ or __init__ method
        3) process its non-universal recommended arguments (in its __new__ or
           __init__ method

    Non-keyword arguments should only be used for required arguments.

    The required and recommended arguments for each child class (Neo object)
    are specified in the _necessary_attrs and _recommended_attrs attributes and
    documentation for the child object.
    """

    # these attributes control relationships, they need to be
    # specified in each child class
    # Parent objects whose children can have a single parent
    _single_parent_objects = ()
    # Parent objects whose children can have multiple parents
    _multi_parent_objects = ()

    # Attributes that an instance is requires to have defined
    _necessary_attrs = ()
    # Attributes that an instance may or may have defined
    _recommended_attrs = (('name', str),
                          ('description', str),
                          ('file_origin', str))
    # Attributes that are used for pretty-printing
    _repr_pretty_attrs_keys_ = ("name", "description", "annotations")

    def __init__(self, name=None, description=None, file_origin=None,
                 **annotations):
        """
        This is the base constructor for all Neo objects.

        Stores universally recommended attributes and creates
        :attr:`annotations` from additional arguments not processed by
        :class:`BaseNeo` or the child class.
        """
        # create `annotations` for additional arguments
        _check_annotations(annotations)
        self.annotations = annotations

        # these attributes are recommended for all objects.
        self.name = name
        self.description = description
        self.file_origin = file_origin

        # initialize parent containers
        for parent in self._single_parent_containers:
            setattr(self, parent, None)
        for parent in self._multi_parent_containers:
            setattr(self, parent, [])

    def annotate(self, **annotations):
        """
        Add annotations (non-standardized metadata) to a Neo object.

        Example:

        >>> obj.annotate(key1=value0, key2=value1)
        >>> obj.key2
        value2
        """
        _check_annotations(annotations)
        self.annotations.update(annotations)

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
        """
        Handle pretty-printing the :class:`BaseNeo`.
        """
        pp.text(self.__class__.__name__)
        if self._has_repr_pretty_attrs_():
            pp.breakable()
            self._repr_pretty_attrs_(pp, cycle)

    @property
    def _single_parent_containers(self):
        """
        Containers for parent objects whose children can have a single parent.
        """
        return tuple([parent.lower() for parent in
                      self._single_parent_objects])

    @property
    def _multi_parent_containers(self):
        """
        Containers for parent objects whose children can have multiple parents.
        """
        return tuple([parent.lower() + 's' for parent in
                      self._multi_parent_objects])

    @property
    def _parent_objects(self):
        """
        All types for parent objects.
        """
        return self._single_parent_objects + self._multi_parent_objects

    @property
    def _parent_containers(self):
        """
        All containers for parent objects.
        """
        return self._single_parent_containers + self._multi_parent_containers

    @property
    def parents(self):
        """
        All parent objects storing the current object.
        """
        single = [getattr(self, attr) for attr in
                  self._single_parent_containers]
        multi = [list(getattr(self, attr)) for attr in
                 self._multi_parent_containers]
        return tuple(single + sum(multi, []))

    @property
    def _all_attrs(self):
        """
        Returns a combination of all required and recommended
        attributes.
        """
        return self._necessary_attrs + self._recommended_attrs

    def merge_annotations(self, other):
        """
        Merge annotations from the other object into this one.

        Merging follows these rules:
        All keys that are in the either object, but not both, are kept.
        For keys that are present in both objects:
            For arrays or lists: concatenate the two arrays
            For dicts: merge recursively
            For strings: concatenate with ';'
            Otherwise: fail if the annotations are not equal
        """
        merged_annotations = merge_annotations(self.annotations,
                                               other.annotations)
        self.annotations.update(merged_annotations)

    def merge(self, other):
        """
        Merge the contents of another object into this one.

        See :meth:`merge_annotations` for details of the merge operation.
        """
        self.merge_annotations(other)
