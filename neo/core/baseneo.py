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
    are specified in the _necessary_attrs and _recommended_attrs attributes and
    documentation for the child object.
    '''

    # these attributes control relationships, they need to be
    # specified in each child class
    # Child objects that are a container and have a single parent
    _container_child_objects = ()
    # Child objects that have data and have a single parent
    _data_child_objects = ()
    # Parent objects whose children can have a single parent
    _single_parent_objects = ()
    # Child objects that can have multiple parents
    _multi_child_objects = ()
    # Parent objects whose children can have multiple parents
    _multi_parent_objects = ()
    # Properties returning children of children [of children...]
    _child_properties = ()

    # Attributes that an instance is requires to have defined
    _necessary_attrs = ()
    # Attributes that an instance may or may have defined
    _recommended_attrs = (('name', str),
                          ('description', str),
                          ('file_origin', str))

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

    @property
    def _single_child_objects(self):
        '''
        Child objects that have a single parent.
        '''
        return self._container_child_objects + self._data_child_objects

    @property
    def _container_child_containers(self):
        '''
        Containers for child objects that are a container and
        have a single parent.
        '''
        return tuple([child.lower() + 's' for child in
                      self._container_child_objects])

    @property
    def _data_child_containers(self):
        '''
        Containers for child objects that have data and have a single parent.
        '''
        return tuple([child.lower() + 's' for child in
                      self._data_child_objects])

    @property
    def _single_child_containers(self):
        '''
        Containers for child objects with a single parent.
        '''
        return tuple([child.lower() + 's' for child in
                      self._single_child_objects])

    @property
    def _single_parent_containers(self):
        '''
        Containers for parent objects whose children can have a single parent.
        '''
        return tuple([parent.lower() for parent in
                      self._single_parent_objects])

    @property
    def _multi_child_containers(self):
        '''
        Containers for child objects that can have multiple parents.
        '''
        return tuple([child.lower() + 's' for child in
                      self._multi_child_objects])

    @property
    def _multi_parent_containers(self):
        '''
        Containers for parent objects whose children can have multiple parents.
        '''
        return tuple([parent.lower() + 's' for parent in
                      self._multi_parent_objects])

    @property
    def _child_objects(self):
        '''
        All types for child objects.
        '''
        return self._single_child_objects + self._multi_child_objects

    @property
    def _child_containers(self):
        '''
        All containers for child objects.
        '''
        return self._single_child_containers + self._multi_child_containers

    @property
    def _parent_objects(self):
        '''
        All types for parent objects.
        '''
        return self._single_parent_objects + self._multi_parent_objects

    @property
    def _parent_containers(self):
        '''
        All containers for parent objects.
        '''
        return self._single_parent_containers + self._multi_parent_containers

    @property
    def children(self):
        '''
        All child objects stored in the current object.
        '''
        childs = [list(getattr(self, attr)) for attr in self._child_containers]
        return tuple(sum(childs, []))

    @property
    def parents(self):
        '''
        All parent objects storing the current object.
        '''
        single = [getattr(self, attr) for attr in
                  self._single_parent_containers]
        multi = [list(getattr(self, attr)) for attr in
                 self._multi_parent_containers]
        return tuple(single + sum(multi, []))

    def create_many_to_one_relationship(self, force=False, recursive=True):
        """
        For each child of the current object, set its parent to be the current
        object.

        Usage:
        >>> a_block.create_many_to_one_relationship()
        >>> a_block.create_many_to_one_relationship(force=True)

        You want to run populate_RecordingChannel first, because this will
        create new objects that this method will link up.

        If force is True overwrite any existing relationships
        If recursive is True desecend into child objects and create
        relationships there

        """
        classname = self.__class__.__name__.lower()
        for child in self.children:
            if (hasattr(child, classname) and
                    getattr(child, classname) is None or force):
                setattr(child, classname, self)

            if recursive:
                child.create_many_to_one_relationship(force=force,
                                                      recursive=True)

    def create_many_to_many_relationship(self, append=True, recursive=True):
        '''
        For children of the current object that can have more than one parent
        of this type, put the current object in the parent list.

        If append is True add it to the list, otherwise overwrite the list.
        If recursive is True desecend into child objects and create
        relationships there
        '''
        classname = self.__class__.__name__.lower() + 's'
        for child in self.children:
            if not hasattr(child, classname):
                pass
            elif append:
                target = getattr(child, classname)
                if not self in target:
                    target.append(self)
            else:
                setattr(child, classname, [self])

            if recursive:
                child.create_many_to_many_relationship(append=append,
                                                       recursive=True)

    def create_relationship(self, force=False, append=True, recursive=True):
        self.create_many_to_one_relationship(force=force, recursive=False)
        self.create_many_to_many_relationship(append=append, recursive=False)
        if recursive:
            for child in self.children:
                child.create_relationship(force=force, append=append,
                                          recursive=True)

    @property
    def _all_attrs(self):
        '''
        Returns a combination of all required and recommended
        attributes.
        '''
        return self._necessary_attrs + self._recommended_attrs
