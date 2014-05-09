# -*- coding: utf-8 -*-
"""
This module implements generic container base class that all neo container
object inherit from.  It provides shared methods for all container types.

:class:`Container` is derived from :class:`BaseNeo` but is
defined in :module:`neo.core.analogsignalarray`.
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

from neo.core.baseneo import BaseNeo


def unique_objs(objs):
    """
    Return a list of objects in the list objs where all objects are unique
    using the "is" test.
    """
    seen = set()
    return [obj for obj in objs
            if id(obj) not in seen and not seen.add(id(obj))]


def filterdata(data, targdict=None, objects=None, **kwargs):
    """
    Return a list of the objects in data matching *any* of the search terms
    in either their attributes or annotations.  Search terms can be
    provided as keyword arguments or a dictionary, either as a positional
    argument after data or to the argument targdict.  targdict can also
    be a list of dictionaries, in which case the filters are applied
    sequentially.  If targdict and kwargs are both supplied, the
    targdict filters are applied first, followed by the kwarg filters.


    objects (optional) should be the name of a Neo object type,
    a neo object class, or a list of one or both of these.  If specified,
    only these objects will be returned.
    """

    # if objects are specified, get the classes
    if objects:
        if hasattr(objects, 'lower') or isinstance(objects, type):
            objects = [objects]
    elif objects is not None:
        return []

    # handle cases with targdict
    if targdict is None:
        targdict = kwargs
    elif not kwargs:
        pass
    elif hasattr(targdict, 'keys'):
        targdict = [targdict, kwargs]
    else:
        targdict += [kwargs]

    if not targdict:
        return []

    # if multiple dicts are provided, apply each filter sequentially
    if not hasattr(targdict, 'keys'):
        # for performance reasons, only do the object filtering on the first
        # iteration
        results = filterdata(data, targdict=targdict[0], objects=objects)
        for targ in targdict[1:]:
            results = filterdata(results, targdict=targ)
        return results

    # do the actual filtering
    results = []
    for key, value in sorted(targdict.items()):
        for obj in data:
            if (hasattr(obj, key) and getattr(obj, key) == value and
                    all([obj is not res for res in results])):
                results.append(obj)
            elif (key in obj.annotations and obj.annotations[key] == value and
                    all([obj is not res for res in results])):
                results.append(obj)

    # keep only objects of the correct classes
    if objects:
        results = [result for result in results if
                   result.__class__ in objects or
                   result.__class__.__name__ in objects]
    return results


class Container(BaseNeo):
    """
    This is the base class from which Neo container objects inherit.  It
    derives from :class:`BaseNeo`.

    In addition to the setup :class:`BaseNeo` does, this class also
    automatically sets up the lists to hold the children of the object.

    Each class can define one or more of the following class attributes
    (in  addition to those of BaseNeo):
        :_container_child_objects: Neo container objects that can be children
                                   of this object. This attribute is used in
                                   cases where the child can only have one
                                   parent of this type. An instance attribute
                                   named class.__name__.lower()+'s' will be
                                   automatically defined to hold this child and
                                   will be initialized to an empty list.
        :_data_child_objects: Neo data objects that can be children
                              of this object. An instance attribute named
                              class.__name__.lower()+'s' will be automatically
                              defined to hold this child and will be
                              initialized to an empty list.
        :_multi_child_objects: Neo container objects that can be children
                               of this object. This attribute is used in
                               cases where the child can have multiple
                               parents of this type. An instance attribute
                               named class.__name__.lower()+'s' will be
                               automatically defined to hold this child and
                               will be initialized to an empty list.
        :_child_properties: Properties that return sub-children of a particular
                            type.  These properties must still be defined.
                            This is mostly used for generate_diagram.
        :_repr_pretty_containers: The names of containers attributes printed
                                  when pretty-printing using iPython.

    The following helper properties are available
    (in  addition to those of BaseNeo):
        :_single_child_objects: All neo container objects that can be children
                                of this object and where the child can only
                                have one parent of this type.
                                :_container_child_objects: +
                                :_data_child_objects:
        :_child_objects: All child objects.
                         :_single_child_objects: + :_multi_child_objects:
        :_container_child_containers: The names of the container attributes
                                      used to store :_container_child_objects:
        :_data_child_containers: The names of the container attributes used
                                 to store :_data_child_objects:
        :_single_child_containers: The names of the container attributes used
                                   to store :_single_child_objects:
        :_multi_child_containers: The names of the container attributes used
                                  to store :_multi_child_objects:
        :_child_containers: All child container attributes.
                            :_single_child_containers: +
                            :_multi_child_containers:
        :_single_children: All objects that are children of the current object
                           where the child can only have one parent of
                           this type.
        :_multi_children: All objects that are children of the current object
                          where the child can have multiple parents of
                          this type.
        :data_children: All data objects that are children of
                        the current object.
        :container_children: All container objects that are children of
                             the current object.
        :children: All Neo objects that are children of the current object.
        :data_children_recur: All data objects that are children of
                              the current object or any of its children,
                              any of its children's children, etc.
        :container_children_recur: All container objects that are children of
                                   the current object or any of its children,
                                   any of its children's children, etc.
        :children_recur: All Neo objects that are children of
                         the current object or any of its children,
                         any of its children's children, etc.

    The following "universal" methods are available
    (in  addition to those of BaseNeo):
        :size: A dictionary where each key is an attribute storing child
               objects and the value is the number of objects stored in that
               attribute.

        :filter(**args): Retrieves children of the current object that
                         have particular properties.

        :list_children_by_class(**args): Retrieves all children of the current
                                         object recursively that are of a
                                         particular class.

        :create_many_to_one_relationship(**args): For each child of the current
                                                  object that can only have a
                                                  single parent, set its parent
                                                  to be the current object.

        :create_many_to_many_relationship(**args): For children of the current
                                                   object that can have more
                                                   than one parent of this
                                                   type, put the current
                                                   object in the parent list.

        :create_relationship(**args): Combines
                                      :create_many_to_one_relationship: and
                                      :create_many_to_many_relationship:

        :merge(**args): Annotations are merged based on the rules of
                        :merge_annotations:.  Child objects with the same name
                        and a :merge: method are merged using that method.
                        Other child objects are appended to the relevant
                        container attribute.  Parents attributes are NOT
                        changed in this operation.
                        Unlike :BaseNeo.merge:, this method implements
                        all necessary merge rules for a container class.


    Each child class should:
        0) call Container.__init__(self, name=name, description=description,
                                   file_origin=file_origin, **annotations)
           with the universal recommended arguments, plus optional annotations
        1) process its required arguments in its __new__ or __init__ method
        2) process its non-universal recommended arguments (in its __new__ or
           __init__ method
    """
    # Child objects that are a container and have a single parent
    _container_child_objects = ()
    # Child objects that have data and have a single parent
    _data_child_objects = ()
    # Child objects that can have multiple parents
    _multi_child_objects = ()
    # Properties returning children of children [of children...]
    _child_properties = ()
    # Containers that are listed when pretty-printing
    _repr_pretty_containers = ()

    def __init__(self, name=None, description=None, file_origin=None,
                 **annotations):
        """
        Initalize a new :class:`Container` instance.
        """
        super(Container, self).__init__(name=name, description=description,
                                        file_origin=file_origin, **annotations)

        # initialize containers
        for container in self._child_containers:
            setattr(self, container, [])

    @property
    def _single_child_objects(self):
        """
        Child objects that have a single parent.
        """
        return self._container_child_objects + self._data_child_objects

    @property
    def _container_child_containers(self):
        """
        Containers for child objects that are a container and
        have a single parent.
        """
        return tuple([child.lower() + 's' for child in
                      self._container_child_objects])

    @property
    def _data_child_containers(self):
        """
        Containers for child objects that have data and have a single parent.
        """
        return tuple([child.lower() + 's' for child in
                      self._data_child_objects])

    @property
    def _single_child_containers(self):
        """
        Containers for child objects with a single parent.
        """
        return tuple([child.lower() + 's' for child in
                      self._single_child_objects])

    @property
    def _multi_child_containers(self):
        """
        Containers for child objects that can have multiple parents.
        """
        return tuple([child.lower() + 's' for child in
                      self._multi_child_objects])

    @property
    def _child_objects(self):
        """
        All types for child objects.
        """
        return self._single_child_objects + self._multi_child_objects

    @property
    def _child_containers(self):
        """
        All containers for child objects.
        """
        return self._single_child_containers + self._multi_child_containers

    @property
    def _single_children(self):
        """
        All child objects that can only have single parents.
        """
        childs = [list(getattr(self, attr)) for attr in
                  self._single_child_containers]
        return tuple(sum(childs, []))

    @property
    def _multi_children(self):
        """
        All child objects that can have multiple parents.
        """
        childs = [list(getattr(self, attr)) for attr in
                  self._multi_child_containers]
        return tuple(sum(childs, []))

    @property
    def data_children(self):
        """
        All data child objects stored in the current object.
        Not recursive.
        """
        childs = [list(getattr(self, attr)) for attr in
                  self._data_child_containers]
        return tuple(sum(childs, []))

    @property
    def container_children(self):
        """
        All container child objects stored in the current object.
        Not recursive.
        """
        childs = [list(getattr(self, attr)) for attr in
                  self._container_child_containers +
                  self._multi_child_containers]
        return tuple(sum(childs, []))

    @property
    def children(self):
        """
        All child objects stored in the current object.
        Not recursive.
        """
        return self.data_children + self.container_children

    @property
    def data_children_recur(self):
        """
        All data child objects stored in the current object,
        obtained recursively.
        """
        childs = [list(child.data_children_recur) for child in
                  self.container_children]
        return self.data_children + tuple(sum(childs, []))

    @property
    def container_children_recur(self):
        """
        All container child objects stored in the current object,
        obtained recursively.
        """
        childs = [list(child.container_children_recur) for child in
                  self.container_children]
        return self.container_children + tuple(sum(childs, []))

    @property
    def children_recur(self):
        """
        All child objects stored in the current object,
        obtained recursively.
        """
        return self.data_children_recur + self.container_children_recur

    @property
    def size(self):
        """
        Get dictionary containing the names of child containers in the current
        object as keys and the number of children of that type as values.
        """
        return dict((name, len(getattr(self, name)))
                    for name in self._child_containers)

    def filter(self, targdict=None, data=True, container=False, recursive=True,
               objects=None, **kwargs):
        """
        Return a list of child objects matching *any* of the search terms
        in either their attributes or annotations.  Search terms can be
        provided as keyword arguments or a dictionary, either as a positional
        argument after data or to the argument targdict.  targdict can also
        be a list of dictionaries, in which case the filters are applied
        sequentially.  If targdict and kwargs are both supplied, the
        targdict filters are applied first, followed by the kwarg filters.

        If data is True (default), include data objects.
        If container is True (default False), include container objects.
        If recursive is True (default), descend into child containers for
        objects.

        objects (optional) should be the name of a Neo object type,
        a neo object class, or a list of one or both of these.  If specified,
        only these objects will be returned.  Note that if recursive is True,
        containers not in objects will still be descended into.
        This overrides data and container.


        Examples::

            >>> obj.filter(name="Vm")
        """
        # if objects are specified, get the classes
        if objects:
            data = True
            container = True

        children = []
        # get the objects we want
        if data:
            if recursive:
                children.extend(self.data_children_recur)
            else:
                children.extend(self.data_children)
        if container:
            if recursive:
                children.extend(self.container_children_recur)
            else:
                children.extend(self.container_children)

        return filterdata(children, objects=objects,
                          targdict=targdict, **kwargs)

    def list_children_by_class(self, cls):
        """
        List all children of a particular class recursively.

        You can either provide a class object, a class name,
        or the name of the container storing the class.
        """
        if not hasattr(cls, 'lower'):
            cls = cls.__name__
        cls = cls.lower()
        if cls[-1] != 's':
            cls = cls + 's'
        objs = list(getattr(self, cls, []))
        for child in self.container_children_recur:
            objs.extend(getattr(child, cls, []))
        return objs

    def create_many_to_one_relationship(self, force=False, recursive=True):
        """
        For each child of the current object that can only have a single
        parent, set its parent to be the current object.

        Usage:
        >>> a_block.create_many_to_one_relationship()
        >>> a_block.create_many_to_one_relationship(force=True)

        If the current object is a :class:`Block`, you want to run
        populate_RecordingChannel first, because this will create new objects
        that this method will link up.

        If force is True overwrite any existing relationships
        If recursive is True desecend into child objects and create
        relationships there
        """
        classname = self.__class__.__name__.lower()
        for child in self._single_children:
            if (hasattr(child, classname) and
                    getattr(child, classname) is None or force):
                setattr(child, classname, self)

        if recursive:
            for child in self.container_children:
                child.create_many_to_one_relationship(force=force,
                                                      recursive=True)

    def create_many_to_many_relationship(self, append=True, recursive=True):
        """
        For children of the current object that can have more than one parent
        of this type, put the current object in the parent list.

        If append is True add it to the list, otherwise overwrite the list.
        If recursive is True desecend into child objects and create
        relationships there
        """
        classname = self.__class__.__name__.lower() + 's'
        for child in self._multi_children:
            if not hasattr(child, classname):
                continue
            if append:
                target = getattr(child, classname)
                if not self in target:
                    target.append(self)
                continue
            setattr(child, classname, [self])

        if recursive:
            for child in self.container_children:
                child.create_many_to_many_relationship(append=append,
                                                       recursive=True)

    def create_relationship(self, force=False, append=True, recursive=True):
        """
        For each child of the current object that can only have a single
        parent, set its parent to be the current object.
        For children of the current object that can have more than one parent
        of this type, put the current object in the parent list.

        If the current object is a :class:`Block`, you want to run
        populate_RecordingChannel first, because this will create new objects
        that this method will link up.

        If force is True overwrite any existing relationships
        If append is True add it to the list, otherwise overwrite the list.
        If recursive is True desecend into child objects and create
        relationships there
        """
        self.create_many_to_one_relationship(force=force, recursive=False)
        self.create_many_to_many_relationship(append=append, recursive=False)
        if recursive:
            for child in self.container_children:
                child.create_relationship(force=force, append=append,
                                          recursive=True)

    def merge(self, other):
        """
        Merge the contents of another object into this one.

        Container children of the current object with the same name will be
        merged.  All other objects will be appended to the list of objects
        in this one.  Duplicate copies of the same object will be skipped.

        Annotations are merged such that only items not present in the current
        annotations are added.
        """
        # merge containers with the same name
        for container in (self._container_child_containers +
                          self._multi_child_containers):
            lookup = dict((obj.name, obj) for obj in getattr(self, container))
            ids = [id(obj) for obj in getattr(self, container)]
            for obj in getattr(other, container):
                if id(obj) in ids:
                    continue
                if obj.name in lookup:
                    lookup[obj.name].merge(obj)
                else:
                    lookup[obj.name] = obj
                    ids.append(id(obj))
                    getattr(self, container).append(obj)

        # for data objects, ignore the name and just add them
        for container in self._data_child_containers:
            objs = getattr(self, container)
            lookup = dict((obj.name, i) for i, obj in enumerate(objs))
            ids = [id(obj) for obj in objs]
            for obj in getattr(other, container):
                if id(obj) in ids:
                    continue
                if hasattr(obj, 'merge') and obj.name in lookup:
                    ind = lookup[obj.name]
                    try:
                        newobj = getattr(self, container)[ind].merge(obj)
                        getattr(self, container)[ind] = newobj
                    except NotImplementedError:
                        getattr(self, container).append(obj)
                        ids.append(id(obj))
                else:
                    lookup[obj.name] = obj
                    ids.append(id(obj))
                    getattr(self, container).append(obj)

        # use the BaseNeo merge as well
        super(Container, self).merge(other)

    def _repr_pretty_(self, pp, cycle):
        """
        Handle pretty-printing.
        """
        pp.text(self.__class__.__name__)
        pp.text(" with ")

        vals = []
        for container in self._child_containers:
            objs = getattr(self, container)
            if objs:
                vals.append('%s %s' % (len(objs), container))
        pp.text(', '.join(vals))

        if self._has_repr_pretty_attrs_():
            pp.breakable()
            self._repr_pretty_attrs_(pp, cycle)

        for container in self._repr_pretty_containers:
            pp.breakable()
            objs = getattr(self, container)
            pp.text("# %s (N=%s)" % (container, len(objs)))
            for (i, obj) in enumerate(objs):
                pp.breakable()
                pp.text("%s: " % i)
                with pp.indent(3):
                    pp.pretty(obj)
