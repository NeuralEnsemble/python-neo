"""
This module implements generic container base class that all neo container
object inherit from.  It provides shared methods for all container types.

:class:`Container` is derived from :class:`BaseNeo`
"""

from copy import deepcopy

from neo.core import filters
from neo.core.baseneo import BaseNeo, _reference_name, _container_name
from neo.core.objectlist import ObjectList
from neo.core.spiketrain import SpikeTrain
from neo.core.spiketrainlist import SpikeTrainList


def unique_objs(objs):
    """
    Return a list of objects in the list objs where all objects are unique
    using the "is" test.
    """
    seen = set()
    return [obj for obj in objs if id(obj) not in seen and not seen.add(id(obj))]


def filterdata(data, targdict=None, objects=None, **kwargs):
    """
    Return a list of the objects in data matching *any* of the search terms
    in either their attributes or annotations.  Search terms can be
    provided as keyword arguments or a dictionary, either as a positional
    argument after data or to the argument targdict.
    A key of a provided dictionary is the name of the requested annotation
    and the value is a FilterCondition object.
    E.g.: Equal(x), LessThan(x), InRange(x, y).

    targdict can also
    be a list of dictionaries, in which case the filters are applied
    sequentially.

    A list of dictionaries is handled as follows: [ { or } and { or } ]
    If targdict and kwargs are both supplied, the
    targdict filters are applied first, followed by the kwarg filters.
    A targdict of None or {} corresponds to no filters applied, therefore
    returning all child objects. Default targdict is None.
    """

    # if objects are specified, get the classes
    if objects:
        if hasattr(objects, "lower") or isinstance(objects, type):
            objects = [objects]
    elif objects is not None:
        return []

    # handle cases with targdict
    if targdict is None:
        targdict = kwargs
    elif not kwargs:
        pass
    elif hasattr(targdict, "keys"):
        targdict = [targdict, kwargs]
    else:
        targdict += [kwargs]

    if not targdict:
        results = data

    # if multiple dicts are provided, apply each filter sequentially
    elif not hasattr(targdict, "keys"):
        # for performance reasons, only do the object filtering on the first
        # iteration
        results = filterdata(data, targdict=targdict[0], objects=objects)
        for targ in targdict[1:]:
            results = filterdata(results, targdict=targ)
        return results
    else:
        # do the actual filtering
        results = []
        for obj in data:
            for key, value in sorted(targdict.items()):
                if hasattr(obj, key) and getattr(obj, key) == value:
                    results.append(obj)
                    break
                if isinstance(value, filters.FilterCondition) and key in obj.annotations:
                    if value.evaluate(obj.annotations[key]):
                        results.append(obj)
                        break
                if key in obj.annotations and obj.annotations[key] == value:
                    results.append(obj)
                    break

    # remove duplicates from results
    results = list({id(res): res for res in results}.values())

    # keep only objects of the correct classes
    if objects:
        results = [result for result in results if result.__class__ in objects or result.__class__.__name__ in objects]

    if results and all(isinstance(obj, SpikeTrain) for obj in results):
        return SpikeTrainList(results)
    else:
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
                                   of this object. An instance attribute
                                   named class.__name__.lower()+'s' will be
                                   automatically defined to hold this child and
                                   will be initialized to an empty list.
        :_data_child_objects: Neo data objects that can be children
                              of this object. An instance attribute named
                              class.__name__.lower()+'s' will be automatically
                              defined to hold this child and will be
                              initialized to an empty list.
        :_repr_pretty_containers: The names of containers attributes printed
                                  when pretty-printing using IPython.

    The following helper properties are available
    (in  addition to those of BaseNeo):
        :_child_objects: All neo objects that can be children of this object.
                                :_container_child_objects: +
                                :_data_child_objects:
        :_container_child_containers: The names of the container attributes
                                      used to store :_container_child_objects:
        :_data_child_containers: The names of the container attributes used
                                 to store :_data_child_objects:
        :_single_child_containers: The names of the container attributes used
                                   to store :_single_child_objects:
        :_child_containers: All child container attributes. Same as :_single_child_containers:
        :_single_children: All objects that are children of the current object
                           where the child can only have one parent of
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

        :create_relationship(**args): For each child of the current
                                      object, set its parent
                                      to be the current object.

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

    # Child objects that are a container
    _container_child_objects = ()
    # Child objects that have data
    _data_child_objects = ()
    # Containers that are listed when pretty-printing
    _repr_pretty_containers = ()

    def __init__(self, name=None, description=None, file_origin=None, **annotations):
        """
        Initialize a new :class:`Container` instance.
        """
        super().__init__(name=name, description=description, file_origin=file_origin, **annotations)

    def _get_object_list(self, name):
        """
        Return the container's ObjectList with the given (private) attribute name

        Example:
        >>> segment._get_object_list("_analogsignals")
        """
        return getattr(self, name)

    def _set_object_list(self, name, value):
        """
        Set the contents of the container's ObjectList with the given (private) attribute name

        Example:
        >>> segment._set_object_list("_analogsignals", [sig1, sig2])
        """
        if isinstance(value, list):
            object_list = getattr(self, name)
            object_list.clear()
            object_list.extend(value)
        elif isinstance(value, ObjectList):  # from __iadd__
            setattr(self, name, value)
        else:
            TypeError("value must be a list or an ObjectList")

    @property
    def _child_objects(self):
        """
        Return the names of the classes that can be children of this container.
        """
        return self._container_child_objects + self._data_child_objects

    @property
    def _container_child_containers(self):
        """
        Containers for child objects that are a container and
        have a single parent.
        """
        return tuple([_container_name(child) for child in self._container_child_objects])

    @property
    def _data_child_containers(self):
        """
        Containers for child objects that have data and have a single parent.
        """
        # the following construction removes the duplicate 'regionsofinterest'
        # while preserving the child order (which `set()` would not do)
        # I don't know if preserving the order is important, but I'm playing it safe
        return tuple({_container_name(child): None for child in self._data_child_objects}.keys())

    @property
    def _child_containers(self):
        """
        Containers for child objects with a single parent.
        """
        return tuple({_container_name(child): None for child in self._child_objects}.keys())

    @property
    def _single_children(self):
        """
        All child objects that can only have single parents.
        """
        childs = [list(getattr(self, attr)) for attr in self._child_containers]
        return tuple(sum(childs, []))

    @property
    def data_children(self):
        """
        All data child objects stored in the current object.
        Not recursive.
        """
        childs = [list(getattr(self, attr)) for attr in self._data_child_containers]
        return tuple(sum(childs, []))

    @property
    def container_children(self):
        """
        All container child objects stored in the current object.
        Not recursive.
        """
        childs = [list(getattr(self, attr)) for attr in self._container_child_containers]
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
        childs = [list(child.data_children_recur) for child in self.container_children]
        return self.data_children + tuple(sum(childs, []))

    @property
    def container_children_recur(self):
        """
        All container child objects stored in the current object,
        obtained recursively.
        """
        childs = [list(child.container_children_recur) for child in self.container_children]
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
        return {name: len(getattr(self, name)) for name in self._child_containers}

    @property
    def _container_lookup(self):
        return {
            cls_name: getattr(self, container_name)
            for cls_name, container_name in zip(self._child_objects, self._child_containers)
        }

    def _get_container(self, cls):
        if hasattr(cls, "proxy_for"):
            cls = cls.proxy_for
        return self._container_lookup[cls.__name__]

    def add(self, *objects):
        """Add a new Neo object to the Container"""
        for obj in objects:
            if obj.__class__.__name__ in self._child_objects or (
                hasattr(obj, "proxy_for") and obj.proxy_for.__name__ in self._child_objects
            ):
                container = self._get_container(obj.__class__)
                container.append(obj)
            else:
                raise TypeError(
                    f"Cannot add object of type {obj.__class__.__name__} "
                    f"to a {self.__class__.__name__}, can only add objects of the "
                    f"following types: {self._child_objects}"
                )

    def filter(self, targdict=None, data=True, container=False, recursive=True, objects=None, **kwargs):
        """
        Return a list of child objects matching *any* of the search terms
        in either their attributes or annotations.  Search terms can be
        provided as keyword arguments or a dictionary, either as a positional
        argument after data or to the argument targdict.
        A key of a provided dictionary is the name of the requested annotation
        and the value is a FilterCondition object.
        E.g.: equal(x), less_than(x), InRange(x, y).

        targdict can also
        be a list of dictionaries, in which case the filters are applied
        sequentially.

        A list of dictionaries is handled as follows: [ { or } and { or } ]
        If targdict and kwargs are both supplied, the
        targdict filters are applied first, followed by the kwarg filters.
        A targdict of None or {} corresponds to no filters applied, therefore
        returning all child objects. Default targdict is None.

        If data is True (default), include data objects.
        If container is True (default False), include container objects.
        If recursive is True (default), descend into child containers for
        objects.

        objects (optional) should be the name of a Neo object type,
        a neo object class, or a list of one or both of these.  If specified,
        only these objects will be returned. If not specified any type of
        object is  returned. Default is None.
        Note that if recursive is True, containers not in objects will still
        be descended into. This overrides data and container.


        Examples::

            >>> obj.filter(name="Vm")
            >>> obj.filter(objects=neo.SpikeTrain)
            >>> obj.filter(targdict={'myannotation':3})
            >>> obj.filter(name=neo.core.filters.Equal(5))
            >>> obj.filter({'name': neo.core.filters.LessThan(5)})
        """

        if isinstance(targdict, str):
            raise TypeError("filtering is based on key-value pairs." " Only a single string was provided.")

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

        filtered = filterdata(children, objects=objects, targdict=targdict, **kwargs)
        if objects == SpikeTrain:
            return SpikeTrainList(items=filtered)
        else:
            return filtered

    def list_children_by_class(self, cls):
        """
        List all children of a particular class recursively.

        You can either provide a class object, a class name,
        or the name of the container storing the class.
        """
        if not hasattr(cls, "lower"):
            cls = cls.__name__
        container_name = _container_name(cls)
        objs = list(getattr(self, container_name, []))
        for child in self.container_children_recur:
            objs.extend(getattr(child, container_name, []))
        return objs

    def check_relationships(self, recursive=True):
        """
        Check that the expected child-parent relationships exist.
        """
        parent_name = _reference_name(self.__class__.__name__)
        for child in self._single_children:
            if hasattr(child, "proxy_for"):
                container = getattr(self, _container_name(child.proxy_for.__name__))
            else:
                container = getattr(self, _container_name(child.__class__.__name__))
            if container.parent is not None:
                if getattr(child, parent_name, None) is not self:
                    raise AttributeError("Child should have its parent as an attribute")
        if recursive:
            for child in self.container_children:
                child.check_relationships(recursive=True)

    def create_relationship(self, force=False, recursive=True):
        """
        For each child of the current object that can only have a single
        parent, set its parent to be the current object.
        For children of the current object, put the current object in the parent list.

        If force is True overwrite any existing relationships
        If recursive is True descend into child objects and create
        relationships there
        """
        parent_name = _reference_name(self.__class__.__name__)
        for child in self._single_children:
            if hasattr(child, parent_name) and getattr(child, parent_name) is None or force:
                setattr(child, parent_name, self)
        if recursive:
            for child in self.container_children:
                child.create_relationship(force=force, recursive=True)

    def __deepcopy__(self, memo):
        """
        Creates a deep copy of the container.
        All contained objects will also be deep copied and relationships
        between all objects will be identical to the original relationships.
        Attributes and annotations of the container are deep copied as well.

        :param memo: (dict) Objects that have been deep copied already
        :return: (Container) Deep copy of input Container
        """
        cls = self.__class__
        necessary_attrs = {}
        for k in self._necessary_attrs:
            necessary_attrs[k[0]] = getattr(self, k[0], None)
        new_container = cls(**necessary_attrs)
        new_container.__dict__.update(self.__dict__)
        memo[id(self)] = new_container
        for k, v in self.__dict__.items():
            try:
                setattr(new_container, k, deepcopy(v, memo))
            except TypeError:
                setattr(new_container, k, v)

        new_container.create_relationship()

        return new_container

    def merge(self, other):
        """
        Merge the contents of another object into this one.

        Container children of the current object with the same name will be
        merged.  All other objects will be appended to the list of objects
        in this one.  Duplicate copies of the same object will be skipped.

        Annotations are merged such that only items not present in the current
        annotations are added.

        Note that the other object will be linked inconsistently to other Neo objects
        after the merge operation and should not be used further.
        """
        # merge containers with the same name
        for container in self._container_child_containers:
            lookup = {obj.name: obj for obj in getattr(self, container)}
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
            lookup = {obj.name: i for i, obj in enumerate(objs)}
            ids = [id(obj) for obj in objs]
            for obj in getattr(other, container):
                if id(obj) in ids:
                    pass
                elif hasattr(obj, "merge") and obj.name is not None and obj.name in lookup:
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
                obj.set_parent(self)

        # use the BaseNeo merge as well
        super().merge(other)

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
                vals.append(f"{objs} {container}")
        pp.text(", ".join(vals))

        if self._has_repr_pretty_attrs_():
            pp.breakable()
            self._repr_pretty_attrs_(pp, cycle)

        for container in self._repr_pretty_containers:
            pp.breakable()
            objs = getattr(self, container)
            pp.text(f"# {container} (N={objs})")
            for i, obj in enumerate(objs):
                pp.breakable()
                pp.text(f"{i}: ")
                with pp.indent(3):
                    pp.pretty(obj)
