"""
This module implements the ObjectList class, which is used to peform type checks
and handle relationships within the Neo Block-Segment-Data hierarchy.
"""

import sys

from neo.core.baseneo import BaseNeo


class ObjectList:
    """
    This class behaves like a list, but has additional functionality
    to handle relationships within Neo hierarchy, and perform type checks.
    """

    def __init__(self, allowed_contents, parent=None):
        # validate allowed_contents and normalize it to a tuple
        if isinstance(allowed_contents, type) and issubclass(allowed_contents, BaseNeo):
            self.allowed_contents = (allowed_contents,)
        else:
            for item in allowed_contents:
                if not issubclass(item, BaseNeo):
                    raise TypeError("Each item in allowed_contents must be a subclass of BaseNeo")
            self.allowed_contents = tuple(allowed_contents)
        self._items = []
        self.parent = parent

    def _handle_append(self, obj):
        if not (
            isinstance(obj, self.allowed_contents)
            or (  # also allow proxy objects of the correct type
                hasattr(obj, "proxy_for") and obj.proxy_for in self.allowed_contents
            )
        ):
            raise TypeError(f"Object is a {type(obj)}. It should be one of {self.allowed_contents}.")

        if self._contains(obj):
            raise ValueError("Cannot add this object because it is already contained within the list")

        # set the child-parent relationship
        if self.parent:
            relationship_name = self.parent.__class__.__name__.lower()
            if relationship_name == "group":
                raise Exception("Objects in groups should not link to the group as their parent")
            current_parent = getattr(obj, relationship_name)
            if current_parent != self.parent:
                # use weakref here? - see https://github.com/NeuralEnsemble/python-neo/issues/684
                setattr(obj, relationship_name, self.parent)

    def _contains(self, obj):
        if self._items is None:
            obj_ids = []
        else:
            obj_ids = [id(item) for item in self._items]
        return id(obj) in obj_ids

    def __str__(self):
        return str(self._items)

    def __repr__(self):
        return repr(self._items)

    def __add__(self, objects):
        # todo: decision: return a list, or a new DataObjectList?
        if isinstance(objects, ObjectList):
            return self._items + objects._items
        else:
            return self._items + objects

    def __radd__(self, objects):
        if isinstance(objects, ObjectList):
            return objects._items + self._items
        else:
            return objects + self._items

    def __contains__(self, key):
        return key in self._items

    def __iadd__(self, objects):
        for obj in objects:
            self._handle_append(obj)
        self._items.extend(objects)
        return self

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)

    def __setitem__(self, key, value):
        self._items[key] = value

    def append(self, obj):
        self._handle_append(obj)
        self._items.append(obj)

    def extend(self, objects):
        for obj in objects:
            self._handle_append(obj)
        self._items.extend(objects)

    def clear(self):
        self._items = []

    def count(self, value):
        return self._items.count(value)

    def index(self, value, start=0, stop=sys.maxsize):
        return self._items.index(value, start, stop)

    def insert(self, index, obj):
        self._handle_append(obj)
        self._items.insert(index, obj)

    def pop(self, index=-1):
        return self._items.pop(index)

    def remove(self, value):
        return self._items.remove(value)

    def reverse(self):
        raise self._items.reverse()

    def sort(self, *args, key=None, reverse=False):
        self._items.sort(*args, key=key, reverse=reverse)
