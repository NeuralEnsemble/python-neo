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
                assert issubclass(item, BaseNeo)
            self.allowed_contents = tuple(allowed_contents)
        self.contents = []
        self.parent = parent

    def _handle_append(self, obj):
        if not (
            isinstance(obj, self.allowed_contents)
            or (  # also allow proxy objects of the correct type
                hasattr(obj, "proxy_for") and obj.proxy_for in self.allowed_contents
            )
        ):
            raise TypeError(f"Object is a {type(obj)}. It should be one of {self.allowed_contents}.")
        # set the child-parent relationship
        if self.parent:
            relationship_name = self.parent.__class__.__name__.lower()
            if relationship_name == "group":
                raise Exception("Objects in groups should not link to the group as their parent")
            current_parent = getattr(obj, relationship_name)
            if current_parent != self.parent:
                # use weakref here? - see https://github.com/NeuralEnsemble/python-neo/issues/684
                setattr(obj, relationship_name, self.parent)

    def __str__(self):
        return str(self.contents)

    def __repr__(self):
        return repr(self.contents)

    def __add__(self, objects):
        # todo: decision: return a list, or a new DataObjectList?
        if isinstance(objects, ObjectList):
            return self.contents + objects.contents
        else:
            return self.contents + objects

    def __radd__(self, objects):
        if isinstance(objects, ObjectList):
            return objects.contents + self.contents
        else:
            return objects + self.contents

    def __contains__(self, key):
        return key in self.contents

    def __iadd__(self, objects):
        for obj in objects:
            self._handle_append(obj)
        self.contents.extend(objects)
        return self

    def __iter__(self):
        return iter(self.contents)

    def __getitem__(self, i):
        return self.contents[i]

    def __len__(self):
        return len(self.contents)

    def __setitem__(self, key, value):
        self.contents[key] = value

    def append(self, obj):
        self._handle_append(obj)
        self.contents.append(obj)

    def extend(self, objects):
        for obj in objects:
            self._handle_append(obj)
        self.contents.extend(objects)

    def clear(self):
        self.contents = []

    def count(self, value):
        return self.contents.count(value)

    def index(self, value, start=0, stop=sys.maxsize):
        return self.contents.index(value, start, stop)

    def insert(self, index, obj):
        self._handle_append(obj)
        self.contents.insert(index, obj)

    def pop(self, index=-1):
        return self.contents.pop(index)

    def remove(self, value):
        return self.contents.remove(value)

    def reverse(self):
        raise self.contents.reverse()

    def sort(self, *args, key=None, reverse=False):
        self.contents.sort(*args, key=key, reverse=reverse)
