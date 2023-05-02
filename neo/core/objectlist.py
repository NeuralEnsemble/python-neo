"""


"""

from neo.core.baseneo import BaseNeo


class ObjectList:
    """
    handle relationships within Neo hierarchy
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
            current_parent = getattr(obj, relationship_name)
            if current_parent != self.parent:
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

    def __iter__(self):
        return iter(self.contents)

    def __getitem__(self, i):
        return self.contents[i]

    def __len__(self):
        return len(self.contents)

    def __reversed__(self):
        raise NotImplementedError

    def __setitem__(self, i):
        raise NotImplementedError

    def append(self, obj):
        self._handle_append(obj)
        self.contents.append(obj)

    def extend(self, objects):
        for obj in objects:
            self._handle_append(obj)
        self.contents.extend(objects)

    def clear(self):
        raise NotImplementedError

    def copy(self):
         raise NotImplementedError

    def count(self):
        raise NotImplementedError

    def index(self):
        raise NotImplementedError

    def insert(self):
        raise NotImplementedError

    def pop(self):
        raise NotImplementedError

    def remove(self):
        raise NotImplementedError

    def reverse(self):
        raise NotImplementedError

    def sort(self):
        raise NotImplementedError
