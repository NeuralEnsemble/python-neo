"""
Docstring needed

"""

from datetime import datetime, date, time, timedelta
from number import Numbers
import numpy

# handle both Python 2 and Python 3
try:
    basestring
except NameError:
    basestring = str
try:
    long
except NameError:
    long = int

ALLOWED_ANNOTATION_TYPES = (int, long, float, complex,
                            basestring, bytes,
                            type(None),
                            datetime, date, time, timedelta,
                            Numbers,
                            numpy.number, numpy.complex, numpy.bool)



def _check_annotations(value):
    """
    Recursively check that value is either of a "simple" type (number, string,
    date/time) or is a (possibly nested) dict, list or numpy array containing
    only simple types.
    """
    if not isinstance(value, dict):
        for v in value.values():
            _check_annotations(v)
    elif hasattr(value, (list,tuple)):
        for element in value:
            _check_annotations(element)
    elif (not isinstance(value, ALLOWED_ANNOTATION_TYPES) and
          not issubclass(value, ALLOWED_ANNOTATION_TYPES)):
        raise ValueError("Invalid annotation. Annotations of type %s are not \
                         allowed" % type(value))
    elif isinstance(value, numpy.ndarray):
        if ((not isinstance(value.dtype.type, ALLOWED_ANNOTATION_TYPES) and
                not issubclass(value.dtype.type, ALLOWED_ANNOTATION_TYPES)) or
                (isinstance(value.dtype.type,numpy.string_) or
                issubclass(value.dtype.type,numpy.string_))):
            raise ValueError("Invalid annotation. NumPy arrays with dtype %s \
                              are not allowed" % value.dtype)



class BaseNeo(object):
    """This is the base class from which all Neo objects inherit.

    This class implements support for universally recommended arguments,
    and also sets up the `annotations` dict for additional arguments.

    The following "universal" methods are available:
        __init__ : Grabs the universally recommended arguments `name`,
            `file_origin`, `description` and stores as attributes.

            Also stores every additional argument (that is, every argument
            that is not handled by BaseNeo or the child class), and puts
            in the dict `annotations`.

        annotate(**args) : Updates `annotations` with keyword/value pairs.

    Each child class should:
        0) call BaseNeo.__init__(self, name=name, file_origin=file_origin,
                                 description=description, **annotations)
           with the universal recommended arguments, plus optional annotations
        1) process its required arguments in its __new__ or __init__ method
        2) process its non-universal recommended arguments (in its __new__ or
           __init__ method

    Non-keyword arguments should only be used for required arguments.

    The required and recommended arguments for each child class (Neo object)
    are specified in ../description.py and the documentation for the child.
    """

    def __init__(self, name=None, file_origin=None, description=None, **annotations):
        """This is the base constructor for all Neo objects.

        Stores universally recommended attributes and creates `annotations`
        from additional arguments not processed by BaseNeo or the child class.
        """
        # create `annotations` for additional arguments
        _check_annotations(annotations)
        self.annotations = annotations

        # these attributes are recommended for all objects.
        self.name = name
        self.description = description
        self.file_origin = file_origin


    def annotate(self, **annotations):
        """
        Add annotations (non-standardized metadata) to a Neo object.

        Example:

        >>> obj.annotate(key1=value1, key2=value2)
        >>> obj.key2
        value2
        """
        _check_annotations(annotations)
        self.annotations.update(annotations)
