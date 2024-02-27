"""
Tools for use with neo tests.
"""

import hashlib
import os

import numpy as np
import quantities as pq

import neo
from neo.core import objectlist
from neo.core.baseneo import _reference_name, _container_name
from neo.core.basesignal import BaseSignal
from neo.core.container import Container
from neo.core.objectlist import ObjectList
from neo.core.spiketrainlist import SpikeTrainList
from neo.io.basefromrawio import proxyobjectlist, EventProxy, EpochProxy


def assert_arrays_equal(a, b, dtype=False):
    """
    Check if two arrays have the same shape and contents.

    If dtype is True (default=False), then also theck that they have the same
    dtype.
    """
    assert isinstance(a, np.ndarray), f"a is a {type(a)}"
    assert isinstance(b, np.ndarray), f"b is a {type(b)}"
    assert a.shape == b.shape, f"{a} != {b}"
    # assert a.dtype == b.dtype, "%s and %s not same dtype %s %s" % (a, b,
    #                                                               a.dtype,
    #                                                               b.dtype)
    try:
        assert (a.flatten() == b.flatten()).all(), f"{a} != {b}"
    except (AttributeError, ValueError):
        try:
            ar = np.array(a)
            br = np.array(b)
            assert (ar.flatten() == br.flatten()).all(), f"{ar} != {br}"
        except (AttributeError, ValueError):
            assert np.all(a.flatten() == b.flatten()), f"{a} != {b}"

    if dtype:
        assert a.dtype == b.dtype, f"{a} and {b} not same dtype {a.dtype} and {b.dtype}"


def assert_arrays_almost_equal(a, b, threshold, dtype=False):
    """
    Check if two arrays have the same shape and contents that differ
    by abs(a - b) <= threshold for all elements.

    If threshold is None, do an absolute comparison rather than a relative
    comparison.
    """
    if threshold is None:
        return assert_arrays_equal(a, b, dtype=dtype)

    assert isinstance(a, np.ndarray), f"a is a {type(a)}"
    assert isinstance(b, np.ndarray), f"b is a {type(b)}"
    assert a.shape == b.shape, f"{a} != {b}"
    # assert a.dtype == b.dtype, "%s and %b not same dtype %s %s" % (a, b,
    #                                                               a.dtype,
    #                                                               b.dtype)
    if a.dtype.kind in ["f", "c", "i"]:
        assert (abs(a - b) < threshold).all(), (
            f"abs({a} - {b})    max(|a - b|) = {(abs(a - b)).max()}    threshold:{threshold}" ""
        )

    if dtype:
        assert a.dtype == b.dtype, f"{a} and {b} not same dtype {a.dtype} and {b.dtype}"


def file_digest(filename):
    """
    Get the sha1 hash of the file with the given filename.
    """
    with open(filename, "rb") as fobj:
        return hashlib.sha1(fobj.read()).hexdigest()


def assert_file_contents_equal(a, b):
    """
    Assert that two files have the same size and hash.
    """

    def generate_error_message(a, b):
        """
        This creates the error message for the assertion error
        """
        size_a = os.stat(a).st_size
        size_b = os.stat(b).st_size
        if size_a == size_b:
            return "Files have the same size but different contents"
        else:
            return f"Files have different sizes: a:{size_a} b: {size_b}"

    assert file_digest(a) == file_digest(b), generate_error_message(a, b)


def assert_neo_object_is_compliant(ob, check_type=True):
    """
    Test neo compliance of one object and sub objects
    (one_to_many_relation only):
      * check types and/or presence of necessary and recommended attribute.
      * If attribute is Quantities or numpy.ndarray it also check ndim.
      * If attribute is numpy.ndarray also check dtype.kind.

    check_type=True by default can be set to false for testing ProxyObject
    """
    if check_type:
        assert type(ob) in objectlist, f"{type(ob)} is not a neo object"
    classname = ob.__class__.__name__

    # test presence of necessary attributes
    for ioattr in ob._necessary_attrs:
        attrname, attrtype = ioattr[0], ioattr[1]
        # ~ if attrname != '':
        if not hasattr(ob, "_quantity_attr"):
            assert hasattr(ob, attrname), f"{classname} neo obect does not have {attrname}"

    # test attributes types
    for ioattr in ob._all_attrs:
        attrname, attrtype = ioattr[0], ioattr[1]

        if (
            hasattr(ob, "_quantity_attr")
            and ob._quantity_attr == attrname
            and (attrtype == pq.Quantity or attrtype == np.ndarray)
        ):
            # object inherits from Quantity (AnalogSignal, SpikeTrain, ...)
            ndim = ioattr[2]
            assert ob.ndim == ndim, f"{classname} dimension is {ob.ndim} should be {ndim}"
            if attrtype == np.ndarray:
                dtp = ioattr[3]
                assert ob.dtype.kind == dtp.kind, f"{classname} dtype.kind is {ob.dtype.kind} should be {dtp.kind}" ""

        elif hasattr(ob, attrname):
            if getattr(ob, attrname) is not None:
                obattr = getattr(ob, attrname)
                assert issubclass(type(obattr), attrtype), (
                    f"{attrname} in {classname} is {type(obattr)} should be {attrtype}" ""
                )
                if attrtype == pq.Quantity or attrtype == np.ndarray:
                    ndim = ioattr[2]
                    assert obattr.ndim == ndim, f"{classname}.{attrname} dimension is {obattr.ndim} should be {ndim}" ""
                if attrtype == np.ndarray:
                    dtp = ioattr[3]
                    assert obattr.dtype.kind == dtp.kind, (
                        f"{classname}.{attrname} dtype.kind is {obattr.dtype.kind} should be {dtp.kind}" ""
                    )

    # test bijectivity : parents and children
    if classname != "Group":  # objects in a Group do not keep a reference to the group.
        for container in getattr(ob, "_single_child_containers", []):
            for i, child in enumerate(getattr(ob, container, [])):
                assert hasattr(child, _reference_name(classname)), (
                    f"{container} should have {_reference_name(classname)} attribute (2 way relationship)" ""
                )
                if hasattr(child, _reference_name(classname)):
                    parent = getattr(child, _reference_name(classname))
                    assert parent == ob, (
                        f"{container}.{_reference_name(classname)} {i} is not symmetric with {classname}.{container}" ""
                    )

    # recursive on one to many rel
    for i, child in enumerate(getattr(ob, "children", [])):
        try:
            assert_neo_object_is_compliant(child)
        # intercept exceptions and add more information
        except BaseException as exc:
            exc.args += (f"from {child.__class__.__name__} {i} of {classname}",)
            raise


def types_match(ob1, ob2):
    if type(ob1) == type(ob2):
        return True
    elif isinstance(ob1, ObjectList):
        return isinstance(ob2, (list, ObjectList))
    elif isinstance(ob2, ObjectList):
        return isinstance(ob1, (list, ObjectList))
    else:
        return False


def assert_same_sub_schema(ob1, ob2, equal_almost=True, threshold=1e-10, exclude=None):
    """
    Test if ob1 and ob2 has the same sub schema.
    Explore all parent/child relationships.
    Many_to_many_relationship is not tested
    because of infinite recursive loops.

    Arguments:
        equal_almost: if False do a strict arrays_equal if
                      True do arrays_almost_equal
        exclude: a list of attributes and annotations to ignore in
                 the comparison

    """
    if isinstance(ob1, SpikeTrainList) and isinstance(ob2, list):
        # for debugging occasional test failure
        raise Exception(
            f"items={str(ob1._items)}\nspike_time_array={str(ob1._spike_time_array)}\nlist length: {len(ob2)}"
        )
    errmsg = f"type({type(ob1)}) != type({type(ob2)})"
    assert types_match(ob1, ob2), errmsg
    classname = ob1.__class__.__name__

    if exclude is None:
        exclude = []

    if isinstance(ob1, (list, ObjectList)):
        assert len(ob1) == len(ob2), f"lens {len(ob1)} and {len(ob2)} not equal for {ob1} and {ob2}" ""
        for i, (sub1, sub2) in enumerate(zip(ob1, ob2)):
            try:
                assert_same_sub_schema(sub1, sub2, equal_almost=equal_almost, threshold=threshold, exclude=exclude)
            # intercept exceptions and add more information
            except BaseException as exc:
                exc.args += f"{classname}[{i}]"
                raise
        return

    # test parent/child relationship
    for container in getattr(ob1, "_single_child_containers", []):
        if container in exclude:
            continue
        if not hasattr(ob1, container):
            assert not hasattr(ob2, container), f"{classname} 2 does have {container} but not {classname} 1" ""
            continue
        else:
            assert hasattr(ob2, container), f"{classname} 1 has {container} but not {classname} 2"

        sub1 = getattr(ob1, container)
        sub2 = getattr(ob2, container)

        assert len(sub1) == len(sub2), (
            f"these two {classname} do not have the same {container} number: {len(sub1)} and {len(sub2)}" ""
        )
        for i in range(len(getattr(ob1, container))):
            # previously lacking parameter
            try:
                assert_same_sub_schema(
                    sub1[i], sub2[i], equal_almost=equal_almost, threshold=threshold, exclude=exclude
                )
            # intercept exceptions and add more information
            except BaseException as exc:
                exc.args += f"from {container}[{i}] of {classname}"
                raise

    assert_same_attributes(ob1, ob2, equal_almost=equal_almost, threshold=threshold, exclude=exclude)


def assert_same_attributes(ob1, ob2, equal_almost=True, threshold=1e-10, exclude=None):
    """
    Test if ob1 and ob2 has the same attributes.

    Arguments:
        equal_almost: if False do a strict arrays_equal if
                      True do arrays_almost_equal
        exclude: a list of attributes and annotations to ignore in
                 the comparison

    """
    classname = ob1.__class__.__name__

    if exclude is None:
        exclude = []

    if not equal_almost:
        threshold = None
        dtype = True
    else:
        dtype = False

    for ioattr in ob1._all_attrs:
        if ioattr[0] in exclude:
            continue
        attrname, attrtype = ioattr[0], ioattr[1]
        # ~ if attrname =='':
        if hasattr(ob1, "_quantity_attr") and ob1._quantity_attr == attrname:
            # object is inherited from Quantity (AnalogSignal, SpikeTrain, ...)
            try:
                assert_arrays_almost_equal(ob1.magnitude, ob2.magnitude, threshold=threshold, dtype=dtype)
            # intercept exceptions and add more information
            except BaseException as exc:
                exc.args += (f"from {classname} {attrname}",)
                raise
            assert ob1.dimensionality.string == ob2.dimensionality.string, (
                f"Units of {classname} {attrname} are not the same: {ob1.dimensionality.string} and {ob2.dimensionality.string}"
                ""
            )
            continue

        if not hasattr(ob1, attrname):
            assert not hasattr(ob2, attrname), f"{classname} 2 does have {attrname} but not {classname} 1" ""
            continue
        else:
            assert hasattr(ob2, attrname), f"%{classname} 1 has {attrname} but not {classname} 2" ""

        if getattr(ob1, attrname) is None:
            assert getattr(ob2, attrname) is None, (
                f"In {classname}.{attrname} {getattr(ob1, attrname)} and {getattr(ob2, attrname)} differed" ""
            )
            continue
        if getattr(ob2, attrname) is None:
            assert getattr(ob1, attrname) is None, (
                f"In {classname}.{attrname} {getattr(ob1, attrname)} and { getattr(ob2, attrname)} differed" ""
            )
            continue

        if attrtype == pq.Quantity:
            # Compare magnitudes
            mag1 = getattr(ob1, attrname).magnitude
            mag2 = getattr(ob2, attrname).magnitude
            # print "2. ob1(%s) %s:%s\n   ob2(%s) %s:%s" % \
            # (ob1,attrname,mag1,ob2,attrname,mag2)
            try:
                assert_arrays_almost_equal(mag1, mag2, threshold=threshold, dtype=dtype)
            # intercept exceptions and add more information
            except BaseException as exc:
                exc.args += (f"from {attrname} of {classname}",)
                raise
            # Compare dimensionalities
            dim1 = getattr(ob1, attrname).dimensionality.simplified
            dim2 = getattr(ob2, attrname).dimensionality.simplified
            dimstr1 = getattr(ob1, attrname).dimensionality.string
            dimstr2 = getattr(ob2, attrname).dimensionality.string
            assert dim1 == dim2, f"Attribute {attrname} of {classname} are not the same: {dimstr1} != {dimstr2}" ""
        elif attrtype == np.ndarray:
            try:
                assert_arrays_almost_equal(
                    getattr(ob1, attrname), getattr(ob2, attrname), threshold=threshold, dtype=dtype
                )
            # intercept exceptions and add more information
            except BaseException as exc:
                exc.args += (f"from {attrname} of {classname}",)
                raise

        elif isinstance(attrtype, tuple):
            attr1 = getattr(ob1, attrname)
            attr2 = getattr(ob2, attrname)
            assert attr1.__class__.__name__ in attrtype
            assert attr2.__class__.__name__ in attrtype
            if isinstance(attr1, BaseSignal):
                assert isinstance(attr2, BaseSignal)
                # Compare magnitudes
                assert_arrays_almost_equal(attr1.magnitude, attr2.magnitude, threshold=threshold, dtype=dtype)
                # Compare dimensionalities
                dim1 = attr1.dimensionality
                dim2 = attr2.dimensionality
                errmsg = f"Attribute {attrname} of {classname} are not the same: {dim1.string} != {dim2.string}"
                assert dim1.simplified == dim2.simplified, errmsg
                assert attr1.name == attr2.name
                # todo: check annotations
        else:
            # ~ print 'yep', getattr(ob1, attrname),  getattr(ob2, attrname)
            assert getattr(ob1, attrname) == getattr(ob2, attrname), (
                f"Attribute {classname}.{attrname} are not the same { type(getattr(ob1, attrname))} {getattr(ob1, attrname)} {type(getattr(ob2, attrname))} {getattr(ob2, attrname)}"
                ""
            )


def assert_same_annotations(ob1, ob2, equal_almost=True, threshold=1e-10, exclude=None):
    """
    Test if ob1 and ob2 has the same annotations.

    Arguments:
        equal_almost: if False do a strict arrays_equal if
                      True do arrays_almost_equal
        exclude: a list of attributes and annotations to ignore in
                 the comparison

    """
    if exclude is None:
        exclude = []

    if not equal_almost:
        threshold = None
        dtype = False
    else:
        dtype = True

    for key in ob2.annotations:
        if key in exclude:
            continue
        assert key in ob1.annotations

    for key, value1 in ob1.annotations.items():
        if key in exclude:
            continue
        assert key in ob2.annotations
        value2 = ob2.annotations[key]
        if isinstance(value1, np.ndarray):
            assert isinstance(value2, np.ndarray)
            assert_arrays_almost_equal(value1, value2, threshold=threshold, dtype=False)
        else:
            assert value1 == value2


def assert_same_array_annotations(ob1, ob2, equal_almost=True, threshold=1e-10, exclude=None):
    """
    Test if ob1 and ob2 has the same annotations.

    Arguments:
        equal_almost: if False do a strict arrays_equal if
                      True do arrays_almost_equal
        exclude: a list of attributes and annotations to ignore in
                 the comparison

    """
    if exclude is None:
        exclude = []

    if not equal_almost:
        threshold = None
        dtype = False
    else:
        dtype = True

    for key in ob2.array_annotations:
        if key in exclude:
            continue
        assert key in ob1.array_annotations

    for key, value in ob1.array_annotations.items():
        if key in exclude:
            continue
        assert key in ob2.array_annotations
        try:
            assert_arrays_equal(value, ob2.array_annotations[key])
        except ValueError:
            assert_arrays_almost_equal(ob1, ob2, threshold=threshold, dtype=False)


def assert_sub_schema_is_lazy_loaded(ob):
    """
    This is util for testing lazy load. All data object must be in proxyobjectlist.
    """
    classname = ob.__class__.__name__

    if isinstance(ob, Container):
        for container in getattr(ob, "_single_child_containers", []):
            if not hasattr(ob, container):
                continue
            sub = getattr(ob, container)
            for i, child in enumerate(sub):
                try:
                    assert_sub_schema_is_lazy_loaded(child)
                # intercept exceptions and add more information
                except BaseException as exc:
                    exc.args += (f"from {container} {i} of {classname}",)
                    raise
    else:
        assert ob.__class__ in proxyobjectlist, f"Data object must lazy {classname}"
        loaded_ob = ob.load()
        assert_neo_object_is_compliant(loaded_ob)
        assert_same_annotations(ob, loaded_ob)
        exclude = []
        if isinstance(ob, EventProxy):
            exclude = ["labels"]
        elif isinstance(ob, EpochProxy):
            exclude = ["labels", "durations"]
        else:
            exclude = []
        assert_same_array_annotations(ob, loaded_ob, exclude=exclude)


def assert_objects_equivalent(obj1, obj2):
    """
    Compares two NEO objects by looping over the attributes and annotations
    and asserting their hashes. No relationships involved.
    """

    def assert_attr(obj1, obj2, attr_name):
        """
        Assert a single attribute and annotation are the same
        """
        assert hasattr(obj1, attr_name)
        attr1 = hashlib.md5(getattr(obj1, attr_name)).hexdigest()
        assert hasattr(obj2, attr_name)
        attr2 = hashlib.md5(getattr(obj2, attr_name)).hexdigest()
        assert attr1 == attr2, f"Attribute {attr_name} for class {obj1.__class__.__name__} is not equal." ""

    obj_type = obj1.__class__.__name__
    assert obj_type == obj2.__class__.__name__
    for ioattr in obj1._necessary_attrs:
        assert_attr(obj1, obj2, ioattr[0])
    for ioattr in obj1._recommended_attrs:
        if hasattr(obj1, ioattr[0]) or hasattr(obj2, ioattr[0]):
            assert_attr(obj1, obj2, ioattr[0])
    if hasattr(obj1, "annotations"):
        assert hasattr(obj2, "annotations")
        for key, value in obj1.annotations:
            assert hasattr(obj2.annotations, key)
            assert obj2.annotations[key] == value


def assert_children_empty(obj, parent):
    """
    Check that the children of a neo object are empty.  Used
    to check the cascade is implemented properly
    """
    classname = obj.__class__.__name__
    errmsg = f"""{parent.__name__} reader with cascade=False should return
        empty children"""
    if hasattr(obj, "children"):
        assert not obj.children, errmsg
