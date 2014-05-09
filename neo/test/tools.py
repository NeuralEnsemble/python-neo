# -*- coding: utf-8 -*-
'''
Tools for use with neo tests.
'''

import hashlib
import os

import numpy as np
import quantities as pq

import neo
from neo.core import objectlist


def assert_arrays_equal(a, b, dtype=False):
    '''
    Check if two arrays have the same shape and contents.

    If dtype is True (default=False), then also theck that they have the same
    dtype.
    '''
    assert isinstance(a, np.ndarray), "a is a %s" % type(a)
    assert isinstance(b, np.ndarray), "b is a %s" % type(b)
    assert a.shape == b.shape, "%s != %s" % (a, b)
    #assert a.dtype == b.dtype, "%s and %s not same dtype %s %s" % (a, b,
    #                                                               a.dtype,
    #                                                               b.dtype)
    try:
        assert (a.flatten() == b.flatten()).all(), "%s != %s" % (a, b)
    except (AttributeError, ValueError):
        try:
            ar = np.array(a)
            br = np.array(b)
            assert (ar.flatten() == br.flatten()).all(), "%s != %s" % (ar, br)
        except (AttributeError, ValueError):
            assert np.all(a.flatten() == b.flatten()), "%s != %s" % (a, b)

    if dtype:
        assert a.dtype == b.dtype, \
            "%s and %s not same dtype %s and %s" % (a, b, a.dtype, b.dtype)


def assert_arrays_almost_equal(a, b, threshold, dtype=False):
    '''
    Check if two arrays have the same shape and contents that differ
    by abs(a - b) <= threshold for all elements.

    If threshold is None, do an absolute comparison rather than a relative
    comparison.
    '''
    if threshold is None:
        return assert_arrays_equal(a, b, dtype=dtype)

    assert isinstance(a, np.ndarray), "a is a %s" % type(a)
    assert isinstance(b, np.ndarray), "b is a %s" % type(b)
    assert a.shape == b.shape, "%s != %s" % (a, b)
    #assert a.dtype == b.dtype, "%s and %b not same dtype %s %s" % (a, b,
    #                                                               a.dtype,
    #                                                               b.dtype)
    if a.dtype.kind in ['f', 'c', 'i']:
        assert (abs(a - b) < threshold).all(), \
            "abs(%s - %s)    max(|a - b|) = %s    threshold:%s" % \
            (a, b, (abs(a - b)).max(), threshold)

    if dtype:
        assert a.dtype == b.dtype, \
            "%s and %s not same dtype %s and %s" % (a, b, a.dtype, b.dtype)


def file_digest(filename):
    '''
    Get the sha1 hash of the file with the given filename.
    '''
    with open(filename, 'rb') as fobj:
        return hashlib.sha1(fobj.read()).hexdigest()


def assert_file_contents_equal(a, b):
    '''
    Assert that two files have the same size and hash.
    '''
    def generate_error_message(a, b):
        '''
        This creates the error message for the assertion error
        '''
        size_a = os.stat(a).st_size
        size_b = os.stat(b).st_size
        if size_a == size_b:
            return "Files have the same size but different contents"
        else:
            return "Files have different sizes: a:%d b: %d" % (size_a, size_b)

    assert file_digest(a) == file_digest(b), generate_error_message(a, b)


def assert_neo_object_is_compliant(ob):
    '''
    Test neo compliance of one object and sub objects
    (one_to_many_relation only):
      * check types and/or presence of necessary and recommended attribute.
      * If attribute is Quantities or numpy.ndarray it also check ndim.
      * If attribute is numpy.ndarray also check dtype.kind.
    '''
    assert type(ob) in objectlist, \
        '%s is not a neo object' % (type(ob))
    classname = ob.__class__.__name__

    # test presence of necessary attributes
    for ioattr in ob._necessary_attrs:
        attrname, attrtype = ioattr[0], ioattr[1]
        #~ if attrname != '':
        if not hasattr(ob, '_quantity_attr'):
            assert hasattr(ob, attrname), '%s neo obect does not have %s' % \
                (classname, attrname)

    # test attributes types
    for ioattr in ob._all_attrs:
        attrname, attrtype = ioattr[0], ioattr[1]

        if (hasattr(ob, '_quantity_attr') and
                ob._quantity_attr == attrname and
                (attrtype == pq.Quantity or attrtype == np.ndarray)):
            # object inherits from Quantity (AnalogSIgnal, SpikeTrain, ...)
            ndim = ioattr[2]
            assert ob.ndim == ndim, \
                '%s dimension is %d should be %d' % (classname, ob.ndim, ndim)
            if attrtype == np.ndarray:
                dtp = ioattr[3]
                assert ob.dtype.kind == dtp.kind, \
                    '%s dtype.kind is %s should be %s' % (classname,
                                                          ob.dtype.kind,
                                                          dtp.kind)

        elif hasattr(ob, attrname):
            if getattr(ob, attrname) is not None:
                obattr = getattr(ob, attrname)
                assert issubclass(type(obattr), attrtype), \
                    '%s in %s is %s should be %s' % \
                    (attrname, classname, type(obattr), attrtype)
                if attrtype == pq.Quantity or attrtype == np.ndarray:
                    ndim = ioattr[2]
                    assert obattr.ndim == ndim,  \
                        '%s.%s dimension is %d should be %d' % \
                        (classname, attrname, obattr.ndim, ndim)
                if attrtype == np.ndarray:
                    dtp = ioattr[3]
                    assert obattr.dtype.kind == dtp.kind, \
                        '%s.%s dtype.kind is %s should be %s' % \
                        (classname, attrname, obattr.dtype.kind, dtp.kind)

    # test bijectivity : parents and children
    for container in getattr(ob, '_single_child_containers', []):
        for i, child in enumerate(getattr(ob, container, [])):
            assert hasattr(child, classname.lower()), \
                '%s should have %s attribute (2 way relationship)' % \
                (container, classname.lower())
            if hasattr(child, classname.lower()):
                parent = getattr(child, classname.lower())
                assert parent == ob, \
                    '%s.%s %s is not symetric with %s.%s' % \
                    (container, classname.lower(), i,
                     classname, container)

    # recursive on one to many rel
    for i, child in enumerate(getattr(ob, 'children', [])):
        try:
            assert_neo_object_is_compliant(child)
        # intercept exceptions and add more information
        except BaseException as exc:
            exc.args += ('from %s %s of %s' % (child.__class__.__name__, i,
                                               classname),)
            raise


def assert_same_sub_schema(ob1, ob2, equal_almost=True, threshold=1e-10,
                           exclude=None):
    '''
    Test if ob1 and ob2 has the same sub schema.
    Explore all parent/child relationships.
    Many_to_many_relationship is not tested
    because of infinite recursive loops.

    Arguments:
        equal_almost: if False do a strict arrays_equal if
                      True do arrays_almost_equal
        exclude: a list of attributes and annotations to ignore in
                 the comparison

    '''
    assert type(ob1) == type(ob2), 'type(%s) != type(%s)' % (type(ob1),
                                                             type(ob2))
    classname = ob1.__class__.__name__

    if exclude is None:
        exclude = []

    if isinstance(ob1, list):
        assert len(ob1) == len(ob2), \
            'lens %s and %s not equal for %s and %s' % \
            (len(ob1), len(ob2), ob1, ob2)
        for i, (sub1, sub2) in enumerate(zip(ob1, ob2)):
            try:
                assert_same_sub_schema(sub1, sub2, equal_almost=equal_almost,
                                       threshold=threshold, exclude=exclude)
            # intercept exceptions and add more information
            except BaseException as exc:
                exc.args += ('%s[%s]' % (classname, i),)
                raise
        return

    # test parent/child relationship
    for container in getattr(ob1, '_single_child_containers', []):
        if container in exclude:
            continue
        if not hasattr(ob1, container):
            assert not hasattr(ob2, container), \
                '%s 2 does have %s but not %s 1' % (classname, container,
                                                    classname)
            continue
        else:
            assert hasattr(ob2, container), \
                '%s 1 has %s but not %s 2' % (classname, container,
                                              classname)

        sub1 = getattr(ob1, container)
        sub2 = getattr(ob2, container)

        assert len(sub1) == len(sub2), \
            'theses two %s do not have the same %s number: %s and %s' % \
            (classname, container, len(sub1), len(sub2))
        for i in range(len(getattr(ob1, container))):
            # previously lacking parameter
            try:
                assert_same_sub_schema(sub1[i], sub2[i],
                                       equal_almost=equal_almost,
                                       threshold=threshold,
                                       exclude=exclude)
            # intercept exceptions and add more information
            except BaseException as exc:
                exc.args += ('from %s[%s] of %s' % (container, i,
                                                    classname),)
                raise

    assert_same_attributes(ob1, ob2, equal_almost=equal_almost,
                           threshold=threshold, exclude=exclude)


def assert_same_attributes(ob1, ob2, equal_almost=True, threshold=1e-10,
                           exclude=None):
    '''
    Test if ob1 and ob2 has the same attributes.

    Arguments:
        equal_almost: if False do a strict arrays_equal if
                      True do arrays_almost_equal
        exclude: a list of attributes and annotations to ignore in
                 the comparison

    '''
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
        #~ if attrname =='':
        if hasattr(ob1, '_quantity_attr') and ob1._quantity_attr == attrname:
            # object is hinerited from Quantity (AnalogSIgnal, SpikeTrain, ...)
            try:
                assert_arrays_almost_equal(ob1.magnitude, ob2.magnitude,
                                           threshold=threshold,
                                           dtype=dtype)
            # intercept exceptions and add more information
            except BaseException as exc:
                exc.args += ('from %s %s' % (classname, attrname),)
                raise
            assert ob1.dimensionality.string == ob2.dimensionality.string, \
                'Units of %s %s are not the same: %s and %s' % \
                (classname, attrname,
                 ob1.dimensionality.string, ob2.dimensionality.string)
            continue

        if not hasattr(ob1, attrname):
            assert not hasattr(ob2, attrname), \
                '%s 2 does have %s but not %s 1' % (classname, attrname,
                                                    classname)
            continue
        else:
            assert hasattr(ob2, attrname), \
                '%s 1 has %s but not %s 2' % (classname, attrname, classname)

        if getattr(ob1, attrname) is None:
            assert getattr(ob2, attrname) is None, \
                'In %s.%s %s and %s differed' % (classname, attrname,
                                                 getattr(ob1, attrname),
                                                 getattr(ob2, attrname))
            continue
        if getattr(ob2, attrname) is None:
            assert getattr(ob1, attrname) is None, \
                'In %s.%s %s and %s differed' % (classname, attrname,
                                                 getattr(ob1, attrname),
                                                 getattr(ob2, attrname))
            continue

        if attrtype == pq.Quantity:
            # Compare magnitudes
            mag1 = getattr(ob1, attrname).magnitude
            mag2 = getattr(ob2, attrname).magnitude
            #print "2. ob1(%s) %s:%s\n   ob2(%s) %s:%s" % \
                #(ob1,attrname,mag1,ob2,attrname,mag2)
            try:
                assert_arrays_almost_equal(mag1, mag2,
                                           threshold=threshold,
                                           dtype=dtype)
            # intercept exceptions and add more information
            except BaseException as exc:
                exc.args += ('from %s of %s' % (attrname, classname),)
                raise
            # Compare dimensionalities
            dim1 = getattr(ob1, attrname).dimensionality.simplified
            dim2 = getattr(ob2, attrname).dimensionality.simplified
            dimstr1 = getattr(ob1, attrname).dimensionality.string
            dimstr2 = getattr(ob2, attrname).dimensionality.string
            assert dim1 == dim2, \
                'Attribute %s of %s are not the same: %s != %s' % \
                (attrname, classname, dimstr1, dimstr2)
        elif attrtype == np.ndarray:
            try:
                assert_arrays_almost_equal(getattr(ob1, attrname),
                                           getattr(ob2, attrname),
                                           threshold=threshold,
                                           dtype=dtype)
            # intercept exceptions and add more information
            except BaseException as exc:
                exc.args += ('from %s of %s' % (attrname, classname),)
                raise
        else:
            #~ print 'yep', getattr(ob1, attrname),  getattr(ob2, attrname)
            assert getattr(ob1, attrname) == getattr(ob2, attrname), \
                'Attribute %s.%s are not the same %s %s %s %s' % \
                (classname, attrname,
                 type(getattr(ob1, attrname)), getattr(ob1, attrname),
                 type(getattr(ob2, attrname)), getattr(ob2, attrname))


def assert_same_annotations(ob1, ob2, equal_almost=True, threshold=1e-10,
                            exclude=None):
    '''
    Test if ob1 and ob2 has the same annotations.

    Arguments:
        equal_almost: if False do a strict arrays_equal if
                      True do arrays_almost_equal
        exclude: a list of attributes and annotations to ignore in
                 the comparison

    '''
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

    for key, value in ob1.annotations.items():
        if key in exclude:
            continue
        assert key in ob2.annotations
        try:
            assert value == ob2.annotations[key]
        except ValueError:
            assert_arrays_almost_equal(ob1, ob2,
                                       threshold=threshold, dtype=False)


def assert_sub_schema_is_lazy_loaded(ob):
    '''
    This is util for testing lazy load. All object must load with ndarray.size
    or Quantity.size ==0
    '''
    classname = ob.__class__.__name__

    for container in getattr(ob, '_single_child_containers', []):
        if not hasattr(ob, container):
            continue
        sub = getattr(ob, container)
        for i, child in enumerate(sub):
            try:
                assert_sub_schema_is_lazy_loaded(child)
            # intercept exceptions and add more information
            except BaseException as exc:
                exc.args += ('from %s %s of %s' % (container, i, classname),)
                raise

    for ioattr in ob._all_attrs:
        attrname, attrtype = ioattr[0], ioattr[1]
        #~ print 'xdsd', classname, attrname
        #~ if attrname == '':
        if hasattr(ob, '_quantity_attr') and ob._quantity_attr == attrname:
            assert ob.size == 0, \
                'Lazy loaded error %s.size = %s' % (classname, ob.size)
            assert hasattr(ob, 'lazy_shape'), \
                'Lazy loaded error, %s should have lazy_shape attribute' % \
                classname
            continue

        if not hasattr(ob, attrname) or getattr(ob, attrname) is None:
            continue
        #~ print 'hjkjh'
        if (attrtype == pq.Quantity or attrtype == np.ndarray):

            # FIXME: it is a workaround for recordingChannelGroup.channel_names
            # which is nupy.array but allowed to be loaded when lazy == True
            if ob.__class__ == neo.RecordingChannelGroup:
                continue

            ndim = ioattr[2]
            #~ print 'ndim', ndim
            #~ print getattr(ob, attrname).size
            if ndim >= 1:
                assert getattr(ob, attrname).size == 0, \
                    'Lazy loaded error %s.%s.size = %s' % \
                    (classname, attrname, getattr(ob, attrname).size)
                assert hasattr(ob,  'lazy_shape'), \
                    'Lazy loaded error ' +\
                    '%s should have lazy_shape attribute ' % classname +\
                    'because of %s attribute' % attrname


lazy_shape_arrays = {'SpikeTrain': 'times', 'Spike': 'waveform',
                     'AnalogSignal': 'signal',
                     'AnalogSignalArray': 'signal',
                     'EventArray': 'times', 'EpochArray': 'times'}


def assert_lazy_sub_schema_can_be_loaded(ob, io):
    '''
    This is util for testing lazy load. All object must load with ndarray.size
    or Quantity.size ==0
    '''
    classname = ob.__class__.__name__

    if classname in lazy_shape_arrays:
        new_load = io.load_lazy_object(ob)
        assert hasattr(ob, 'lazy_shape'), \
            'Object %s was not lazy loaded' % classname
        assert not hasattr(new_load, 'lazy_shape'), \
            'Newly loaded object from %s was also lazy loaded' % classname
        if hasattr(ob, '_quantity_attr'):
            assert ob.lazy_shape == new_load.shape, \
                'Shape of loaded object %sis not equal to lazy shape' % \
                classname
        else:
            assert ob.lazy_shape == \
                getattr(new_load, lazy_shape_arrays[classname]).shape, \
                'Shape of loaded object %s not equal to lazy shape' %\
                classname
        return

    for container in getattr(ob, '_single_child_containers', []):
        if not hasattr(ob, container):
            continue
        sub = getattr(ob, container)
        for i, child in enumerate(sub):
            try:
                assert_lazy_sub_schema_can_be_loaded(child, io)
            # intercept exceptions and add more information
            except BaseException as exc:
                exc.args += ('from of %s %s of %s' %
                             (container, i, classname),)
                raise


def assert_objects_equivalent(obj1, obj2):
    '''
    Compares two NEO objects by looping over the attributes and annotations
    and asserting their hashes. No relationships involved.
    '''
    def assert_attr(obj1, obj2, attr_name):
        '''
        Assert a single attribute and annotation are the same
        '''
        assert hasattr(obj1, attr_name)
        attr1 = hashlib.md5(getattr(obj1, attr_name)).hexdigest()
        assert hasattr(obj2, attr_name)
        attr2 = hashlib.md5(getattr(obj2, attr_name)).hexdigest()
        assert attr1 == attr2, "Attribute %s for class %s is not equal." % \
            (attr_name, obj1.__class__.__name__)
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
    '''
    Check that the children of a neo object are empty.  Used
    to check the cascade is implemented properly
    '''
    classname = obj.__class__.__name__
    errmsg = '''%s reader with cascade=False should return
        empty children''' % parent.__name__
    if hasattr(obj, 'children'):
        assert not obj.children, errmsg
