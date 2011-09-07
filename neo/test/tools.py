from __future__ import with_statement
import numpy
import hashlib
import os
import quantities as pq

from neo import description

def assert_arrays_equal(a, b):
    assert isinstance(a, numpy.ndarray), "a is a %s" % type(a)
    assert isinstance(b, numpy.ndarray), "b is a %s" % type(b)
    assert a.shape == b.shape, "%s != %s" % (a,b)
    #assert a.dtype == b.dtype, "%s and %s not same dtype %s %s" % (a, b, a.dtype, b.dtype)
    assert (a.flatten()==b.flatten()).all(), "%s != %s" % (a, b)

def assert_arrays_almost_equal(a, b, threshold):
    assert isinstance(a, numpy.ndarray), "a is a %s" % type(a)
    assert isinstance(b, numpy.ndarray), "b is a %s" % type(b)
    assert a.shape == b.shape, "%s != %s" % (a,b)
    #assert a.dtype == b.dtype, "%s and %b not same dtype %s %s" % (a,b,a.dtype, b.dtype)
    if a.dtype.kind in ['f', 'c', 'i']:
        assert (abs(a - b) < threshold).all(), "max(|a - b|) = %s" % (abs(a - b)).max()

def file_digest(filename):
    with open(filename, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()

def assert_file_contents_equal(a, b):
    def generate_error_message(a, b):
        size_a = os.stat(a).st_size
        size_b = os.stat(b).st_size
        if size_a == size_b:
            return "Files have the same size but different contents"
        else:
            return "Files have different sizes: a:%d b: %d" % (size_a, size_b) 
                                 
    assert file_digest(a) == file_digest(b), generate_error_message(a, b)


def assert_same_sub_schema(ob1, ob2, equal_almost = False, threshold = 1e-10):
    """
    Test if ob1 and ob2 has the same sub schema.
    Explore all one_to_many_relationship.
    Many_to_many_reslationship is not tested because of infinite recursive loops.
    
    Arguments:
        equal_almost: if False do a strict arrays_equal if True do arrays_almost_equal
    
    """
    assert type(ob1) == type(ob2), '%s != %s' % (type(ob1), type(ob2))
        
    #~ _type = type(ob1)
    #~ classname =description.name_by_class[_type]
    classname =ob1.__class__.__name__
    
    if classname in description.one_to_many_reslationship:
        # test one_to_many_relationship
        for child in description.one_to_many_reslationship[classname]:
            if not hasattr(ob1, '_'+child.lower()+'s'):
                assert not hasattr(ob2, '_'+child.lower()+'s'), '%s 2 do have %s but not %s 1'%(classname, child, classname)
                continue
            else:
                assert hasattr(ob2, '_'+child.lower()+'s'), '%s 1 have %s but not %s 2'%(classname, child, classname)
            
            sub1 = getattr(ob1, '_'+child.lower()+'s')
            sub2 = getattr(ob2, '_'+child.lower()+'s')
            assert len(sub1) == len(sub2), 'theses two %s have not the same %s number'%(classname, child)
            for i in range(len(getattr(ob1, '_'+child.lower()+'s'))):
                assert_same_sub_schema(sub1[i], sub2[i], equal_almost = equal_almost)
    
    # check if all attributes are equal
    if equal_almost:
        def assert_arrays_equal_and_dtype(a,b):
            assert_arrays_equal(a,b)
            assert a.dtype == b.dtype, "%s and %s not same dtype %s %s" % (a, b, a.dtype, b.dtype)
        assert_eg = assert_arrays_equal_and_dtype
    else:
        def assert_arrays_almost_and_dtype(a,b):
            assert_arrays_almost_equal(a,b,threshold)
            assert a.dtype == b.dtype, "%s and %s not same dtype %s %s" % (a, b, a.dtype, b.dtype)
        assert_eg = assert_arrays_almost_and_dtype
    
    #~ classname = description.name_by_class[_type]
    necess = description.classes_necessary_attributes[classname]
    recomm = description.classes_recommended_attributes[classname]
    attributes = necess + recomm
    for i, attr in enumerate(attributes):
        attrname, attrtype = attr[0], attr[1]
        if not hasattr(ob1, attrname):
            assert not hasattr(ob2, attrname), '%s 2 do have %s but not %s 1'%(classname, attrname, classname)
            continue
        else:
            assert hasattr(ob2, attrname), '%s 1 have %s but not %s 2'%(classname, attrname, classname)
        
        if getattr(ob1,attrname)  is None:
            assert getattr(ob2,attrname)  is None, '%s and %s differed'
            continue
        if getattr(ob2,attrname)  is None:
            assert getattr(ob1,attrname)  is None, '%s and %s differed'
            continue
        
        if attrname =='': 
            # object is hinerited from Quantity (AnalogSIgnal, SpikeTrain, ...)
            assert_eg(ob1.magnitude, ob2.magnitude)
            assert ob1.dimensionality.string == ob2.dimensionality.string, 'Units of %s are not the same' % classname
        elif attrtype == pq.Quantity:
            assert_eg(ob1.__getattr__(attrname).magnitude, ob2.__getattr__(attrname).magnitude)
            assert ob1.__getattr__(attrname).dimensionality.string == ob2.__getattr__(attrname).dimensionality.string, 'Attribute %s of %s are not the same' % (attrname, classname)
        elif attrtype == numpy.ndarray:
            assert_eg(ob1.__getattr__(attrname), ob2.__getattr__(attrname))
        else:
            assert ob1.__getattr__(attrname) == ob2.__getattr__(attrname), 'Attribute %s.%s are not the same %s %s' % (classname,attrname, type(ob1.__getattr__(attrname)),  type(ob2.__getattr__(attrname)))

