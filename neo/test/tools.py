from __future__ import with_statement
import numpy
import hashlib
import os

def assert_arrays_equal(a, b):
    assert isinstance(a, numpy.ndarray), "a is a %s" % type(a)
    assert isinstance(b, numpy.ndarray), "b is a %s" % type(b)
    assert a.shape == b.shape, "%s != %s" % (a,b)
    assert (a.flatten()==b.flatten()).all(), "%s != %s" % (a,b)

def assert_arrays_almost_equal(a, b, threshold):
    assert isinstance(a, numpy.ndarray), "a is a %s" % type(a)
    assert isinstance(b, numpy.ndarray), "b is a %s" % type(b)
    assert a.shape == b.shape, "%s != %s" % (a,b)
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


def assert_same_sub_schema(ob1, ob2, equal_almost = False):
    """
    Test if ob1 and ob2 has the same sub schema.
    Explore all one_to_many_relationship.
    Many_to_many_reslationship is not tested because of infinite recursive loops.
    
    Arguments:
        equal_almost: if False do a strict arrays_equal if True do arrays_almost_equal
    
    """
    assert type(ob1) == type(ob2), '%s != %s' % (type(ob1), type(ob2))
        
    _type = type(ob1)
    classname =description.name_by_class[_type]
    
    if classname in description.one_to_many_reslationship:
        # test one_to_many_relationship
        for child in description.one_to_many_reslationship[classname]:
            if not hasattr(ob1, '_'+child+'s'):
                assert not hasattr(ob2, '_'+child+'s'), '%s 2 do have %s but not %s 1'%(classname, child, classname)
                continue
            else:
                assert hasattr(ob2, '_'+child+'s'), '%s 1 have %s but not %s 2'%(classname, child, classname)
            
            sub1 = getattr(ob1, '_'+child+'s')
            sub2 = getattr(ob2, '_'+child+'s')
            print len(sub1), len(sub2)
            assert len(sub1) == len(sub2), 'theses two %s have not the same %s number'%(classname, child)
            for i in range(len(getattr(ob1, '_'+child+'s'))):
                assert_same_sub_schema(sub1[i], sub2[i], equal_almost = equal_almost)
