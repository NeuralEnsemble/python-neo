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
