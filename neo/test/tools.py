import numpy

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
