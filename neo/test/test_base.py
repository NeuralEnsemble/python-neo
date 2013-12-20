# -*- coding: utf-8 -*-
"""
Tests of the neo.core.baseneo.BaseNeo class and related functions
"""

from datetime import datetime, date, time, timedelta
from decimal import Decimal
from fractions import Fraction
import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo.core.baseneo import (BaseNeo, _check_annotations,
                              merge_annotation, merge_annotations)
from neo.test.tools import assert_arrays_equal


if sys.version_info[0] >= 3:
    _bytes = bytes

    long = int

    def bytes(s):
        return _bytes(s, encoding='ascii')


class Test_check_annotations(unittest.TestCase):
    '''
    TestCase to make sure _check_annotations works
    '''
    def setUp(self):
        self.values = [1, 2.2, 3 + 2j,
                       'test', r'test', b'test',
                       None,
                       datetime(year=2008, month=12, day=3, hour=10, minute=4),
                       timedelta(weeks=2, days=7, hours=18, minutes=28,
                                 seconds=18, milliseconds=28, microseconds=45),
                       time(hour=10, minute=4),
                       Decimal("3.14"), Fraction(13, 21),
                       np.array([1.1, 1.2, 1.3]),
                       np.array([1, 2, 3]),
                       np.array('test', dtype='S'),
                       np.array([True, False])]

    def test__check_annotations__invalid_ValueError(self):
        value = set([])
        self.assertRaises(ValueError, _check_annotations, value)

    def test__check_annotations__invalid_dtype_ValueError(self):
        value = np.array([], dtype='O')
        self.assertRaises(ValueError, _check_annotations, value)

    def test__check_annotations__valid_dtypes(self):
        for value in self.values:
            _check_annotations(value)

    def test__check_annotations__list(self):
        _check_annotations(self.values)

    def test__check_annotations__tuple(self):
        _check_annotations(tuple(self.values))
        _check_annotations((self.values, self.values))

    def test__check_annotations__dict(self):
        names = ['value%s' % i for i in range(len(self.values))]
        values = dict(zip(names, self.values))
        _check_annotations(values)


class Test_merge_annotation_annotations(unittest.TestCase):
    '''
    TestCase to make sure merge_annotation and merge_annotations work
    '''
    def test_merge_annotation__different_type_AssertionError(self):
        value1 = 'test'
        value2 = 5.5
        self.assertRaises(AssertionError, merge_annotation, value1, value2)

    def test_merge_annotation__unmergable_unequal_AssertionError(self):
        value1 = 5.6
        value2 = 5.5
        self.assertRaises(AssertionError, merge_annotation, value1, value2)

    def test_merge_annotation__str_unequal(self):
        value1 = 'test1'
        value2 = 'test2'
        targ = 'test1;test2'
        res = merge_annotation(value1, value2)
        self.assertEqual(targ, res)

    def test_merge_annotation__str_equal(self):
        value1 = 'test1'
        value2 = 'test1'
        targ = 'test1'
        res = merge_annotation(value1, value2)
        self.assertEqual(targ, res)

    def test_merge_annotation__ndarray(self):
        value1 = np.array([1, 2, 3])
        value2 = np.array([4, 5])
        targ = np.array([1, 2, 3, 4, 5])
        res = merge_annotation(value1, value2)
        assert_arrays_equal(targ, res)

    def test_merge_annotation__float_equal(self):
        value1 = 5.5
        value2 = 5.5
        targ = 5.5
        res = merge_annotation(value1, value2)
        self.assertEqual(targ, res)

    def test_merge_annotation__dict(self):
        value1 = {'val1': 1, 'val2': 2.2, 'val3': 'test1'}
        value2 = {'val2': 2.2, 'val3': 'test2', 'val4': [4, 4.4], 'val5': True}
        targ = {'val1': 1, 'val2': 2.2, 'val3': 'test1;test2',
                'val4': [4, 4.4], 'val5': True}
        res = merge_annotation(value1, value2)
        self.assertEqual(targ, res)

    def test_merge_annotations__dict(self):
        value1 = {'val1': 1, 'val2': 2.2, 'val3': 'test1'}
        value2 = {'val2': 2.2, 'val3': 'test2', 'val4': [4, 4.4], 'val5': True}
        targ = {'val1': 1, 'val2': 2.2, 'val3': 'test1;test2',
                'val4': [4, 4.4], 'val5': True}
        res = merge_annotations(value1, value2)
        self.assertEqual(targ, res)

    def test_merge_annotations__different_type_AssertionError(self):
        value1 = {'val1': 1, 'val2': 2.2, 'val3': 'tester'}
        value2 = {'val3': False, 'val4': [4, 4.4], 'val5': True}
        self.assertRaises(AssertionError, merge_annotations, value1, value2)

    def test_merge_annotations__unmergable_unequal_AssertionError(self):
        value1 = {'val1': 1, 'val2': 2.2, 'val3': True}
        value2 = {'val3': False, 'val4': [4, 4.4], 'val5': True}
        self.assertRaises(AssertionError, merge_annotation, value1, value2)


class TestBaseNeo(unittest.TestCase):
    '''
    TestCase to make sure basic initialization and methods work
    '''
    def test_init(self):
        '''test to make sure initialization works properly'''
        base = BaseNeo(name='a base', description='this is a test')
        self.assertEqual(base.name, 'a base')
        self.assertEqual(base.description, 'this is a test')
        self.assertEqual(base.file_origin, None)

    def test_annotate(self):
        '''test to make sure annotation works properly'''
        base = BaseNeo()
        base.annotate(test1=1, test2=1)
        result1 = {'test1': 1, 'test2': 1}
        self.assertDictEqual(result1, base.annotations)
        base.annotate(test3=2, test4=3)
        result2 = {'test3': 2, 'test4': 3}
        result2a = dict(list(result1.items()) + list(result2.items()))
        self.assertDictContainsSubset(result1, base.annotations)
        self.assertDictContainsSubset(result2, base.annotations)
        self.assertDictEqual(result2a, base.annotations)
        base.annotate(test1=5, test2=8)
        result3 = {'test1': 5, 'test2': 8}
        result3a = dict(list(result3.items()) + list(result2.items()))
        self.assertDictContainsSubset(result2, base.annotations)
        self.assertDictContainsSubset(result3, base.annotations)
        self.assertDictEqual(result3a, base.annotations)
        self.assertNotEqual(base.annotations['test1'], result1['test1'])
        self.assertNotEqual(base.annotations['test2'], result1['test2'])


class TestBaseNeoCoreTypes(unittest.TestCase):
    '''
    TestCase to make sure annotations are properly checked for core built-in
    python data types
    '''
    def setUp(self):
        '''create the instance to be tested, called before every test'''
        self.base = BaseNeo()

    def test_python_nonetype(self):
        '''test to make sure None type data is accepted'''
        value = None
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_int(self):
        '''test to make sure int type data is accepted'''
        value = 10
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_long(self):
        '''test to make sure long type data is accepted'''
        value = long(7)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_float(self):
        '''test to make sure float type data is accepted'''
        value = 9.2
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_complex(self):
        '''test to make sure complex type data is accepted'''
        value = complex(23.17, 11.29)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_string(self):
        '''test to make sure string type data is accepted'''
        value = 'this is a test'
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_unicode(self):
        '''test to make sure unicode type data is accepted'''
        value = u'this is also a test'
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_bytes(self):
        '''test to make sure bytes type data is accepted'''
        value = bytes('1,2,3,4,5')
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)


class TestBaseNeoStandardLibraryTypes(unittest.TestCase):
    '''
    TestCase to make sure annotations are properly checked for data types from
    the python standard library that are not core built-in data types
    '''
    def setUp(self):
        '''create the instance to be tested, called before every test'''
        self.base = BaseNeo()

    def test_python_fraction(self):
        '''test to make sure Fraction type data is accepted'''
        value = Fraction(13, 21)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_decimal(self):
        '''test to make sure Decimal type data is accepted'''
        value = Decimal("3.14")
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_datetime(self):
        '''test to make sure datetime type data is accepted'''
        value = datetime(year=2008, month=12, day=3, hour=10, minute=4)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_date(self):
        '''test to make sure date type data is accepted'''
        value = date(year=2008, month=12, day=3)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_time(self):
        '''test to make sure time type data is accepted'''
        value = time(hour=10, minute=4)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_timedelta(self):
        '''test to make sure timedelta type data is accepted'''
        value = timedelta(weeks=2, days=7, hours=18, minutes=28,
                          seconds=18, milliseconds=28,
                          microseconds=45)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)


class TestBaseNeoContainerTypes(unittest.TestCase):
    '''
    TestCase to make sure annotations are properly checked for data type
    inside python built-in container types
    '''
    def setUp(self):
        '''create the instance to be tested, called before every test'''
        self.base = BaseNeo()

    def test_python_list(self):
        '''test to make sure list type data is accepted'''
        value = [None, 10, 9.2, complex(23, 11),
                 ['this is a test', bytes('1,2,3,4,5')],
                 [Fraction(13, 21), Decimal("3.14")]]
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertListEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_tuple(self):
        '''test to make sure tuple type data is accepted'''
        value = (None, 10, 9.2, complex(23, 11),
                 ('this is a test', bytes('1,2,3,4,5')),
                 (Fraction(13, 21), Decimal("3.14")))
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertTupleEqual(value, self.base.annotations['data'])
        self.assertDictEqual(result, self.base.annotations)

    def test_python_dict(self):
        '''test to make sure dict type data is accepted'''
        value = {'NoneType': None, 'int': 10, 'float': 9.2,
                 'complex': complex(23, 11),
                 'dict1': {'string': 'this is a test',
                           'bytes': bytes('1,2,3,4,5')},
                 'dict2': {'Fraction': Fraction(13, 21),
                           'Decimal': Decimal("3.14")}}
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_python_set(self):
        '''test to make sure set type data is rejected'''
        value = set([None, 10, 9.2, complex(23, 11)])
        self.assertRaises(ValueError, self.base.annotate, data=value)

    def test_python_frozenset(self):
        '''test to make sure frozenset type data is rejected'''
        value = frozenset([None, 10, 9.2, complex(23, 11)])
        self.assertRaises(ValueError, self.base.annotate, data=value)

    def test_python_iter(self):
        '''test to make sure iter type data is rejected'''
        value = iter([None, 10, 9.2, complex(23, 11)])
        self.assertRaises(ValueError, self.base.annotate, data=value)


class TestBaseNeoNumpyArrayTypes(unittest.TestCase):
    '''
    TestCase to make sure annotations are properly checked for numpy arrays
    '''
    def setUp(self):
        '''create the instance to be tested, called before every test'''
        self.base = BaseNeo()

    def test_numpy_array_int(self):
        '''test to make sure int type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.int)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_uint(self):
        '''test to make sure uint type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.uint)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_int0(self):
        '''test to make sure int0 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.int0)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_uint0(self):
        '''test to make sure uint0 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.uint0)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_int8(self):
        '''test to make sure int8 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.int8)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_uint8(self):
        '''test to make sure uint8 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_int16(self):
        '''test to make sure int16 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_uint16(self):
        '''test to make sure uint16 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_int32(self):
        '''test to make sure int32 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_uint32(self):
        '''test to make sure uint32 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_int64(self):
        '''test to make sure int64 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_uint64(self):
        '''test to make sure uint64 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_float(self):
        '''test to make sure float type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.float)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_floating(self):
        '''test to make sure floating type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.floating)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_double(self):
        '''test to make sure double type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.double)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_float16(self):
        '''test to make sure float16 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.float16)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_float32(self):
        '''test to make sure float32 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_float64(self):
        '''test to make sure float64 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    @unittest.skipUnless(hasattr(np, "float128"), "float128 not available")
    def test_numpy_array_float128(self):
        '''test to make sure float128 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.float128)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_complex(self):
        '''test to make sure complex type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.complex)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_complex64(self):
        '''test to make sure complex64 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.complex64)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_complex128(self):
        '''test to make sure complex128 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    @unittest.skipUnless(hasattr(np, "complex256"),
                         "complex256 not available")
    def test_numpy_scalar_complex256(self):
        '''test to make sure complex256 type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.complex256)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_bool(self):
        '''test to make sure bool type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.bool)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_str(self):
        '''test to make sure str type numpy arrays are accepted'''
        value = np.array([1, 2, 3, 4, 5], dtype=np.str)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_string0(self):
        '''test to make sure string0 type numpy arrays are accepted'''
        if sys.version_info[0] >= 3:
            dtype = np.str0
        else:
            dtype = np.string0
        value = np.array([1, 2, 3, 4, 5], dtype=dtype)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)


class TestBaseNeoNumpyScalarTypes(unittest.TestCase):
    '''
    TestCase to make sure annotations are properly checked for numpy scalars
    '''
    def setUp(self):
        '''create the instance to be tested, called before every test'''
        self.base = BaseNeo()

    def test_numpy_scalar_int(self):
        '''test to make sure int type numpy scalars are accepted'''
        value = np.array(99, dtype=np.int)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_uint(self):
        '''test to make sure uint type numpy scalars are accepted'''
        value = np.array(99, dtype=np.uint)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_int0(self):
        '''test to make sure int0 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.int0)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_uint0(self):
        '''test to make sure uint0 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.uint0)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_int8(self):
        '''test to make sure int8 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.int8)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_uint8(self):
        '''test to make sure uint8 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.uint8)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_int16(self):
        '''test to make sure int16 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.int16)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_uint16(self):
        '''test to make sure uint16 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.uint16)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_int32(self):
        '''test to make sure int32 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.int32)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_uint32(self):
        '''test to make sure uint32 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.uint32)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_int64(self):
        '''test to make sure int64 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.int64)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_uint64(self):
        '''test to make sure uint64 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.uint64)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_float(self):
        '''test to make sure float type numpy scalars are accepted'''
        value = np.array(99, dtype=np.float)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_floating(self):
        '''test to make sure floating type numpy scalars are accepted'''
        value = np.array(99, dtype=np.floating)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_double(self):
        '''test to make sure double type numpy scalars are accepted'''
        value = np.array(99, dtype=np.double)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_float16(self):
        '''test to make sure float16 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.float16)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_float32(self):
        '''test to make sure float32 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.float32)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_float64(self):
        '''test to make sure float64 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.float64)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    @unittest.skipUnless(hasattr(np, "float128"), "float128 not available")
    def test_numpy_scalar_float128(self):
        '''test to make sure float128 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.float128)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_complex(self):
        '''test to make sure complex type numpy scalars are accepted'''
        value = np.array(99, dtype=np.complex)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_complex64(self):
        '''test to make sure complex64 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.complex64)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_complex128(self):
        '''test to make sure complex128 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.complex128)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    @unittest.skipUnless(hasattr(np, "complex256"), "complex256 not available")
    def test_numpy_scalar_complex256(self):
        '''test to make sure complex256 type numpy scalars are accepted'''
        value = np.array(99, dtype=np.complex256)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_bool(self):
        '''test to make sure bool type numpy scalars are rejected'''
        value = np.array(99, dtype=np.bool)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_array_str(self):
        '''test to make sure str type numpy scalars are accepted'''
        value = np.array(99, dtype=np.str)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_numpy_scalar_string0(self):
        '''test to make sure string0 type numpy scalars are rejected'''
        if sys.version_info[0] >= 3:
            dtype = np.str0
        else:
            dtype = np.string0
        value = np.array(99, dtype=dtype)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)


class TestBaseNeoQuantitiesArrayTypes(unittest.TestCase):
    '''
    TestCase to make sure annotations are properly checked for quantities
    arrays
    '''
    def setUp(self):
        '''create the instance to be tested, called before every test'''
        self.base = BaseNeo()

    def test_quantities_array_int(self):
        '''test to make sure int type quantites arrays are accepted'''
        value = pq.Quantity([1, 2, 3, 4, 5], dtype=np.int, units=pq.s)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_quantities_array_uint(self):
        '''test to make sure uint type quantites arrays are accepted'''
        value = pq.Quantity([1, 2, 3, 4, 5], dtype=np.uint, units=pq.meter)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_quantities_array_float(self):
        '''test to make sure float type quantites arrays are accepted'''
        value = [1, 2, 3, 4, 5] * pq.kg
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_quantities_array_str(self):
        '''test to make sure str type quantites arrays are accepted'''
        value = pq.Quantity([1, 2, 3, 4, 5], dtype=np.str, units=pq.meter)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)


class TestBaseNeoQuantitiesScalarTypes(unittest.TestCase):
    '''
    TestCase to make sure annotations are properly checked for quantities
    scalars
    '''
    def setUp(self):
        '''create the instance to be tested, called before every test'''
        self.base = BaseNeo()

    def test_quantities_scalar_int(self):
        '''test to make sure int type quantites scalars are accepted'''
        value = pq.Quantity(99, dtype=np.int, units=pq.s)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_quantities_scalar_uint(self):
        '''test to make sure uint type quantites scalars are accepted'''
        value = pq.Quantity(99, dtype=np.uint, units=pq.meter)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_quantities_scalar_float(self):
        '''test to make sure float type quantites scalars are accepted'''
        value = 99 * pq.kg
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)

    def test_quantities_scalar_str(self):
        '''test to make sure str type quantites scalars are accepted'''
        value = pq.Quantity(99, dtype=np.str, units=pq.meter)
        self.base.annotate(data=value)
        result = {'data': value}
        self.assertDictEqual(result, self.base.annotations)


class TestBaseNeoUserDefinedTypes(unittest.TestCase):
    '''
    TestCase to make sure annotations are properly checked for arbitrary
    objects
    '''

    def setUp(self):
        '''create the instance to be tested, called before every test'''
        self.base = BaseNeo()

    def test_my_class(self):
        '''test to make sure user defined class type data is rejected'''
        class Foo(object):
            pass
        value = Foo()
        self.assertRaises(ValueError, self.base.annotate, data=value)

    def test_my_class_list(self):
        '''test to make sure user defined class type data is rejected'''
        class Foo(object):
            pass
        value = [Foo(), Foo(), Foo()]
        self.assertRaises(ValueError, self.base.annotate, data=value)


if __name__ == "__main__":
    unittest.main()
