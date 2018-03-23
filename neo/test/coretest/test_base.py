# -*- coding: utf-8 -*-
"""
Tests of the neo.core.baseneo.BaseNeo class and related functions
"""

from datetime import datetime, date, time, timedelta
from decimal import Decimal
from fractions import Fraction
import sys

import unittest

import numpy as np
import quantities as pq

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.baseneo import (BaseNeo, _check_annotations,
                              merge_annotations, merge_annotation)
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

    def test__children(self):
        base = BaseNeo()

        self.assertEqual(base._single_parent_objects, ())
        self.assertEqual(base._multi_parent_objects, ())

        self.assertEqual(base._single_parent_containers, ())
        self.assertEqual(base._multi_parent_containers, ())

        self.assertEqual(base._parent_objects, ())
        self.assertEqual(base._parent_containers, ())

        self.assertEqual(base.parents, ())


class Test_BaseNeo_merge_annotations_merge(unittest.TestCase):
    '''
    TestCase to make sure merge_annotations and merge methods work
    '''

    def setUp(self):
        self.name1 = 'a base 1'
        self.name2 = 'a base 2'
        self.description1 = 'this is a test 1'
        self.description2 = 'this is a test 2'
        self.base1 = BaseNeo(name=self.name1, description=self.description1)
        self.base2 = BaseNeo(name=self.name2, description=self.description2)

    def test_merge_annotations__dict(self):
        self.base1.annotations = {'val0': 'val0', 'val1': 1,
                                  'val2': 2.2, 'val3': 'test1',
                                  'val4': [.4], 'val5': {0: 0, 1: {0: 0}},
                                  'val6': np.array([0, 1, 2])}
        self.base2.annotations = {'val2': 2.2, 'val3': 'test2',
                                  'val4': [4, 4.4], 'val5': {1: {1: 1}, 2: 2},
                                  'val6': np.array([4, 5, 6]), 'val7': True}

        ann1 = self.base1.annotations
        ann2 = self.base2.annotations
        ann1c = self.base1.annotations.copy()
        ann2c = self.base2.annotations.copy()

        targ = {'val0': 'val0', 'val1': 1, 'val2': 2.2, 'val3': 'test1;test2',
                'val4': [.4, 4, 4.4], 'val5': {0: 0, 1: {0: 0, 1: 1}, 2: 2},
                'val7': True}

        self.base1.merge_annotations(self.base2)

        val6t = np.array([0, 1, 2, 4, 5, 6])
        val61 = ann1.pop('val6')
        val61c = ann1c.pop('val6')
        val62 = ann2.pop('val6')
        val62c = ann2c.pop('val6')

        self.assertEqual(ann1, self.base1.annotations)
        self.assertNotEqual(ann1c, self.base1.annotations)
        self.assertEqual(ann2c, self.base2.annotations)
        self.assertEqual(targ, self.base1.annotations)

        assert_arrays_equal(val61, val6t)
        self.assertRaises(AssertionError, assert_arrays_equal, val61c, val6t)
        assert_arrays_equal(val62, val62c)

        self.assertEqual(self.name1, self.base1.name)
        self.assertEqual(self.name2, self.base2.name)
        self.assertEqual(self.description1, self.base1.description)
        self.assertEqual(self.description2, self.base2.description)

    def test_merge_annotations__func__dict(self):
        ann1 = {'val0': 'val0', 'val1': 1, 'val2': 2.2, 'val3': 'test1',
                'val4': [.4], 'val5': {0: 0, 1: {0: 0}},
                'val6': np.array([0, 1, 2])}
        ann2 = {'val2': 2.2, 'val3': 'test2',
                'val4': [4, 4.4], 'val5': {1: {1: 1}, 2: 2},
                'val6': np.array([4, 5, 6]), 'val7': True}

        ann1c = ann1.copy()
        ann2c = ann2.copy()

        targ = {'val0': 'val0', 'val1': 1, 'val2': 2.2, 'val3': 'test1;test2',
                'val4': [.4, 4, 4.4], 'val5': {0: 0, 1: {0: 0, 1: 1}, 2: 2},
                'val7': True}

        res = merge_annotations(ann1, ann2)

        val6t = np.array([0, 1, 2, 4, 5, 6])
        val6r = res.pop('val6')
        val61 = ann1.pop('val6')
        val61c = ann1c.pop('val6')
        val62 = ann2.pop('val6')
        val62c = ann2c.pop('val6')

        self.assertEqual(ann1, ann1c)
        self.assertEqual(ann2, ann2c)
        self.assertEqual(res, targ)

        assert_arrays_equal(val6r, val6t)
        self.assertRaises(AssertionError, assert_arrays_equal, val61, val6t)
        assert_arrays_equal(val61, val61c)
        assert_arrays_equal(val62, val62c)

    def test_merge_annotation__func__str(self):
        ann1 = 'test1'
        ann2 = 'test2'

        targ = 'test1;test2'

        res = merge_annotation(ann1, ann2)

        self.assertEqual(res, targ)

    def test_merge_annotation__func__ndarray(self):
        ann1 = np.array([0, 1, 2])
        ann2 = np.array([4, 5, 6])

        ann1c = ann1.copy()
        ann2c = ann2.copy()

        targ = np.array([0, 1, 2, 4, 5, 6])

        res = merge_annotation(ann1, ann2)

        assert_arrays_equal(res, targ)
        assert_arrays_equal(ann1, ann1c)
        assert_arrays_equal(ann2, ann2c)

    def test_merge_annotation__func__list(self):
        ann1 = [0, 1, 2]
        ann2 = [4, 5, 6]

        ann1c = ann1[:]
        ann2c = ann2[:]

        targ = [0, 1, 2, 4, 5, 6]

        res = merge_annotation(ann1, ann2)

        self.assertEqual(res, targ)
        self.assertEqual(ann1, ann1c)
        self.assertEqual(ann2, ann2c)

    def test_merge_annotation__func__dict(self):
        ann1 = {0: 0, 1: {0: 0}}
        ann2 = {1: {1: 1}, 2: 2}

        ann1c = ann1.copy()
        ann2c = ann2.copy()

        targ = {0: 0, 1: {0: 0, 1: 1}, 2: 2}

        res = merge_annotation(ann1, ann2)

        self.assertEqual(res, targ)
        self.assertEqual(ann1, ann1c)
        self.assertEqual(ann2, ann2c)

    def test_merge_annotation__func__int(self):
        ann1 = 1
        ann2 = 1
        ann3 = 3

        targ = 1

        res = merge_annotation(ann1, ann2)

        self.assertEqual(res, targ)
        self.assertRaises(AssertionError, merge_annotation, ann1, ann3)

    def test_merge_annotation__func__float(self):
        ann1 = 1.1
        ann2 = 1.1
        ann3 = 1.3

        targ = 1.1

        res = merge_annotation(ann1, ann2)

        self.assertEqual(res, targ)
        self.assertRaises(AssertionError, merge_annotation, ann1, ann3)

    def test_merge_annotation__func__bool(self):
        ann1 = False
        ann2 = False
        ann3 = True
        ann4 = True

        targ1 = False
        targ2 = True

        res1 = merge_annotation(ann1, ann2)
        res2 = merge_annotation(ann3, ann4)

        self.assertEqual(res1, targ1)
        self.assertEqual(res2, targ2)
        self.assertRaises(AssertionError, merge_annotation, ann1, ann3)
        self.assertRaises(AssertionError, merge_annotation, ann2, ann4)

    def test_merge__dict(self):
        self.base1.annotations = {'val0': 'val0', 'val1': 1,
                                  'val2': 2.2, 'val3': 'test1'}
        self.base2.annotations = {'val2': 2.2, 'val3': 'test2',
                                  'val4': [4, 4.4], 'val5': True}

        ann1 = self.base1.annotations
        ann1c = self.base1.annotations.copy()
        ann2c = self.base2.annotations.copy()

        targ = {'val0': 'val0', 'val1': 1, 'val2': 2.2, 'val3': 'test1;test2',
                'val4': [4, 4.4], 'val5': True}

        self.base1.merge(self.base2)

        self.assertEqual(ann1, self.base1.annotations)
        self.assertNotEqual(ann1c, self.base1.annotations)
        self.assertEqual(ann2c, self.base2.annotations)
        self.assertEqual(targ, self.base1.annotations)

        self.assertEqual(self.name1, self.base1.name)
        self.assertEqual(self.name2, self.base2.name)
        self.assertEqual(self.description1, self.base1.description)
        self.assertEqual(self.description2, self.base2.description)

    def test_merge_annotations__different_type_AssertionError(self):
        self.base1.annotations = {'val1': 1, 'val2': 2.2, 'val3': 'tester'}
        self.base2.annotations = {'val3': False, 'val4': [4, 4.4],
                                  'val5': True}
        self.base1.merge_annotations(self.base2)
        self.assertEqual(self.base1.annotations,
                         {'val1': 1,
                          'val2': 2.2,
                          'val3': 'MERGE CONFLICT',
                          'val4': [4, 4.4],
                          'val5': True})

    def test_merge__different_type_AssertionError(self):
        self.base1.annotations = {'val1': 1, 'val2': 2.2, 'val3': 'tester'}
        self.base2.annotations = {'val3': False, 'val4': [4, 4.4],
                                  'val5': True}
        self.base1.merge(self.base2)
        self.assertEqual(self.base1.annotations,
                         {'val1': 1,
                          'val2': 2.2,
                          'val3': 'MERGE CONFLICT',
                          'val4': [4, 4.4],
                          'val5': True})

    def test_merge_annotations__unmergable_unequal_AssertionError(self):
        self.base1.annotations = {'val1': 1, 'val2': 2.2, 'val3': True}
        self.base2.annotations = {'val3': False, 'val4': [4, 4.4],
                                  'val5': True}
        self.base1.merge_annotations(self.base2)
        self.assertEqual(self.base1.annotations,
                         {'val1': 1,
                          'val2': 2.2,
                          'val3': 'MERGE CONFLICT',
                          'val4': [4, 4.4],
                          'val5': True})

    def test_merge__unmergable_unequal_AssertionError(self):
        self.base1.annotations = {'val1': 1, 'val2': 2.2, 'val3': True}
        self.base2.annotations = {'val3': False, 'val4': [4, 4.4],
                                  'val5': True}
        self.base1.merge(self.base2)
        self.assertEqual(self.base1.annotations,
                         {'val1': 1,
                          'val2': 2.2,
                          'val3': 'MERGE CONFLICT',
                          'val4': [4, 4.4],
                          'val5': True})


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


@unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
class Test_pprint(unittest.TestCase):
    def test__pretty(self):
        name = 'an object'
        description = 'this is a test'
        obj = BaseNeo(name=name, description=description)
        res = pretty(obj)
        targ = "BaseNeo name: '%s' description: '%s'" % (name, description)
        self.assertEqual(res, targ)


if __name__ == "__main__":
    unittest.main()
