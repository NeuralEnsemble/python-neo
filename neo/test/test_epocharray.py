# -*- coding: utf-8 -*-
"""
Tests of the neo.core.epocharray.EpochArray class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo.core.epocharray import EpochArray
from neo.test.tools import (assert_neo_object_is_compliant,
                            assert_arrays_equal, assert_same_sub_schema)


class TestEpochArray(unittest.TestCase):
    def test_EpochArray_creation(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        epca = EpochArray([1.1, 1.5, 1.7]*pq.ms, durations=[20, 40, 60]*pq.ns,
                          labels=np.array(['test epoch 1',
                                           'test epoch 2',
                                           'test epoch 3'], dtype='S'),
                          name='test', description='tester',
                          file_origin='test.file',
                          testarg1=1, **params)
        epca.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        assert_neo_object_is_compliant(epca)

        assert_arrays_equal(epca.times, [1.1, 1.5, 1.7]*pq.ms)
        assert_arrays_equal(epca.durations, [20, 40, 60]*pq.ns)
        assert_arrays_equal(epca.labels, np.array(['test epoch 1',
                                                   'test epoch 2',
                                                   'test epoch 3'], dtype='S'))
        self.assertEqual(epca.name, 'test')
        self.assertEqual(epca.description, 'tester')
        self.assertEqual(epca.file_origin, 'test.file')
        self.assertEqual(epca.annotations['testarg0'], [1, 2, 3])
        self.assertEqual(epca.annotations['testarg1'], 1.1)
        self.assertEqual(epca.annotations['testarg2'], 'yes')
        self.assertTrue(epca.annotations['testarg3'])

    def test_EpochArray_repr(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        epca = EpochArray([1.1, 1.5, 1.7]*pq.ms, durations=[20, 40, 60]*pq.ns,
                          labels=np.array(['test epoch 1',
                                           'test epoch 2',
                                           'test epoch 3'], dtype='S'),
                          name='test', description='tester',
                          file_origin='test.file',
                          testarg1=1, **params)
        epca.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        assert_neo_object_is_compliant(epca)

        targ = ('<EventArray: test epoch 1@1.1 ms for 20.0 ns, ' +
                'test epoch 2@1.5 ms for 40.0 ns, ' +
                'test epoch 3@1.7 ms for 60.0 ns>')

        res = repr(epca)

        self.assertEqual(targ, res)

    def test_EpochArray_merge(self):
        params1 = {'testarg2': 'yes', 'testarg3': True}
        params2 = {'testarg2': 'no', 'testarg4': False}
        paramstarg = {'testarg2': 'yes;no',
                      'testarg3': True,
                      'testarg4': False}
        epca1 = EpochArray([1.1, 1.5, 1.7]*pq.ms,
                           durations=[20, 40, 60]*pq.us,
                           labels=np.array(['test epoch 1 1',
                                            'test epoch 1 2',
                                            'test epoch 1 3'], dtype='S'),
                           name='test', description='tester 1',
                           file_origin='test.file',
                           testarg1=1, **params1)
        epca2 = EpochArray([2.1, 2.5, 2.7]*pq.us,
                           durations=[3, 5, 7]*pq.ms,
                           labels=np.array(['test epoch 2 1',
                                            'test epoch 2 2',
                                            'test epoch 2 3'], dtype='S'),
                           name='test', description='tester 2',
                           file_origin='test.file',
                           testarg1=1, **params2)
        epcatarg = EpochArray([1.1, 1.5, 1.7, .0021, .0025, .0027]*pq.ms,
                              durations=[20, 40, 60, 3000, 5000, 7000]*pq.ns,
                              labels=np.array(['test epoch 1 1',
                                               'test epoch 1 2',
                                               'test epoch 1 3',
                                               'test epoch 2 1',
                                               'test epoch 2 2',
                                               'test epoch 2 3'], dtype='S'),
                              name='test',
                              description='merge(tester 1, tester 2)',
                              file_origin='test.file',
                              testarg1=1, **paramstarg)
        assert_neo_object_is_compliant(epca1)
        assert_neo_object_is_compliant(epca2)
        assert_neo_object_is_compliant(epcatarg)

        epcares = epca1.merge(epca2)
        assert_neo_object_is_compliant(epcares)
        assert_same_sub_schema(epcatarg, epcares)


if __name__ == "__main__":
    unittest.main()
