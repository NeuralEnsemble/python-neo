# -*- coding: utf-8 -*-
"""
Tests of the neo.core.unit.Unit class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo.core.unit import Unit
from neo.core.spiketrain import SpikeTrain
from neo.core.spike import Spike
from neo.test.tools import assert_neo_object_is_compliant, assert_arrays_equal
from neo.io.tools import create_many_to_one_relationship


class TestUnit(unittest.TestCase):
    def setUp(self):
        self.setup_spikes()
        self.setup_spiketrains()
        self.setup_units()

    def setup_units(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        self.unit1 = Unit(name='test', description='tester 1',
                          file_origin='test.file', channels_indexes=[1],
                          testarg1=1, **params)
        self.unit2 = Unit(name='test', description='tester 2',
                          file_origin='test.file', channels_indexes=[2],
                          testarg1=1, **params)
        self.unit1.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        self.unit2.annotate(testarg11=1.1, testarg10=[1, 2, 3])

        self.unit1.spiketrains = self.train1
        self.unit2.spiketrains = self.train2

        self.unit1.spikes = self.spike1
        self.unit2.spikes = self.spike2

        create_many_to_one_relationship(self.unit1)
        create_many_to_one_relationship(self.unit2)

    def setup_spikes(self):
        spikename11 = 'spike 1 1'
        spikename12 = 'spike 1 2'
        spikename21 = 'spike 2 1'
        spikename22 = 'spike 2 2'

        spikedata11 = 10 * pq.ms
        spikedata12 = 20 * pq.ms
        spikedata21 = 30 * pq.s
        spikedata22 = 40 * pq.s

        self.spikenames1 = [spikename11, spikename12]
        self.spikenames2 = [spikename21, spikename22]
        self.spikenames = [spikename11, spikename12, spikename21, spikename22]

        spike11 = Spike(spikedata11, t_stop=100*pq.s, name=spikename11)
        spike12 = Spike(spikedata12, t_stop=100*pq.s, name=spikename12)
        spike21 = Spike(spikedata21, t_stop=100*pq.s, name=spikename21)
        spike22 = Spike(spikedata22, t_stop=100*pq.s, name=spikename22)

        self.spike1 = [spike11, spike12]
        self.spike2 = [spike21, spike22]
        self.spike = [spike11, spike12, spike21, spike22]

    def setup_spiketrains(self):
        trainname11 = 'spiketrain 1 1'
        trainname12 = 'spiketrain 1 2'
        trainname21 = 'spiketrain 2 1'
        trainname22 = 'spiketrain 2 2'

        traindata11 = np.arange(0, 10) * pq.ms
        traindata12 = np.arange(10, 20) * pq.ms
        traindata21 = np.arange(20, 30) * pq.s
        traindata22 = np.arange(30, 40) * pq.s

        self.trainnames1 = [trainname11, trainname12]
        self.trainnames2 = [trainname21, trainname22]
        self.trainnames = [trainname11, trainname12, trainname21, trainname22]

        train11 = SpikeTrain(traindata11, t_stop=100*pq.s, name=trainname11)
        train12 = SpikeTrain(traindata12, t_stop=100*pq.s, name=trainname12)
        train21 = SpikeTrain(traindata21, t_stop=100*pq.s, name=trainname21)
        train22 = SpikeTrain(traindata22, t_stop=100*pq.s, name=trainname22)

        self.train1 = [train11, train12]
        self.train2 = [train21, train22]
        self.train = [train11, train12, train21, train22]

    def test_unit_creation(self):
        assert_neo_object_is_compliant(self.unit1)
        assert_neo_object_is_compliant(self.unit2)

        self.assertEqual(self.unit1.name, 'test')
        self.assertEqual(self.unit2.name, 'test')

        self.assertEqual(self.unit1.description, 'tester 1')
        self.assertEqual(self.unit2.description, 'tester 2')

        self.assertEqual(self.unit1.file_origin, 'test.file')
        self.assertEqual(self.unit2.file_origin, 'test.file')

        self.assertEqual(self.unit1.annotations['testarg0'], [1, 2, 3])
        self.assertEqual(self.unit2.annotations['testarg10'], [1, 2, 3])

        self.assertEqual(self.unit1.annotations['testarg1'], 1.1)
        self.assertEqual(self.unit2.annotations['testarg1'], 1)
        self.assertEqual(self.unit2.annotations['testarg11'], 1.1)

        self.assertEqual(self.unit1.annotations['testarg2'], 'yes')
        self.assertEqual(self.unit2.annotations['testarg2'], 'yes')

        self.assertTrue(self.unit1.annotations['testarg3'])
        self.assertTrue(self.unit2.annotations['testarg3'])

        self.assertTrue(hasattr(self.unit1, 'spikes'))
        self.assertTrue(hasattr(self.unit2, 'spikes'))

        self.assertEqual(len(self.unit1.spikes), 2)
        self.assertEqual(len(self.unit2.spikes), 2)

        for res, targ in zip(self.unit1.spikes, self.spike1):
            self.assertEqual(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.unit2.spikes, self.spike2):
            self.assertEqual(res, targ)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.unit1, 'spiketrains'))
        self.assertTrue(hasattr(self.unit2, 'spiketrains'))

        self.assertEqual(len(self.unit1.spiketrains), 2)
        self.assertEqual(len(self.unit2.spiketrains), 2)

        for res, targ in zip(self.unit1.spiketrains, self.train1):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.unit2.spiketrains, self.train2):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

    def test_unit_merge(self):
        self.unit1.merge(self.unit2)

        spikeres1 = [sig.name for sig in self.unit1.spikes]
        spikeres2 = [sig.name for sig in self.unit2.spikes]

        trainres1 = [sig.name for sig in self.unit1.spiketrains]
        trainres2 = [sig.name for sig in self.unit2.spiketrains]

        self.assertEqual(spikeres1, self.spikenames)
        self.assertEqual(spikeres2, self.spikenames2)

        self.assertEqual(trainres1, self.trainnames)
        self.assertEqual(trainres2, self.trainnames2)

        for res, targ in zip(self.unit1.spikes, self.spike):
            self.assertEqual(res, targ)

        for res, targ in zip(self.unit2.spikes, self.spike2):
            self.assertEqual(res, targ)

        for res, targ in zip(self.unit1.spiketrains, self.train):
            assert_arrays_equal(res, targ)

        for res, targ in zip(self.unit2.spiketrains, self.train2):
            assert_arrays_equal(res, targ)


if __name__ == "__main__":
    unittest.main()
