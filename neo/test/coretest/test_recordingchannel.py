# -*- coding: utf-8 -*-
"""
Tests of the neo.core.recordingchannel.RecordingChannel class
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.recordingchannel import RecordingChannel
from neo.core import (AnalogSignal, IrregularlySampledSignal,
                      RecordingChannelGroup, Unit)
from neo.test.tools import (assert_neo_object_is_compliant,
                            assert_arrays_equal,
                            assert_same_sub_schema)
from neo.test.generate_datasets import (fake_neo, get_fake_value,
                                        get_fake_values, get_annotations,
                                        clone_object, TEST_ANNOTATIONS)


class Test__generate_datasets(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.annotations = dict([(str(x), TEST_ANNOTATIONS[x]) for x in
                                 range(len(TEST_ANNOTATIONS))])

    def test__get_fake_values(self):
        self.annotations['seed'] = 0
        index = get_fake_value('index', int, seed=0)
        coordinate = get_fake_value('coordinate', pq.Quantity, seed=1, dim=1)
        name = get_fake_value('name', str, seed=2, obj=RecordingChannel)
        description = get_fake_value('description', str, seed=3,
                                     obj='RecordingChannel')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'index': index,
                  'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(RecordingChannel, annotate=False, seed=0)
        res12 = get_fake_values('RecordingChannel', annotate=False, seed=0)
        res21 = get_fake_values(RecordingChannel, annotate=True, seed=0)
        res22 = get_fake_values('RecordingChannel', annotate=True, seed=0)

        assert_arrays_equal(res11.pop('coordinate'), coordinate)
        assert_arrays_equal(res12.pop('coordinate'), coordinate)
        assert_arrays_equal(res21.pop('coordinate'), coordinate)
        assert_arrays_equal(res22.pop('coordinate'), coordinate)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = RecordingChannel
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, RecordingChannel))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.analogsignals), 1)
        self.assertEqual(len(res.irregularlysampledsignals), 1)

        for child in res.children:
            del child.annotations['i']
            del child.annotations['j']
        self.assertEqual(res.analogsignals[0].annotations,
                         self.annotations)
        self.assertEqual(res.irregularlysampledsignals[0].annotations,
                         self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = 'RecordingChannel'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, RecordingChannel))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.analogsignals), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)


class TestRecordingChannel(unittest.TestCase):
    def setUp(self):
        self.nchildren = 2
        self.seed1 = 0
        self.seed2 = 10000
        self.rchan1 = fake_neo(RecordingChannel,
                               seed=self.seed1, n=self.nchildren)
        self.rchan2 = fake_neo(RecordingChannel,
                               seed=self.seed2, n=self.nchildren)
        self.targobj = self.rchan1

        self.sigs1 = self.rchan1.analogsignals
        self.sigs2 = self.rchan2.analogsignals
        self.irsigs1 = self.rchan1.irregularlysampledsignals
        self.irsigs2 = self.rchan2.irregularlysampledsignals

        self.sigs1a = clone_object(self.sigs1)
        self.irsigs1a = clone_object(self.irsigs1)

    def check_creation(self, rchan):
        assert_neo_object_is_compliant(rchan)

        seed = rchan.annotations['seed']

        targ0 = get_fake_value('index', int, seed=seed+0, obj=RecordingChannel)
        self.assertEqual(rchan.index, targ0)

        targ1 = get_fake_value('coordinate', pq.Quantity, dim=1, seed=seed+1)
        assert_arrays_equal(rchan.coordinate, targ1)

        targ2 = get_fake_value('name', str, seed=seed+2, obj=RecordingChannel)
        self.assertEqual(rchan.name, targ2)

        targ3 = get_fake_value('description', str,
                               seed=seed+3, obj=RecordingChannel)
        self.assertEqual(rchan.description, targ3)

        targ4 = get_fake_value('file_origin', str)
        self.assertEqual(rchan.file_origin, targ4)

        targ5 = get_annotations()
        targ5['seed'] = seed
        self.assertEqual(rchan.annotations, targ5)

        self.assertTrue(hasattr(rchan, 'analogsignals'))
        self.assertTrue(hasattr(rchan, 'irregularlysampledsignals'))

        self.assertEqual(len(rchan.analogsignals), self.nchildren)
        self.assertEqual(len(rchan.irregularlysampledsignals), self.nchildren)

    def test__creation(self):
        self.check_creation(self.rchan1)
        self.check_creation(self.rchan2)

    def test__merge(self):
        rchan1a = fake_neo(RecordingChannel, seed=self.seed1, n=self.nchildren)
        assert_same_sub_schema(self.rchan1, rchan1a)
        rchan1a.annotate(seed=self.seed2)
        rchan1a.analogsignals.append(self.sigs2[0])
        rchan1a.merge(self.rchan2)
        self.check_creation(self.rchan2)

        assert_same_sub_schema(self.sigs1a + self.sigs2[:1] + self.sigs2,
                               rchan1a.analogsignals)
        assert_same_sub_schema(self.irsigs1a + self.irsigs2,
                               rchan1a.irregularlysampledsignals)

    def test__children(self):
        rcg1 = RecordingChannelGroup(name='rcg1')
        rcg2 = RecordingChannelGroup(name='rcg2')
        rcg1.recordingchannels = [self.rchan1]
        rcg2.recordingchannels = [self.rchan1]
        rcg2.create_relationship()
        rcg1.create_relationship()
        assert_neo_object_is_compliant(self.rchan1)
        assert_neo_object_is_compliant(rcg1)
        assert_neo_object_is_compliant(rcg2)

        self.assertEqual(self.rchan1._container_child_objects, ())
        self.assertEqual(self.rchan1._data_child_objects,
                         ('AnalogSignal', 'IrregularlySampledSignal'))
        self.assertEqual(self.rchan1._single_parent_objects, ())
        self.assertEqual(self.rchan1._multi_child_objects, ())
        self.assertEqual(self.rchan1._multi_parent_objects,
                         ('RecordingChannelGroup',))
        self.assertEqual(self.rchan1._child_properties, ())

        self.assertEqual(self.rchan1._single_child_objects,
                         ('AnalogSignal', 'IrregularlySampledSignal'))

        self.assertEqual(self.rchan1._container_child_containers, ())
        self.assertEqual(self.rchan1._data_child_containers,
                         ('analogsignals', 'irregularlysampledsignals',))
        self.assertEqual(self.rchan1._single_child_containers,
                         ('analogsignals', 'irregularlysampledsignals',))
        self.assertEqual(self.rchan1._single_parent_containers, ())
        self.assertEqual(self.rchan1._multi_child_containers, ())
        self.assertEqual(self.rchan1._multi_parent_containers,
                         ('recordingchannelgroups',))

        self.assertEqual(self.rchan1._child_objects,
                         ('AnalogSignal', 'IrregularlySampledSignal'))
        self.assertEqual(self.rchan1._child_containers,
                         ('analogsignals', 'irregularlysampledsignals',))
        self.assertEqual(self.rchan1._parent_objects,
                         ('RecordingChannelGroup',))
        self.assertEqual(self.rchan1._parent_containers,
                         ('recordingchannelgroups',))

        self.assertEqual(len(self.rchan1.children), self.nchildren*2)

        assert_same_sub_schema(list(self.rchan1.children),
                               self.sigs1a+self.irsigs1a)

        self.assertEqual(len(self.rchan1.parents), 2)
        self.assertEqual(self.rchan1.parents[0].name, 'rcg2')
        self.assertEqual(self.rchan1.parents[1].name, 'rcg1')


if __name__ == "__main__":
    unittest.main()
