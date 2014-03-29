# -*- coding: utf-8 -*-
"""
Tests of neo.io.hdf5io

Usually I run these tests like that. I add neo root folder to the pythonpath
(usually by adding the neo.pth with the path to the cloned repository to, say,
/usr/lib/python2.6/dist-packages/) and run

python <path to the neo repo>/test/io/test_hdf5io.py

For the moment only basic tests are active.

#TODO add performance testing!!
"""

# needed for python 3 compatibility
from __future__ import absolute_import

from hashlib import md5
import os
import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo.core import SpikeTrain, Segment, Block, objectnames
from neo.test.tools import (assert_neo_object_is_compliant,
                            assert_objects_equivalent,
                            assert_same_sub_schema)
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.generate_datasets import fake_neo, get_fake_value

from neo.io.hdf5io import NeoHdf5IO, HAVE_TABLES


#==============================================================================


class HDF5Commontests(BaseTestIO, unittest.TestCase):
    ioclass = NeoHdf5IO
    files_to_test = ['test.h5']
    files_to_download = files_to_test

    @unittest.skipIf(sys.version_info[0] > 2, "not Python 3 compatible")
    @unittest.skipUnless(HAVE_TABLES, "requires PyTables")
    def setUp(self):
        BaseTestIO.setUp(self)


class hdf5ioTest:  # inherit this class from unittest.TestCase when ready
    """
    Tests for the hdf5 library.
    """

    #@unittest.skipIf(sys.version_info[0] > 2, "not Python 3 compatible")
    #@unittest.skipUnless(HAVE_TABLES, "requires PyTables")
    def setUp(self):
        self.test_file = "test.h5"

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_create(self):
        """
        Create test file with signals, segments, blocks etc.
        """
        iom = NeoHdf5IO(filename=self.test_file)
        b1 = fake_neo()  # creating a structure
        iom.save(b1)  # saving
        # must be assigned after save
        self.assertTrue(hasattr(b1, "hdf5_path"))
        iom.close()
        iom.connect(filename=self.test_file)
        b2 = iom.get(b1.hdf5_path)  # new object
        assert_neo_object_is_compliant(b2)
        assert_same_sub_schema(b1, b2)

    def test_property_change(self):
        """ Make sure all attributes are saved properly after the change,
        including quantities, units, types etc."""
        iom = NeoHdf5IO(filename=self.test_file)
        for obj_type in objectnames:
            obj = fake_neo(obj_type, cascade=False)
            iom.save(obj)
            self.assertTrue(hasattr(obj, "hdf5_path"))
            replica = iom.get(obj.hdf5_path, cascade=False)
            assert_objects_equivalent(obj, replica)

    def test_relations(self):
        """
        make sure the change in relationships is saved properly in the file,
        including correct M2M, no redundancy etc. RC -> RCG not tested.
        """
        def assert_children(self, obj, replica):
            obj_type = obj.__name__
            self.assertEqual(md5(str(obj)).hexdigest(),
                             md5(str(replica)).hexdigest())
            for container in getattr(obj, '_child_containers', []):
                ch1 = getattr(obj, container)
                ch2 = getattr(replica, container)
                self.assertEqual(len(ch1), len(ch2))
                for i, v in enumerate(ch1):
                    self.assert_children(ch1[i], ch2[i])
        iom = NeoHdf5IO(filename=self.test_file)
        for obj_type in objectnames:
            obj = fake_neo(obj_type, cascade=True)
            iom.save(obj)
            self.assertTrue(hasattr(obj, "hdf5_path"))
            replica = iom.get(obj.hdf5_path, cascade=True)
            self.assert_children(obj, replica)

    def test_errors(self):
        """ some tests for specific errors """
        f = open("thisisafakehdf.h5", "w")  # wrong file type
        f.write("this is not an HDF5 file. sorry.")
        f.close()
        self.assertRaises(TypeError, NeoHdf5IO(filename="thisisafakehdf.h5"))
        iom = NeoHdf5IO(filename=self.test_file)  # wrong object path test
        self.assertRaises(LookupError, iom.get("/wrong_path"))
        some_object = np.array([1, 2, 3])  # non NEO object test
        self.assertRaises(AssertionError, iom.save(some_object))

    def test_attr_changes(self):
        """ gets an object, changes its attributes, saves it, then compares how
        good the changes were saved. """
        iom = NeoHdf5IO(filename=self.test_file)
        for obj_type in objectnames:
            obj = fake_neo(obj_type=obj_type, cascade=False)
            iom.save(obj)
            orig_obj = iom.get(obj.hdf5_path)
            for attr in obj._all_attrs:
                if hasattr(orig_obj, attr[0]):
                    setattr(obj, attr[0], get_fake_value(*attr))
            iom.save(orig_obj)
            test_obj = iom.get(orig_obj.hdf5_path)
            assert_objects_equivalent(orig_obj, test_obj)


        # changes!!! in attr AS WELL AS in relations!!
        # test annotations
        # test naming - paths
        # unicode!!
        # add a child, then remove, then check it's removed
        # update/removal of relations b/w RC and AS which are/not are in the
        # same segment

class HDF5MoreTests(unittest.TestCase):
    @unittest.skipIf(sys.version_info[0] > 2, "not Python 3 compatible")
    @unittest.skipUnless(HAVE_TABLES, "requires PyTables")
    def test_store_empty_spike_train(self):
        spiketrain0 = SpikeTrain([], t_start=0.0, t_stop=100.0, units="ms")
        spiketrain1 = SpikeTrain([23.4, 45.6, 67.8],
                                 t_start=0.0, t_stop=100.0, units="ms")
        segment = Segment(name="a_segment")
        segment.spiketrains.append(spiketrain0)
        segment.spiketrains.append(spiketrain1)
        block = Block(name="a_block")
        block.segments.append(segment)
        iom = NeoHdf5IO(filename="test987.h5")
        iom.save(block)
        iom.close()

        iom = NeoHdf5IO(filename="test987.h5")
        block1 = iom.get("/Block_0")
        self.assertEqual(block1.segments[0].spiketrains[0].t_stop, 100.0)
        self.assertEqual(len(block1.segments[0].spiketrains[0]), 0)
        self.assertEqual(len(block1.segments[0].spiketrains[1]), 3)
        iom.close()
        os.remove("test987.h5")


if __name__ == '__main__':
    unittest.main()
