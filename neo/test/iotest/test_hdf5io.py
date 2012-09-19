"""
Usually I run these tests like that. I add neo root folder to the pythonpath
(usually by adding the neo.pth with the path to the cloned repository to, say,
/usr/lib/python2.6/dist-packages/) and run

python <path to the neo repo>/test/io/test_hdf5io.py

For the moment only basic tests are active.

"""


# add performance testing!!

import numpy as np
import quantities as pq
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from hashlib import md5
import datetime
import os
import logging

from neo.core import *
from neo.test.tools import assert_neo_object_is_compliant, assert_objects_equivalent
from neo.test.io.common_io_test import BaseTestIO
from neo.description import *

try:
    from neo.io.hdf5io import NeoHdf5IO
    have_hdf5 = True
except ImportError:
    NeoHdf5IO = None
    have_hdf5 = False

#===============================================================================

TEST_ANNOTATIONS = [1, 0, 1.5, "this is a test", datetime.now(), None]

def get_fake_value(attr): # attr = (name, type, [dim, [dtype]])
    """ returns default value for a given attribute based on description.py """
    if attr[1] == pq.Quantity or attr[1] == np.ndarray:
        size = []
        for i in range(int(attr[2])):
            size.append(np.random.randint(100) + 1)
        to_set = np.random.random(size) * pq.millisecond # let it be ms
        if attr[0] == 't_start': to_set = 0.0 * pq.millisecond
        if attr[0] == 't_stop': to_set = 1.0 * pq.millisecond
        if attr[0] == 'sampling_rate': to_set = 10000.0 * pq.Hz
    if attr[1] == np.ndarray:
        to_set = np.array(to_set, dtype=attr[3])
    if attr[1] == str:
        to_set = str(np.random.randint(100000))
    if attr[1] == int:
        to_set = np.random.randint(100)
    if attr[1] == datetime:
        to_set = datetime.now()
    return to_set

def fake_NEO(obj_type="Block", cascade=True, _follow_links=True):
    """ Create a fake NEO object of a given type. Follows one-to-many 
    and many-to-many relationships if cascade. RC, when requested cascade, will
    not create RGCs to avoid dead-locks.

    _follow_links - an internal variable, indicates whether to create objects 
    with 'implicit' relationships, to avoid duplications. Do not use it. """
    kwargs = {} # assign attributes
    attrs = classes_necessary_attributes[obj_type] + \
        classes_recommended_attributes[obj_type]
    for attr in attrs:
        kwargs[attr[0]] = get_fake_value(attr)
    obj = class_by_name[obj_type](**kwargs)
    if cascade:
        if obj_type == "Block":
            _follow_links = False
        if one_to_many_relationship.has_key(obj_type):
            rels = one_to_many_relationship[obj_type]
            if obj_type == "RecordingChannelGroup":
                rels += many_to_many_relationship[obj_type]
            if not _follow_links and implicit_relationship.has_key(obj_type):
                for i in implicit_relationship[obj_type]:
                    if not i in rels: 
                        logging.debug("LOOK HERE!!!" + str(obj_type))
                    rels.remove(i)
            for child in rels:
                setattr(obj, child.lower() + "s", [fake_NEO(child, cascade, 
                        _follow_links)])
    if obj_type == "Block": # need to manually create 'implicit' connections
        # connect a unit to the spike and spike train
        u = obj.recordingchannelgroups[0].units[0]
        st = obj.segments[0].spiketrains[0]
        sp = obj.segments[0].spikes[0]
        u.spiketrains.append(st)
        u.spikes.append(sp)
        # connect RCG with ASA
        asa = obj.segments[0].analogsignalarrays[0]
        obj.recordingchannelgroups[0].analogsignalarrays.append(asa)
        # connect RC to AS, IrSAS and back to RGC 
        rc = obj.recordingchannelgroups[0].recordingchannels[0]
        rc.recordingchannelgroups.append(obj.recordingchannelgroups[0])
        rc.analogsignals.append(obj.segments[0].analogsignals[0])
        rc.irregularlysampledsignals.append(obj.segments[0].irregularlysampledsignals[0])
    # add some annotations, 80%
    at = dict([(str(x), TEST_ANNOTATIONS[x]) for x in range(len(TEST_ANNOTATIONS))])
    obj.annotate(**at)
    return obj

class HDF5Commontests(BaseTestIO, unittest.TestCase):
    ioclass = NeoHdf5IO
    files_to_test = [  ]
    files_to_download =  [   ]
    
    @unittest.skipUnless(have_hdf5, "requires PyTables")
    def setUp(self):
        BaseTestIO.setUp(self)


class hdf5ioTest: # inherit this class from unittest.TestCase when ready
    """
    Tests for the hdf5 library.
    """
    
    #@unittest.skipUnless(have_hdf5, "requires PyTables")
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
        b1 = fake_NEO() # creating a structure
        iom.save(b1) # saving
        self.assertTrue(hasattr(b1, "hdf5_path")) # must be assigned after save
        iom.close()
        iom.connect(filename=self.test_file)
        b2 = iom.get(b1.hdf5_path) # new object
        assert_neo_object_is_compliant(b2)
        assert_same_sub_schema(b1, b2)

    def test_property_change(self):
        """ Make sure all attributes are saved properly after the change, 
        including quantities, units, types etc."""
        iom = NeoHdf5IO(filename=self.test_file)
        for obj_type in class_by_name.keys():
            obj = fake_NEO(obj_type, cascade=False)
            iom.save(obj)
            self.assertTrue(hasattr(obj, "hdf5_path"))
            replica = iom.get(obj.hdf5_path, cascade=False)
            assert_objects_equivalent(obj, replica)

    def test_relations(self):
        """ make sure the change in relationships is saved properly in the file,
        including correct M2M, no redundancy etc. RC -> RCG not tested."""
        def assert_children(self, obj, replica):
            obj_type = name_by_class[obj]
            self.assertEqual(md5(str(obj)).hexdigest(), md5(str(replica)).hexdigest())
            if one_to_many_relationship.has_key(obj_type):
                rels = one_to_many_relationship[obj_type]
                if obj_type == "RecordingChannelGroup":
                    rels += many_to_many_relationship[obj_type]
                for child_type in rels:
                    ch1 = getattr(obj, child_type.lower() + "s")
                    ch2 = getattr(replica, child_type.lower() + "s")
                    self.assertEqual(len(ch1), len(ch2))
                    for i, v in enumerate(ch1):
                        self.assert_children(ch1[i], ch2[i])
        iom = NeoHdf5IO(filename=self.test_file)
        for obj_type in class_by_name.keys():
            obj = fake_NEO(obj_type, cascade=True)
            iom.save(obj)
            self.assertTrue(hasattr(obj, "hdf5_path"))
            replica = iom.get(obj.hdf5_path, cascade=True)
            self.assert_children(obj, replica)

    def test_errors(self):
        """ some tests for specific errors """
        f = open("thisisafakehdf.h5", "w") # wrong file type
        f.write("this is not an HDF5 file. sorry.")
        f.close()
        self.assertRaises(TypeError, NeoHdf5IO(filename="thisisafakehdf.h5"))
        iom = NeoHdf5IO(filename=self.test_file) # wrong object path test
        self.assertRaises(LookupError, iom.get("/wrong_path"))
        some_object = np.array([1,2,3]) # non NEO object test
        self.assertRaises(AssertionError, iom.save(some_object))

    def test_attr_changes(self):
        """ gets an object, changes its attributes, saves it, then compares how
        good the changes were saved. """
        iom = NeoHdf5IO(filename=self.test_file)
        for obj_type in class_by_name.keys():
            obj = fake_NEO(obj_type = obj_type, cascade = False)
            iom.save(obj)
            orig_obj = iom.get(obj.hdf5_path)
            attrs = classes_necessary_attributes[obj_type] + classes_recommended_attributes[obj_type]
            for attr in attrs:
                if hasattr(orig_obj, attr[0]):
                    setattr(obj, attr[0], get_fake_value(attr))
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

def test_store_empty_spike_train():
        spiketrain0 = SpikeTrain([], t_start=0.0, t_stop=100.0, units="ms")
        spiketrain1 = SpikeTrain([23.4, 45.6, 67.8], t_start=0.0, t_stop=100.0, units="ms")
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
        assert block1.segments[0].spiketrains[0].t_stop == 100.0
        assert len(block1.segments[0].spiketrains[0]) == 0
        assert len(block1.segments[0].spiketrains[1]) == 3
        os.remove("test987.h5")

if __name__ == '__main__':
    unittest.main()



