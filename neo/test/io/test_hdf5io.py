# TODO
# Andrey: please remove this to test when your IO is finished
__test__ = False 


# add performance testing!!

import numpy as np
import quantities as pq
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from md5 import md5
import datetime
import os

from neo.core import *
from neo.test.tools import assert_neo_object_is_compliant
from neo.test.io.common_io_test import BaseTestIO
from neo.description import *

try:
    from neo.io.hdf5io import NeoHdf5IO
    have_hdf5 = True
except ImportError:
    NeoHdf5IO = None
    have_hdf5 = False

#===============================================================================

TEST_ANNOTATIONS = [1, 0, 1.5, "this is a test", datetime.datetime.now(), 
    datetime.date.today(), None]

def get_fake_value(attr): # attr = (name, type, [dim, [dtype]])
    """ returns default value for a given attribute based on description.py """
    if attr[1] == pq.Quantity or attr[1] == np.ndarray:
        size = []
        for i in range(int(attr[2])):
            size.append(np.random.randint(100) + 1)
        to_set = np.random.random(size) * pq.millisecond # let it be ms
    if attr[1] == np.ndarray:
        to_set = np.array(to_set, dtype=attr[3])
    if attr[1] == str:
        to_set = str(np.random.randint(100000))
    if attr[1] == int:
        to_set = np.random.randint(100)
    if attr[1] == datetime:
        to_set = datetime.datetime.now()
    return to_set

def fake_NEO(obj_type="Block", cascade=True, follow_links=True):
    """ Create a fake NEO object of a given type. Follows one-to-many 
    and many-to-many relationships if cascade. RC, when requested cascade, will
    not create RGCs to avoid dead-locks.

    follow_links - an internal variable, indicates whether to create objects 
    with 'implicit' relationships, to avoid duplications. Do not use it. """
    kwargs = {}
    for attr in classes_necessary_attributes[obj_type]:
        kwargs[attr[0]] = get_fake_value(attr)
    for attr in classes_recommended_attributes[obj_type]:
        kwargs[attr[0]] = get_fake_value(attr)
    obj = class_by_name[obj_type](**kwargs)
    if cascade:
        if obj_type == "Block":
            follow_links = False
        if one_to_many_reslationship.has_key(obj_type):
            rels = one_to_many_reslationship[obj_type]
            if not follow_links and implicit_reslationship.has_key(obj_type):
                for i in implicit_reslationship[obj_type]:
                    rels.pop(i)
            for child in rels:
                setattr(obj, child.lower() + "s", [fake_NEO(child, cascade, 
                    follow_links)])
        if obj_type = "RecordingChannelGroup":
            for child in many_to_many_reslationship[obj_type]:
                setattr(obj, child.lower() + "s", [fake_NEO(child, cascade, 
                    follow_links)])
    if obj_type = "Block": # need to manually create 'implicit' connections
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
    for i, a in enumerate(TEST_ANNOTATIONS):
        obj.annotate(i=a)
    return obj


class HDF5Commontests(BaseTestIO, unittest.TestCase):
    ioclass = NeoHdf5IO
    files_to_test = [  ]
    files_to_download =  [   ]
    
    @unittest.skipUnless(have_hdf5, "requires PyTables")
    def setUp(self):
        BaseTestIO.setUp(self)


class hdf5ioTest(unittest.TestCase):
    """
    Tests for the hdf5 library.
    """
    
    @unittest.skipUnless(have_hdf5, "requires PyTables")
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
        def assert_attr(self, obj, replica, attr_name)
            a1 = md5(getattr(obj, attr_name)).hexdigest()
            self.assertTrue(hasattr(replica, attr_name))
            a2 = md5(getattr(replica, attr_name)).hexdigest()
            self.assertEqual(a1, a2,
                "Attribute %s for class %s wasn't saved properly" %
                 (attr_name, obj_type))
        iom = NeoHdf5IO(filename=self.test_file)
        for obj_type in class_by_name.keys():
            obj = fake_NEO(obj_type, cascade=False)
            iom.save(obj)
            self.assertTrue(hasattr(obj, "hdf5_path"))
            replica = iom.get(obj.hdf5_path, cascade=False)
            for attr in classes_necessary_attributes[obj_type]:
                self.assert_attr(obj, replica, attr[0])
            for attr in classes_recommended_attributes[obj_type]:
                if hasattr(obj, attr[0]):
                    self.assert_attr(obj, replica, attr[0])
            if hasattr(obj, "annotations"):
                self.assertTrue(hasattr(replica, "annotations"))
                for k, v in obj.annotations:
                    self.assertTrue(hasattr(replica.annotations, k))
                    self.assertEqual(replica.annotations[k], v)

    def test_relations(self):
        """ make sure the change in relationships is saved properly in the file,
        including correct M2M, no redundancy etc. RC -> RCG not tested."""
        def assert_children(self, obj, replica):
            obj_type = name_by_class[obj]
            self.assertEqual(md5(str(obj)).hexdigest(), md5.(str(replica)).hexdigest())
            if one_to_many_reslationship.has_key(obj_type):
                rels = one_to_many_reslationship[obj_type]
                if obj_type = "RecordingChannelGroup":
                    rels += many_to_many_reslationship[obj_type]
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
        # self.assertRaises(LookupError, iom.save(obj.wrong_path))
        # changes!!! in attr AS WELL AS in relations!!
        # test annotations
        # test naming - paths
        # unicode!!
        # add a child, then remove, then check it's removed

if __name__ == '__main__':
    unittest.main()



