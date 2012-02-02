"""
README
================================================================================
This is the implementation of the NEO IO for the HDF5 files.
http://neuralensemble.org/

IO dependencies:
 - NEO
 - types
 - warnings
 - numpy
 - pytables >= 2.2
 - quantities


Quick reference:
================================================================================
Class NeoHdf5IO() with methods get(), save(), delete() is implemented. This 
class represents a connection manager with the HDF5 file with the possibility
to put (save()) or retrieve (get()) runtime NEO objects from the file.

Start by initializing IO:

>>> from hdf5io import NeoHdf5IO
>>> iom = NeoHdf5IO()
>>> iom
<hdf5io.NeoHdf5IO object at 0x7f291ebe6810>

The file is created automatically (filename can be changed in "settings" 
option). So you can also do 

>>> iom = NeoHdf5IO(filename="myfile.h5")

Now you may save any of your neo object into the file (assuming your NEO objects
are in the pythonpath):

>>> b = Block()
>>> iom.write_block(b)

or just do 

>>> iom.save(b)

After you stored an object it receives a unique "path" in the hdf5 file. This is 
exactly the place in the HDF5 hierarchy, where it was written. This information
is now accessible by "hdf5_path" property:

>>> b.hdf5_path
'/block_0'

You may save more complicated NEO stuctures, with relations and arrays:

>>> import numpy as np
>>> import quantities as pq
>>> s = Segment()
>>> b._segments.append(s)
>>> a1 = AnalogSignal(signal=np.random.rand(300), t_start=42*ms)
>>> s._analogsignals.append(a1)

and then

>>> iom.write_block(b)

or just

>>> iom.save(b)

If you already have hdf5 file in NEO format, or you just created one, then you 
may want to read NEO data (providing the path to what to read):

>>> b1 = iom.read_block("/block_0")
>>> b1
<neo.core.block.Block object at 0x34ee590>

or just use 

>>> b1 = iom.get("/block_0")

You may notice, by default the reading function retrieves all available data, 
with all downstream relations and arrays:

>>> b1._segments
[<neo.core.segment.Segment object at 0x34ee750>]
>>> b1._segments[0]._analogsignals[0].signal
array([  3.18987819e-01,   1.08448284e-01,   1.03858980e-01,
        ...
         3.78908705e-01,   3.08669731e-02,   9.48965785e-01]) * dimensionless

When you need to save time and performance, you may load an object without
relations

>>> b2 = iom.get("/block_0", cascade=False)
>>> b2._segments
[]

and/or even without arrays

>>> a2 = iom.get("/block_0/_segments/segment_0/_analogsignals/analogsignal_0", 
lazy=True)
>>> a2.signal
[]

These functions return "pure" NEO objects. They are completely "detached" from 
the HDF5 file - changes to the runtime objects will not cause any changes in the
file:

>>> a2.t_start
array(42.0) * ms
>>> a2.t_start = 32 * pq.ms
>>> a2.t_start
array(32.0) * ms
>>> iom.get("/block_0/_segments/segment_0/_analogsignals/analogsignal_0").t_start
array(42.0) * ms

However, if you want to work directly with HDF5 storage making instant 
modifications, you may use the native PyTables functionality, where all objects
are accessible through "<IO_manager_inst>._data.root":

>>> iom._data.root
/ (RootGroup) 'neo.h5'
  children := ['block_0' (Group)]
>>> b3 = iom._data.root.block_0
>>> b3
/block_0 (Group) ''
  children := ['_recordingchannelgroups' (Group), '_segments' (Group)]

To understand more about this "direct" way of working with data, please refer to
http://www.pytables.org/
Finally, you may get an overview of the contents of the file by running

>>> iom.get_info()
This is a neo.HDF5 file. it contains:
{'spiketrain': 0, 'irsaanalogsignal': 0, 'analogsignalarray': 0, 
'recordingchannelgroup': 0, 'eventarray': 0, 'analogsignal': 1, 'epoch': 0, 
'unit': 0, 'recordingchannel': 0, 'spike': 0, 'epocharray': 0, 'segment': 1, 
'event': 0, 'block': 1}


The general structure of the file:
================================================================================

\'Block_1'
\
\'Block_2'
    \
     \---'_recordingchannelgroups'
     \           \
     \            \---'RecordingChannelGroup_1'
     \            \
     \            \---'RecordingChannelGroup_2'
     \                       \
     \                        \---'_recordingchannels'
     \                                   \
     \                                    \---'RecordingChannel_1'
     \                                    \
     \                                    \---'RecordingChannel_2'
     \                                           \
     \                                            \---'_units'
     \                                                   \
     \                                                    \---'Unit_1'
     \                                                    \
     \                                                    \---'Unit_2'
     \                        
     \---'_segments'
            \
             \--'Segment_1'
             \
             \--'Segment_2'
                    \
                     \---'_epochs'
                     \       \
                     \        \---'Epoch_1'
                     \
                     \---'_epochs'
                   etc.

Plans for future extensions:
================================================================================
#FIXME - lazy load should be only for huge arrays, but not for all Quantities
#FIXME - implement logging mechanism (probably in general for NEO)
#FIXME - implement actions history (probably in general for NEO)
#FIXME - use global IDs for NEO objects (or even UUIDs?)
#FIXME - implement callbacks in functions for GUIs
#FIXME - no performance testing yet

IMPORTANT things:
================================================================================
1. Every NEO node object in HDF5 has a "_type" attribute. Please don't modify.
2. There are reserved attributes "unit__<quantity>" or "<name>__<quantity>" in
objects, containing quantities.
3. Don't use "__" in attribute names, as this symbol is reserved for quantities.


Author: asobolev
"""

from __future__ import absolute_import
from ..core import *
from ..test.tools import assert_neo_object_is_compliant
from ..description import *
from .baseio import BaseIO
from .tools import create_many_to_one_relationship
from tables import NoSuchNodeError as NSNE
import tables as tb
import numpy as np
import quantities as pq
import logging

import tables

#version checking
from distutils import version
if version.LooseVersion(tables.__version__) < '2.2':
    raise ImportError("your pytables version is too old to support NeoHdf5IO, you need at least 2.2 you have %s"%tables.__version__)


"""
SETTINGS:
filename:       the full path to the HDF5 file.
cascade:        If 'True' all children are retrieved when get(object) is called.
lazy:           If 'True' data (arrays) is retrieved when get(object) is called. 
"""
settings = {'filename': "neo.h5", 'cascade': True, 'lazy': True}

def _func_wrapper(func):
    try:
        return func
    except IOError:
        raise IOError("There is no connection with the file or the file was recently corrupted. \
            Please reload the IO manager.")


#---------------------------------------------------------------
# Basic I/O manager, implementing basic I/O functionality
#---------------------------------------------------------------
all_objects = list(class_by_name.values())
all_objects.remove(Block)# the order is important
all_objects = [Block]+all_objects

class NeoHdf5IO(BaseIO):
    """
    The IO Manager is the core I/O class for HDF5 / NEO. It handles the 
    connection with the HDF5 file, and uses PyTables for data operations. Use
    this class to get (load), insert or delete NEO objects to HDF5 file.
    """
    supported_objects = all_objects
    readable_objects    = all_objects
    writeable_objects   = all_objects
    read_params = dict( zip( all_objects, [ ]*len(all_objects)) )
    write_params = dict( zip( all_objects, [ ]*len(all_objects)) )
    name = 'Hdf5'
    extensions = [ 'h5', ]
    mode = 'file'
    
    def __init__(self, filename=settings['filename'], **kwargs):
        self._init_base_io()
        self.connected = False
        self.connect(filename=filename)

    def _read_entity(self, path="/", cascade=True, lazy=False):
        """
        Wrapper for base io "reader" functions.
        """
        ob = self.get(path, cascade, lazy)
        create_many_to_one_relationship(ob)
        return ob

    def _write_entity(self, obj, where="/", cascade=True, lazy=False):
        """
        Wrapper for base io "writer" functions.
        """
        self.save(obj, where, cascade, lazy)

    def _init_base_io(self):
        """
        Base io initialization.
        """
        self.is_readable = True
        self.is_writable = True
        self.supported_objects = class_by_name.keys()
        self.readable_objects = class_by_name.keys()
        self.writeable_objects = class_by_name.keys()
        self.name = 'HDF5 IO'
        # wraps for Base IO functions
        for obj_type in self.readable_objects:
            self.__setattr__("read_" + obj_type.lower(), self._read_entity)
        for obj_type in self.writeable_objects:
            self.__setattr__("write_" + obj_type.lower(), self._write_entity)

    #-------------------------------------------
    # IO connectivity / Session management
    #-------------------------------------------

    def connect(self, filename=settings['filename']):
        """
        Opens / initialises new HDF5 file.
        We rely on PyTables and keep all session management staff there.
        """
        if not self.connected:
            try:
                if tb.isHDF5File(filename):
                    self._data = tb.openFile(filename, mode = "a", title = filename)
                    self.connected = True
                else:
                    raise TypeError("The file specified is not an HDF5 file format.")
            except IOError:
                # create a new file if specified file not found
                self._data = tb.openFile(filename, mode = "w", title = filename)
                self.connected = True
            except:
                raise NameError("Incorrect file path, couldn't find or create a file.")
        else:
            logging.info("Already connected.")

    def close(self):
        """
        Closes the connection.
        """
        self._data.close()
        self.connected = False

    #-------------------------------------------
    # some internal IO functions
    #-------------------------------------------

    def _get_class_by_node(self, node):
        """
        Returns the type of the object (string) depending on node.
        """
        try:
            obj_type = node._f_getAttr("_type")
            return class_by_name[obj_type]
        except:
            return None # that's an alien node

    def _update_path(self, obj, node):
        setattr(obj, "hdf5_name", node._v_name)
        setattr(obj, "hdf5_path", node._v_pathname)

    def _get_next_name(self, obj_type, where):
        """
        Returns the next possible name within a given container (group)
        Expensive with large saves! Define other algorithm?
        """
        prefix = str(obj_type) + "_"
        nodes = []
        for node in self._data.listNodes(where):
            index = node._v_name[node._v_name.find(prefix) + len(prefix):]
            if len(index) > 0: 
                try:
                    nodes.append(int(index))
                except ValueError:
                    pass # index was changed by user, but then we don't care
        nodes.sort(reverse=True)
        if len(nodes) > 0:
            return prefix + str(nodes[0] + 1)
        else:
            return prefix + "0"

    #-------------------------------------------
    # general IO functions, for all NEO objects
    #-------------------------------------------

    @_func_wrapper
    def save(self, obj, where="/", cascade=True, lazy=False):
        """ Saves changes of a given object to the file. Saves object as new at 
        location "where" if it is not in the file yet.

        cascade: True/False process downstream relationships
        lazy: True/False process any quantity/ndarray attributes """

        def assign_attribute(obj_attr, attr_name):
            """ subfunction to serialize a given attribute """
            if isinstance(obj_attr, pq.Quantity) or isinstance(obj_attr, np.ndarray):
                if not lazy:
                    if obj_attr.size == 0:
                        atom = tb.Float64Atom(shape=(1,))
                        new_arr = self._data.createEArray(path, attr_name + "__temp", atom, shape=(0,), expectedrows=1)
                        #raise ValueError("A size of the %s of the %s has \
                        #    length zero and can't be saved." % 
                        #    (attr_name, path))
                    # we try to create new array first, so not to loose the 
                    # data in case of any failure
                    else:
                        new_arr = self._data.createArray(path, attr_name + "__temp", obj_attr)
                    if hasattr(obj_attr, "dimensionality"):
                        for un in obj_attr.dimensionality.items():
                            new_arr._f_setAttr("unit__" + un[0].name, un[1])
                    try:
                        self._data.removeNode(path, attr_name)
                    except:
                        pass # there is no array yet or object is new
                    self._data.renameNode(path, attr_name, name=attr_name + "__temp")
            elif not obj_attr == None:
                node._f_setAttr(attr_name, obj_attr)
            
        #assert_neo_object_is_compliant(obj)
        obj_type = name_by_class[obj.__class__]
        if hasattr(obj, "hdf5_path"): # this is an update case
            try:
                path = str(obj.hdf5_path)
                node = self._data.getNode(obj.hdf5_path)
            except NSNE:  # create a new node?
                raise LookupError("A given object has a path %s attribute, \
                    but such an object does not exist in the file. Please \
                    correct these values or delete this attribute \
                    (.__delattr__('hdf5_path')) to create a new object in \
                    the file." % path)
        else: # create new object                
            node = self._data.createGroup(where, self._get_next_name(obj_type, where))
            node._f_setAttr("_type", obj_type)
            path = node._v_pathname
        # processing attributes
        attrs = classes_necessary_attributes[obj_type] + classes_recommended_attributes[obj_type]
        for attr in attrs: # we checked already obj is compliant, loop over all safely
            if hasattr(obj, attr[0]): # save an attribute if exists
                assign_attribute(getattr(obj, attr[0]), attr[0])
        # not forget to save AS, ASA or ST - NEO "stars"
        if obj_type in classes_inheriting_quantities.keys():
            assign_attribute(obj, classes_inheriting_quantities[obj_type])
        if hasattr(obj, "annotations"): # annotations should be just a dict
            node._f_setAttr("annotations", getattr(obj, "annotations"))
        if one_to_many_relationship.has_key(obj_type) and cascade:
            rels = list(one_to_many_relationship[obj_type])
            if obj_type == "RecordingChannelGroup":
                rels += many_to_many_relationship[obj_type]
            for child_name in rels: # child_name like "Segment", "Event" etc.
                container = child_name.lower() + "s" # like "units"
                try:
                    ch = self._data.getNode(node, container)
                except NSNE:
                    ch = self._data.createGroup(node, container)
                saved = [] # keeps track of saved object names for removal
                for child in getattr(obj, container):
                    new_name = None
                    if hasattr(child, "hdf5_path") and hasattr(child, "hdf5_name"):
                        if not ch._v_pathname in child.hdf5_path:
                        # create a Hard Link as object exists already somewhere
                            target = self._data.getNode(child.hdf5_path)
                            new_name = self._get_next_name(name_by_class[child.__class__], ch._v_pathname)
                            self._data.createHardLink(ch._v_pathname, new_name, target)
                    self.save(child, where=ch._v_pathname)
                    if not new_name: new_name = child.hdf5_name
                    saved.append(new_name)
                for child in self._data.iterNodes(ch._v_pathname):
                    if child._v_name not in saved: # clean-up
                        self._data.removeNode(ch._v_pathname, child._v_name, recursive=True)
        # FIXME needed special processor for RC -> RCG
        self._update_path(obj, node)


    @_func_wrapper
    def get(self, path="/", cascade=True, lazy=False):
        """ Returns a requested NEO object as instance of NEO class. """

        def rem_duplicates(target, source, attr):
            """ removes duplicated objects in case a block is requested: for 
            RCGs, RCs and Units we remove duplicated ASAs, IrSAs, ASs, STs and
            Spikes if those were already initialized in Segment. """
            a = getattr(target, attr) # a container, e.g. "analogsignals"
            b = getattr(source, attr) # a container, e.g. "analogsignals"
            res = list(set(a) - set(b))
            res += list(set(b) -(set(b) - set(a)))
            setattr(target, attr, res)

        def fetch_attribute(attr_name):
            """ fetch required attribute from the corresp. node in the file """
            try:
                if attr[1] == pq.Quantity:
                    arr = self._data.getNode(node, attr_name)
                    units = ""
                    for unit in arr._v_attrs._f_list(attrset='user'):
                        if unit.startswith("unit__"):
                            units += " * " + str(unit[6:]) + " ** " + str(arr._f_getAttr(unit))
                    units = units.replace(" * ", "", 1)
                    if not lazy:
                        nattr = pq.Quantity(arr.read(), units)
                    else: # making an empty array
                        nattr = pq.Quantity(np.empty(tuple([0 for x in range(attr[2])])), units)
                elif attr[1] == np.ndarray:
                    if not lazy:
                        arr = self._data.getNode(node, attr_name)
                        nattr = np.array(arr.read(), attr[3])
                    else: # making an empty array
                        nattr = np.empty((0), attr[3])
                else:
                    nattr = node._f_getAttr(attr_name)
                    if attr[1] == str or attr[1] == int:
                        nattr = attr[1](nattr) # compliance with NEO attr types
            except (AttributeError, NSNE): # not assigned, continue
                nattr = None
            return nattr

        if path == "/": # this is just for convenience. Try to return any object
            found = False
            for n in self._data.listNodes(path):
                for obj_type in class_by_name.keys():
                    if obj_type.lower() in str(n._v_name).lower():
                        path = n._v_pathname
                        found = True
                if found: break
        try:
            if path == "/":
                raise ValueError() # root is not a NEO object
            node = self._data.getNode(path)
        except (NSNE, ValueError): # create a new node?
            raise LookupError("There is no valid object with a given path " +\
                str(path) + " . Please give correct path or just browse the file \
                (e.g. NeoHdf5IO()._data.root.<Block>._segments...) to find an \
                appropriate name.")
        classname = self._get_class_by_node(node)
        if not classname:
            raise LookupError("The requested object with the path " + str(path) +\
                " exists, but is not of a NEO type. Please check the '_type' attribute.")
        obj_type = name_by_class[classname]
        kwargs = {}
        # load attributes (inherited *-ed attrs are also here)
        attrs = classes_necessary_attributes[obj_type] + classes_recommended_attributes[obj_type]
        for i, attr in enumerate(attrs):
            attr_name = attr[0]
            nattr = fetch_attribute(attr_name)
            if nattr is not None:
                kwargs[attr_name] = nattr
        obj = class_by_name[obj_type](**kwargs) # instantiate new object
        self._update_path(obj, node) # set up HDF attributes: name, path
        try:
            setattr(obj, "annotations", node._f_getAttr("annotations"))
        except AttributeError: pass # not assigned, continue
        if lazy: # FIXME is this really needed?
            setattr(obj, "lazy_shape", "some shape should go here..")
        # load relationships
        if cascade:
            if one_to_many_relationship.has_key(obj_type):
                rels = list(one_to_many_relationship[obj_type])
                if obj_type == "RecordingChannelGroup":
                    rels += many_to_many_relationship[obj_type]
                for child in rels: # 'child' is like 'Segment', 'Event' etc.
                    relatives = []
                    container = self._data.getNode(node, child.lower() + "s")
                    for n in self._data.listNodes(container):
                        try:
                            if n._f_getAttr("_type") == child:
                                relatives.append(self.get(n._v_pathname, lazy=lazy))
                        except AttributeError: # alien node
                            pass # not an error
                    setattr(obj, child.lower() + "s", relatives)
        if cascade and obj_type == "Block": # this is a special case
            # We need to clean-up some duplicated objects
            for seg in obj.segments:
                for RCG in obj.recordingchannelgroups:
                    rem_duplicates(RCG, seg, "analogsignalarrays") # clean-up duplicate ASA
                    for RC in RCG.recordingchannels:
                        rem_duplicates(RC, seg, "analogsignals")
                        rem_duplicates(RC, seg, "irregularlysampledsignals")
                    for unit in RCG.units:
                        rem_duplicates(unit, seg, "spiketrains")
                        rem_duplicates(unit, seg, "spikes")
        # FIXME special processor for RC -> RCG
        return obj


    @_func_wrapper
    def delete(self, path, cascade=False):
        """
        Deletes an object in the file. Just a simple alternative of removeNode().
        """
        self._data.removeNode(path, recursive=cascade)

    @_func_wrapper
    def reset(self, obj):
        """
        Resets runtime changes made to the object. TBD.
        """
        pass

    @_func_wrapper
    def get_info(self):
        """
        Returns a quantitative information about the contents of the file.
        """
        logging.info("This is a neo.HDF5 file. it contains:")
        info = {}
        info = info.fromkeys(class_by_name.keys(), 0)
        for node in self._data.walkNodes():
            try:
                t = node._f_getAttr("_type")
                info[t] += 1
            except:
                # node is not of NEO type
                pass
        return info


