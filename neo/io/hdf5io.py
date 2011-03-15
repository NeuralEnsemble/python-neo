"""
README
================================================================================
This is the implementation of the NEO IO for the HDF5 files.
http://neuralensemble.org/

IO dependencies:
 - NEO.core
 - BaseIO
 - types
 - warnings
 - numpy
 - pytables
 - quantities


Quick reference:
================================================================================
Class IOManager() with methods get(), save(), delete() is implemented. This 
class represents a connection manager with the HDF5 file with the possibility
to put (save()) or retrieve (get()) runtime NEO objects from the file.

Start by initializing IO:

>>> from hdf5io import IOManager
>>> iom = IOManager()
>>> iom
<hdf5io.IOManager object at 0x7f291ebe6810>

The file is created automatically (path/filename can be changed in "settings" 
option). So you can also do 

>>> iom = IOManager(filename="myfile.h5")

Now you may save any of your neo object into the file (assuming your NEO objects
are in the python path):

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

The weakness of the HDF5 format for NEO:
================================================================================
We need to keep in mind that NEO is more than a pure tree structure. So it can't
be fully represented in a HDF5 file tree. To minimise inconsistencies, there 
were "hard links" (http://www.pytables.org/docs/manual/ch03.html#LinksTutorial)
implemented for the following types of relations:
    - recordingchannelgroup <- analogsignalarray
    - recordingchannel <- analogsignal
    - recordingchannel <- irsaanalogsignal
    - unit <- spiketrain
    - unit <- spike
    - unit <- recordingchannel
These "implicit" relations are NOT loaded when you run the "get" function.

Plans for future extensions:
================================================================================
#FIXME - implement logging mechanism (probably in general for NEO)
#FIXME - implement caching?
#FIXME - implement actions history (probably in general for NEO)
#FIXME - extend BaseIO compliance
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

import sys, os
sys.path.append(os.path.abspath('../..'))
from neo.core import *
from baseio import BaseIO
import types
import warnings
import tables as tb
import numpy as np
import quantities as pq

"""
SETTINGS:
path:           path to the HDF5 file.
filename:       the name of the HDF5 file. For opening used together with the 
                file.
cascade:        If 'True' all children are retrieved when get(object) is called.
lazy:           If 'True' data (arrays) is retrieved when get(object) is called. 
"""
settings = {'path': "", 'filename': "neo.h5", 'cascade': True, 'lazy': True}
meta_objects = ["block", "segment", "event", "eventarray", "epoch", "epocharray", \
    "unit", "spiketrain", "analogsignal", "analogsignalarray", \
    "irsaanalogsignal", "spike", "recordingchannelgroup", "recordingchannel"]
meta_classnames = {
    "block": Block,
    "segment": Segment,
    "event": Event,
    "eventarray": EventArray,
    "epoch": Epoch,
    "epocharray": EpochArray,
    "unit": Unit,
    "spiketrain": SpikeTrain,
    "analogsignal": AnalogSignal,
    "analogsignalarray": AnalogSignalArray,
    "irsaanalogsignal": IrregularySampledSignal,
    "spike": Spike,
    "recordingchannelgroup": RecordingChannelGroup,
    "recordingchannel": RecordingChannel}
# attribute name
meta_attributes = {
    "block": ['name', 'filedatetime', 'index'],
    "segment": ['name', 'filedatetime', 'index'],
    "event": ['time', 'label'],
    "eventarray": [],
    "epoch": ['time', 'label', 'duration'],
    "epocharray": [],
    "unit": ['name'],
    "spiketrain": ['t_start', 't_stop'],
    "analogsignal": ['name', 'sampling_rate', 't_start'],
    "analogsignalarray": ['sampling_rate', 't_start'],
    "irsaanalogsignal": ['name', 'channel_name'],
    "spike": ['time', 'sampling_rate', 'left_sweep'],
    "recordingchannelgroup": ['name'],
    "recordingchannel": ['name', 'index']}
# explicit relations: relation type, name of NEO attribute (getter), 
# neo attribute name (to set) 
meta_exp_relations = {
    "block": [
        ['segment', '_segments', '_segments'], 
        ['recordingchannelgroup', '_recordingchannelgroups', '_recordingchannelgroups']],
    "segment": [
        ['analogsignal', '_analogsignals', '_analogsignals'],
        ['irsaanalogsignal', '_irsaanalogsignals', '_irsaanalogsignals'],
        ['analogsignalarray', '_analogsignalarrays', '_analogsignalarrays'],
        ['spiketrain', '_spiketrains', '_spiketrains'],
        ['spike', '_spikes', '_spikes'],
        ['event', '_events', '_events'],
        ['eventarray', '_eventarrays', '_eventarrays'],  
        ['epoch', '_epochs', '_epochs'],
        ['epocharray', '_epocharrays', '_epocharrays']],
    "recordingchannelgroup": [
        ['recordingchannel', '_recordingchannels', '_recordingchannels']],
    "recordingchannel": [
        ['unit', '_units', '_units']]}
# implicit relations (using hard links)
meta_imp_relations = {
    "recordingchannelgroup": [
        ['analogsignalarray', '_analogsignalarrays', '_analogsignalarrays']],
    "recordingchannel": [
        ['analogsignal', '_analogsignals', '_analogsignals'],
        ['irsaanalogsignal', '_irsaanalogsignals', '_irsaanalogsignals']],  
    "unit": [
        ['spiketrain', '_spiketrains', '_spiketrains'],
        ['spike', '_spikes', '_spikes'],
        ['recordingchannel', '_recordingchannels', '_recordingchannels']]}
# array name, default value, neo attribute name (get), neo attribute name (set)
meta_arrays = {
    "eventarray": [
        ["times", np.zeros(1) * pq.millisecond, "times", "times"], \
        ["labels", np.array("", dtype="a100"), "labels", "labels"]],
    "epocharray": [
        ["times", np.zeros(1) * pq.millisecond, "times", "times"], \
        ["labels", np.array("", dtype="a100"), "labels", "labels"], \
        ["durations", np.zeros(1) * pq.millisecond, "durations", "durations"]],
    "spiketrain": [
        ["spike_times", np.zeros(1) * pq.millisecond, "times", "times"], \
        ["waveforms", np.zeros([1, 1, 1]) * pq.millisecond, "waveforms", "waveforms"]],
    "analogsignal": [
        ["signal", np.zeros(1) * pq.millivolt, "signal", "signal"]],
    "irsaanalogsignal": [
        ["signal", np.zeros(1) * pq.millisecond, "signal", "signal"],
        ["times", np.zeros(1) * pq.millisecond, "times", "times"]],
    "analogsignalarray": [
        ["signals", np.zeros([1, 1]) * pq.millisecond, "signal", "signal"]],
    "spike": [
        ["waveform", np.zeros([1, 1]) * pq.millisecond, "waveform", "waveform"]]}

def _func_wrapper(func):
    try:
        return func
    except IOError:
        raise IOError("There is no connection with the file or the file was recently corrupted. \
            Please reload the IO manager.")


#---------------------------------------------------------------
# Basic I/O manager, implementing basic I/O functionality
#---------------------------------------------------------------

class IOManager(BaseIO):
    """
    The IO Manager is the core I/O class for HDF5 / NEO. It handles the 
    connection with the HDF5 file, and uses PyTables for data operations. Use
    this class to get (load), insert or delete NEO objects to HDF5 file.
    """
    def __init__(self, connect=True, path=settings['path'], \
            filename=settings['filename']):
        self._init_base_io()
        self.connected = False
        if connect:
            self.connect(path=path, filename=filename)

    def _read_entity(self, path, cascade=True, lazy=False):
        """
        Wrapper for base io "reader" functions.
        """
        return self.get(path, cascade, lazy)

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
        self.supported_objects = meta_objects
        self.readable_objects = meta_objects
        self.writeable_objects = meta_objects
        self.name = 'HDF5 IO'
        # wraps for Base IO functions
        for obj_type in self.readable_objects:
            self.__setattr__("read_" + obj_type, self._read_entity)
        for obj_type in self.writeable_objects:
            self.__setattr__("write_" + obj_type, self._write_entity)

    #-------------------------------------------
    # IO connectivity / Session management
    #-------------------------------------------

    def connect(self, path=settings['path'], filename=settings['filename']):
        """
        Opens / initialises new HDF5 file.
        We rely on PyTables and keep all session management staff there.
        """
        if not self.connected:
            try:
                if tb.isHDF5File(path + filename):
                    self._data = tb.openFile(path + filename, mode = "a", title = filename)
                    self.connected = True
                else:
                    raise TypeError("The file specified is not an HDF5 file format.")
            except IOError:
                # create a new file if specified file not found
                self._data = tb.openFile(path + filename, mode = "w", title = filename)
                self.connected = True
            except:
                raise NameError("Incorrect file path, couldn't find or create a file.")
        else:
            print "Already connected."

    def close(self):
        """
        Closes the connection.
        """
        self._data.close()
        self.connected = False

    #-------------------------------------------
    # some internal IO functions
    #-------------------------------------------

    def _get_type_by_obj(self, obj):
        """
        Returns the type of the object (string) depending on the object given.
        """
        for obj_type in meta_objects:
            if isinstance(obj, meta_classnames[obj_type]):
                return obj_type
        return None

    def _get_class_by_node(self, node):
        """
        Returns the type of the object (string) depending on node.
        """
        try:
            obj_type = node._f_getAttr("_type")
            return meta_classnames[obj_type]
        except:
            # that's an alien node
            return None

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
        """
        Saves changes of a given object to the file. Saves object as new if the 
        given one is not in the file yet.
        """
        obj_type = self._get_type_by_obj(obj)
        if obj_type:
            if hasattr(obj, "hdf5_path"):
                try:
                    node = self._data.getNode(obj.hdf5_path)
                except tb.NoSuchNodeError:  # create a new node?
                    raise LookupError("A given object has a path " + \
                        str(obj.hdf5_path) + " attribute, but such an object \
                        does not exist in the file. Please correct these values \
                        or delete this attribute (.__delattr__('hdf5_path')) \
                        to create a new object in the file.")
            else:
                # create new object
                node = self._data.createGroup(where, self._get_next_name(obj_type, where))
                node._f_setAttr("_type", obj_type)
            # we don't know which attributes have changed, so replace all
            for attr in meta_attributes[obj_type]:
                try:
                    obj_attr = obj.__getattribute__(attr)
                    node._f_setAttr(attr, obj_attr)
                    if hasattr(obj_attr, "dimensionality"):
                        for un in obj_attr.dimensionality.items():
                            node._f_setAttr(str(attr) + "__" + un[0].name, un[1])
                except:
                    # object does not have this attribute
                    pass
            # here we could add metadata-defined attributes processing
            # processing arrays
            if meta_arrays.has_key(obj_type) and not lazy:
                for arr in meta_arrays[obj_type]:
                    if hasattr(obj, arr[2]) and (getattr(obj, arr[2]) is not None):
                        obj_array = getattr(obj, arr[2], False)
                        # we trust this is array
                        if obj_array.size == 0:
                            obj_array = arr[1]
                        if hasattr(obj, "hdf5_path"):
                            arr_path = str(obj.hdf5_path)
                        else:
                            arr_path = node._v_pathname
                        # we try to create new array first, so not to loose the 
                        # data in case of failure
                        new_arr = self._data.createArray(arr_path, arr[0] + "__temp", obj_array)
                        if hasattr(obj_array, "dimensionality"):
                            for un in obj_array.dimensionality.items():
                                new_arr._f_setAttr("unit__" + un[0].name, un[1])
                        try:
                            self._data.removeNode(arr_path, arr[0])
                        except:
                            # there is no array yet or object is new, so just proceed
                            pass
                        # and here rename it back to the original
                        self._data.renameNode(arr_path, arr[0], name=arr[0] + "__temp")
            # process downstream relations: first process explicit
            # relations to physically save all objects, then implicit
            # to save appropriate links
            if meta_exp_relations.has_key(obj_type) and cascade:
                # FIXME removed objects should be removed?
                for r in meta_exp_relations[obj_type]:
                    if hasattr(obj, r[1]):
                        lst = obj.__getattribute__(r[1])
                        if getattr(lst, '__iter__', False):
                            try:
                                ch = self._data.getNode(node, r[1])
                            except tb.NoSuchNodeError:
                                ch = self._data.createGroup(node, r[1])
                            for i in lst:
                                self.save(i, where=ch._v_pathname)
            if meta_imp_relations.has_key(obj_type) and cascade:
                # FIXME removed objects should be removed?
                for r in meta_imp_relations[obj_type]:
                    try:
                        ch = self._data.getNode(node, r[1])
                    except tb.NoSuchNodeError:
                        ch = self._data.createGroup(node, r[1])
                    try:
                        lst = obj.__getattribute__(r[1])
                        for i in lst:
                            target = self._data.getNode(i.hdf5_path, i.hdf5_name)
                            self.createHardLink(ch._v_pathname, i.hdf5_name, target)
                    except tb.NoSuchNodeError:
                        # enlisted nodes are not in the file
                        pass
                    except:
                        # object doesn't have required attributes or doesn't exist yet
                        pass
            self._update_path(obj, node)
        else:
            raise TypeError("Given object " + str(obj) + " is not a NEO object instance.")


    @_func_wrapper
    def get(self, path, cascade=True, lazy=False):
        """
        Returns a requested NEO object as instance of NEO class.
        """
        try:
            node = self._data.getNode(path)
        except tb.NoSuchNodeError:
            # create a new node?
            raise LookupError("There is no valid object with a given path " +\
                str(path) + " . Please give correct path or just browse the file \
                (e.g. IOManager()._data.root.<Block>._segments...) to find an \
                appropriate name.")
        classname = self._get_class_by_node(node)
        if classname:
            obj = classname()
            obj_type = self._get_type_by_obj(obj)
            # set up attributes
            self._update_path(obj, node)
            # may seem ugly, but generic for any possible quantity-enabled attribute
            lst = node._v_attrs._f_list(attrset='user')
            ext = [] # a list of attribute names having quantity
            for i in lst:
                if i.find("__") > 0:
                    val = i[:i.find("__")]
                    if not val in ext: ext.append(val)
            for attr in lst:
                if attr in ext:  # attribute has a quantity
                    unit = ""
                    for bla in lst:
                        if bla.startswith(attr + "__"):
                            unit += " * " + bla[bla.find("__") + 2:] + " ** " + str(node._f_getAttr(bla))
                    unit = unit.replace(" * ", "", 1)
                    setattr(obj, attr, pq.Quantity(node._f_getAttr(attr), unit))
                elif attr[:attr.find("__")] not in ext:
                    setattr(obj, attr, node._f_getAttr(attr))
            # load arrays
            if not lazy and meta_arrays.has_key(obj_type):
                for arr in meta_arrays[obj_type]:
                    try:
                        h_arr = self._data.getNode(node, arr[0])
                        # setting up quantities
                        unit = ""
                        for attr in h_arr._v_attrs._f_list(attrset='user'):
                            if attr.startswith("unit__"):
                                unit += " * " + str(attr[6:]) + " ** " + str(h_arr._f_getAttr(attr))
                        unit = unit.replace(" * ", "", 1)
                        setattr(obj, arr[3], pq.Quantity(h_arr.read(), unit))
                    except tb.NoSuchNodeError:
                        warnings.warn("An array of type '" + arr[0] + "' for object " \
                            + node._v_pathname + " has not been found in the file.")
                    except:
                        # array was damaged or manually modified
                        raise StandardError("Requested array " + str(path) + \
                            " can't be loaded (its quantities information is \
                            wrong?). Please correct and load an object again.")
            # load relations
            if cascade and meta_exp_relations.has_key(obj_type):
                for rel in meta_exp_relations[obj_type]:
                    relatives = []
                    try:
                        container = self._data.getNode(node, rel[1])
                        for n in self._data.listNodes(container):
                            try:
                                if n._f_getAttr("_type") == rel[0]:
                                    relatives.append(self.get(n._v_pathname))
                            except:
                                # alien nodes
                                pass
                    except tb.NoSuchNodeError:
                        # there is no relatives of that type, so just proceed
                        pass
                    setattr(obj, rel[2], relatives)
            # we do not load implicit relations so far
            return obj
        else:
            raise LookupError("The requested object with the path " + str(path) +\
                " exists, but is not of a NEO type. Please check the '_type' attribute.")

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
        print "This is a neo.HDF5 file. it contains:"
        info = {}
        info = info.fromkeys(meta_objects, 0)
        for node in self._data.walkNodes():
            try:
                t = node._f_getAttr("_type")
                info[t] += 1
            except:
                # node is not of NEO type
                pass
        return info


