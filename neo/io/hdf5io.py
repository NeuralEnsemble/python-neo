# -*- coding: utf-8 -*-
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

>>> from neo.io.hdf5io import NeoHdf5IO
>>> iom = NeoHdf5IO('myfile.h5')
>>> iom
<hdf5io.NeoHdf5IO object at 0x7f291ebe6810>

Now you may save any of your neo objects into the file:

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
>>> b.segments.append(s)
>>> a1 = AnalogSignal(signal=np.random.rand(300), t_start=42*pq.ms)
>>> s.analogsignals.append(a1)

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
#FIXME - implement logging mechanism (probably in general for NEO)
#FIXME - implement actions history (probably in general for NEO)
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

# needed for python 3 compatibility
from __future__ import absolute_import

import logging
import uuid

#version checking
from distutils import version

import numpy as np
import quantities as pq

# check tables
try:
    import tables as tb
except ImportError as err:
    HAVE_TABLES = False
    TABLES_ERR = err
else:
    if version.LooseVersion(tb.__version__) < '2.2':
        HAVE_TABLES = False
        TABLES_ERR = ImportError("your pytables version is too old to " +
                                 "support NeoHdf5IO, you need at least 2.2. " +
                                 "You have %s" % tb.__version__)
    else:
        HAVE_TABLES = True
        TABLES_ERR = None

from neo.core import Block, objectlist, objectnames, class_by_name
from neo.io.baseio import BaseIO
from neo.io.tools import LazyList

logger = logging.getLogger("Neo")


def _func_wrapper(func):
    try:
        return func
    except IOError:
        raise IOError("There is no connection with the file or the file was recently corrupted. \
            Please reload the IO manager.")


#---------------------------------------------------------------
# Basic I/O manager, implementing basic I/O functionality
#---------------------------------------------------------------

# Types where an object might have to be loaded multiple times to create
# all realtionships
complex_relationships = ["Unit", "Segment", "RecordingChannel"]

# Arrays node names for lazy shapes
lazy_shape_arrays = {'SpikeTrain': 'times', 'Spike': 'waveform',
                     'AnalogSignal': 'signal',
                     'AnalogSignalArray': 'signal',
                     'EventArray': 'times', 'EpochArray': 'times'}


class NeoHdf5IO(BaseIO):
    """
    The IO Manager is the core I/O class for HDF5 / NEO. It handles the
    connection with the HDF5 file, and uses PyTables for data operations. Use
    this class to get (load), insert or delete NEO objects to HDF5 file.

    dtype: The data type to use when creating the underlying PyTable EArrays
        storing the data. If None, the data type of the runtime data arrays
        will be used. Defaults to `np.float64` to keep old behavior.
    """
    supported_objects = objectlist
    readable_objects = objectlist
    writeable_objects = objectlist
    read_params = dict(zip(objectlist, [] * len(objectlist)))
    write_params = dict(zip(objectlist, [] * len(objectlist)))
    name = 'NeoHdf5 IO'
    extensions = ['h5']
    mode = 'file'
    is_readable = True
    is_writable = True

    def __init__(self, filename=None, array_dtype=np.float64, **kwargs):
        if not HAVE_TABLES:
            raise TABLES_ERR
        BaseIO.__init__(self, filename=filename)
        self.array_dtype = array_dtype
        self.connected = False
        self.objects_by_ref = {}  # Loaded objects by reference id
        self.parent_paths = {}  # Tuples of (Segment, other parent) paths
        self.name_indices = {}
        if filename:
            self.connect(filename=filename)

    def _read_entity(self, path="/", cascade=True, lazy=False):
        """
        Wrapper for base io "reader" functions.
        """
        ob = self.get(path, cascade, lazy)
        if cascade and cascade != 'lazy':
            ob.create_many_to_one_relationship()
        return ob

    def _write_entity(self, obj, where="/", cascade=True, lazy=False):
        """
        Wrapper for base io "writer" functions.
        """
        self.save(obj, where, cascade, lazy)

    #-------------------------------------------
    # IO connectivity / Session management
    #-------------------------------------------

    def connect(self, filename):
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
                    raise TypeError('"%s" is not an HDF5 file format.' % filename)
            except IOError:
                # create a new file if specified file not found
                self._data = tb.openFile(filename, mode = "w", title = filename)
                self.connected = True
            except:
                raise NameError("Incorrect file path, couldn't find or create a file.")
            self.objects_by_ref = {}
            self.name_indices = {}
        else:
            logger.info("Already connected.")

    def close(self):
        """
        Closes the connection.
        """
        self.objects_by_ref = {}
        self.parent_paths = {}
        self.name_indices = {}
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
        setattr(obj, "hdf5_path", node._v_pathname)

    def _get_next_name(self, obj_type, where):
        """
        Returns the next possible name within a given container (group)
        """
        if not (obj_type, where) in self.name_indices:
            self.name_indices[(obj_type, where)] = 0

        index_num = self.name_indices[(obj_type, where)]
        prefix = str(obj_type) + "_"
        if where + '/' + prefix + str(index_num) not in self._data:
            self.name_indices[(obj_type, where)] = index_num + 1
            return prefix + str(index_num)

        nodes = []
        for node in self._data.iterNodes(where):
            index = node._v_name[node._v_name.find(prefix) + len(prefix):]
            if len(index) > 0:
                try:
                    nodes.append(int(index))
                except ValueError:
                    pass # index was changed by user, but then we don't care
        nodes.sort(reverse=True)
        if len(nodes) > 0:
            self.name_indices[(obj_type, where)] = nodes[0] + 2
            return prefix + str(nodes[0] + 1)
        else:
            self.name_indices[(obj_type, where)] = 1
            return prefix + "0"

    #-------------------------------------------
    # general IO functions, for all NEO objects
    #-------------------------------------------

    @_func_wrapper
    def save(self, obj, where="/", cascade=True, lazy=False):
        """ Saves changes of a given object to the file. Saves object as new at
        location "where" if it is not in the file yet. Returns saved node.

        cascade: True/False process downstream relationships
        lazy: True/False process any quantity/ndarray attributes """

        def assign_attribute(obj_attr, attr_name, path, node):
            """ subfunction to serialize a given attribute """
            if isinstance(obj_attr, pq.Quantity) or isinstance(obj_attr, np.ndarray):
                if not lazy:
                    # we need to simplify custom quantities
                    if isinstance(obj_attr, pq.Quantity):
                        for un in obj_attr.dimensionality.keys():
                            if not un.name in pq.units.__dict__ or \
                                    not isinstance(pq.units.__dict__[un.name], pq.Quantity):
                                obj_attr = obj_attr.simplified
                                break

                    # we try to create new array first, so not to loose the
                    # data in case of any failure
                    if obj_attr.size == 0:
                        np_type = obj_attr.dtype if self.array_dtype is None else self.array_dtype
                        atom = tb.Atom.from_dtype(np.dtype((np_type, (1, ))))
                        new_arr = self._data.createEArray(path, attr_name + "__temp", atom, shape=(0,), expectedrows=1)
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
            elif obj_attr is not None:
                node._f_setAttr(attr_name, obj_attr)

        #assert_neo_object_is_compliant(obj)
        obj_type = obj.__class__.__name__
        if self._data.mode != 'w' and hasattr(obj, "hdf5_path"): # this is an update case
            path = str(obj.hdf5_path)
            try:
                node = self._data.getNode(obj.hdf5_path)
            except tb.NoSuchNodeError:  # create a new node?
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
         # Initialize empty parent paths
        if len(getattr(obj, '_single_parent_containers', [])) > 1:
            for par_cont in obj._single_parent_containers:
                node._f_setAttr(par_cont, '')
        # we checked already obj is compliant, loop over all safely
        for attr in obj._all_attrs:
            if hasattr(obj, attr[0]): # save an attribute if exists
                assign_attribute(getattr(obj, attr[0]), attr[0], path, node)
            # not forget to save AS, ASA or ST - NEO "stars"
        if hasattr(obj, '_quantity_attr'):
            assign_attribute(obj, obj._quantity_attr, path, node)
        if hasattr(obj, "annotations"): # annotations should be just a dict
            node._f_setAttr("annotations", getattr(obj, "annotations"))
        node._f_setAttr("object_ref", uuid.uuid4().hex)
        if cascade:
            # container is like segments, spiketrains, etc.
            for container in getattr(obj, '_child_containers', []):
                try:
                    ch = self._data.getNode(node, container)
                except tb.NoSuchNodeError:
                    ch = self._data.createGroup(node, container)
                saved = []  # keeps track of saved object names for removal
                for child in getattr(obj, container):
                    new_name = None
                    child_node = None
                    if hasattr(child, "hdf5_path"):
                        if not child.hdf5_path.startswith(ch._v_pathname):
                        # create a Hard Link if object exists already somewhere
                            try:
                                target = self._data.getNode(child.hdf5_path)
                                new_name = self._get_next_name(
                                    child.__class__.__name__, ch._v_pathname)
                                if not hasattr(ch, new_name):  # Only link if path does not exist
                                    child_node = self._data.createHardLink(ch._v_pathname, new_name, target)
                            except tb.NoSuchNodeError:
                                pass
                    if child_node is None:
                        child_node = self.save(child, where=ch._v_pathname)

                    if len(child._single_parent_containers) > 1:
                        child_node._f_setAttr(obj_type.lower(), path)
                    for par_cont in child._multi_parent_containers:
                        parents = []
                        if par_cont in child_node._v_attrs:
                            parents = child_node._v_attrs[par_cont]
                        parents.append(path)
                        child_node._f_setAttr(par_cont, parents)
                    if not new_name:
                        new_name = child.hdf5_path.split('/')[-1]
                    saved.append(new_name)
                for child in self._data.iterNodes(ch._v_pathname):
                    if child._v_name not in saved: # clean-up
                        self._data.removeNode(ch._v_pathname, child._v_name, recursive=True)

        self._update_path(obj, node)
        return node

    def _get_parent(self, path, ref, parent_type):
        """ Return the path of the parent of type "parent_type" for the object
        in "path" with id "ref". Returns an empty string if no parent extists.
        """
        parts = path.split('/')

        if (parent_type.lower() == 'block' or
                parts[-4] == parent_type.lower() + 's'):
            return '/'.join(parts[:-2])

        object_folder = parts[-2]
        parent_folder = parts[-4]
        if parent_folder in ('recordingchannels', 'units'):
            block_path = '/'.join(parts[:-6])
        else:
            block_path = '/'.join(parts[:-4])

        if parent_type.lower() in ('recordingchannel', 'unit'):
            # We need to search all recording channels
            path = block_path + '/recordingchannelgroups'
            for n in self._data.iterNodes(path):
                if not '_type' in n._v_attrs:
                    continue
                p = self._search_parent(
                    '%s/%ss' % (n._v_pathname, parent_type.lower()),
                    object_folder, ref)
                if p != '':
                    return p
            return ''

        if parent_type.lower() == 'segment':
            path = block_path + '/segments'
        elif parent_type.lower() in ('recordingchannelgroup',
                                     'recordingchannelgroups'):
            path = block_path + '/recordingchannelgroups'
        else:
            return ''

        return self._search_parent(path, object_folder, ref)

    def _get_rcgs(self, path, ref):
        """ Get RecordingChannelGroup parents for a RecordingChannel
        """
        parts = path.split('/')
        object_folder = parts[-2]
        block_path = '/'.join(parts[:-4])
        path = block_path + '/recordingchannelgroups'
        return self._search_parent(path, object_folder, ref, True)

    def _search_parent(self, path, object_folder, ref, multi=False):
        """ Searches a folder for an object with a given reference
        and returns the path of the parent node.

        :param str path: Path to search
        :param str object_folder: The name of the folder within the parent
            object containing the objects to search.
        :param ref: Object reference
        """
        if multi:
            ret = []
        else:
            ret = ''

        for n in self._data.iterNodes(path):
            if not '_type' in n._v_attrs:
                continue
            for c in self._data.iterNodes(n._f_getChild(object_folder)):
                try:
                    if c._f_getAttr("object_ref") == ref:
                        if not multi:
                            return n._v_pathname
                        else:
                            ret.append(n._v_pathname)
                except AttributeError:  # alien node
                    pass  # not an error

        return ret

    _second_parent = {  # Second parent type apart from Segment
        'AnalogSignal': 'RecordingChannel',
        'AnalogSignalArray': 'RecordingChannelGroup',
        'IrregularlySampledSignal': 'RecordingChannel',
        'Spike': 'Unit', 'SpikeTrain': 'Unit'}

    def load_lazy_cascade(self, path, lazy):
        """ Load an object with the given path in lazy cascade mode.
        """
        o = self.get(path, cascade='lazy', lazy=lazy)
        t = type(o).__name__
        node = self._data.getNode(path)

        if not path in self.parent_paths:
            ppaths = []
            if len(getattr(o, '_single_parent_containers', [])) > 1:
                for par_cont in o._single_parent_containers:
                    if par_cont in node._v_attrs:
                        ppaths.append(node._f_getAttr(par_cont))
                    else:
                        ppaths.append(None)
                self.parent_paths[path] = ppaths
            elif  t == 'RecordingChannel':
                if 'recordingchannelgroups' in node._v_attrs:
                    self.parent_paths[path] = node._f_getAttr('recordingchannelgroups')

        # Set parent objects
        if path in self.parent_paths:
            paths = self.parent_paths[path]

            if t == 'RecordingChannel':  # Set list of parnet channel groups
                for rcg in self.parent_paths[path]:
                    o.recordingchannelgroups.append(self.get(rcg, cascade='lazy', lazy=lazy))
            else:  # Set parents: Segment and another parent
                if paths[0] is None:
                    paths[0] = self._get_parent(
                        path, self._data.getNodeAttr(path, 'object_ref'),
                        'Segment')
                if paths[0]:
                    o.segment = self.get(paths[0], cascade='lazy', lazy=lazy)

                parent = self._second_parent[t]
                if paths[1] is None:
                    paths[1] = self._get_parent(
                        path, self._data.getNodeAttr(path, 'object_ref'),
                        parent)
                if paths[1]:
                    setattr(o, parent.lower(), self.get(paths[1], cascade='lazy', lazy=lazy))
        elif t != 'Block':
            ref = self._data.getNodeAttr(path, 'object_ref')

            if t == 'RecordingChannel':
                rcg_paths = self._get_rcgs(path, ref)
                for rcg in rcg_paths:
                    o.recordingchannelgroups.append(self.get(rcg, cascade='lazy', lazy=lazy))
                self.parent_paths[path] = rcg_paths
            else:
                for p in o._single_parent_containers:
                    parent = self._get_parent(path, ref, p)
                    if parent:
                        setattr(o, p.lower(), self.get(parent, cascade='lazy', lazy=lazy))
        return o

    def load_lazy_object(self, obj):
        """ Return the fully loaded version of a lazily loaded object. Does not
        set links to parent objects.
        """
        return self.get(obj.hdf5_path, cascade=False, lazy=False, lazy_loaded=True)

    @_func_wrapper
    def get(self, path="/", cascade=True, lazy=False, lazy_loaded=False):
        """ Returns a requested NEO object as instance of NEO class.
        Set lazy_loaded to True to load a previously lazily loaded object
        (cache is ignored in this case)."""
        def fetch_attribute(attr_name, attr, node):
            """ fetch required attribute from the corresp. node in the file """
            try:
                if attr[1] == pq.Quantity:
                    arr = self._data.getNode(node, attr_name)
                    units = ""
                    for unit in arr._v_attrs._f_list(attrset='user'):
                        if unit.startswith("unit__"):
                            units += " * " + str(unit[6:]) + " ** " + str(arr._f_getAttr(unit))
                    units = units.replace(" * ", "", 1)
                    if not lazy or sum(arr.shape) <= 1:
                        nattr = pq.Quantity(arr.read(), units)
                    else:  # making an empty array
                        nattr = pq.Quantity(np.empty(tuple([0 for _ in range(attr[2])])), units)
                elif attr[1] == np.ndarray:
                    arr = self._data.getNode(node, attr_name)
                    if not lazy:
                        nattr = np.array(arr.read(), attr[3])
                        if nattr.shape == (0, 1):  # Fix: Empty arrays should have only one dimension
                            nattr = nattr.reshape(-1)
                    else:  # making an empty array
                        nattr = np.empty(0, attr[3])
                else:
                    nattr = node._f_getAttr(attr_name)
                    if attr[1] == str or attr[1] == int:
                        nattr = attr[1](nattr)  # compliance with NEO attr types
            except (AttributeError, tb.NoSuchNodeError):  # not assigned, continue
                nattr = None
            return nattr

        def get_lazy_shape(obj, node):
            attr = lazy_shape_arrays[type(obj).__name__]
            arr = self._data.getNode(node, attr)
            return arr.shape

        if path == "/":  # this is just for convenience. Try to return any object
            found = False
            for n in self._data.iterNodes(path):
                for obj_type in objectnames:
                    if obj_type.lower() in str(n._v_name).lower():
                        path = n._v_pathname
                        found = True
                if found:
                    break
        try:
            if path == "/":
                raise ValueError()  # root is not a NEO object
            node = self._data.getNode(path)
        except (tb.NoSuchNodeError, ValueError):  # create a new node?
            raise LookupError("There is no valid object with a given path " +
                              str(path) + ' . Please give correct path or just browse the file '
                              '(e.g. NeoHdf5IO()._data.root.<Block>._segments...) to find an '
                              'appropriate name.')
        classname = self._get_class_by_node(node)
        if not classname:
            raise LookupError("The requested object with the path " + str(path) +
                              " exists, but is not of a NEO type. Please check the '_type' attribute.")

        obj_type = classname.__name__
        try:
            object_ref = self._data.getNodeAttr(node, 'object_ref')
        except AttributeError:  # Object does not have reference, e.g. because this is an old file format
            object_ref = None
        if object_ref in self.objects_by_ref and not lazy_loaded:
            obj = self.objects_by_ref[object_ref]
            if cascade == 'lazy' or obj_type not in complex_relationships:
                return obj
        else:
            kwargs = {}
            # load attributes (inherited *-ed attrs are also here)
            attrs = classname._necessary_attrs + classname._recommended_attrs
            for i, attr in enumerate(attrs):
                attr_name = attr[0]
                nattr = fetch_attribute(attr_name, attr, node)
                if nattr is not None:
                    kwargs[attr_name] = nattr
            obj = class_by_name[obj_type](**kwargs)  # instantiate new object
            if lazy and obj_type in lazy_shape_arrays:
                obj.lazy_shape = get_lazy_shape(obj, node)
            self._update_path(obj, node)  # set up HDF attributes: name, path
            try:
                setattr(obj, "annotations", node._f_getAttr("annotations"))
            except AttributeError:
                pass  # not assigned, continue

        if object_ref and not lazy_loaded:
            self.objects_by_ref[object_ref] = obj
        # load relationships
        if cascade:
            # container is like segments, spiketrains, etc.
            for containername in getattr(obj, '_child_containers', []):
                if cascade == 'lazy':
                    relatives = LazyList(self, lazy)
                else:
                    relatives = []
                container = self._data.getNode(node, containername)
                for n in self._data.iterNodes(container):
                    if cascade == 'lazy':
                        relatives.append(n._v_pathname)
                    else:
                        try:
                            typename = n._f_getAttr("_type").lower()+'s'
                            if typename == containername:
                                relatives.append(self.get(n._v_pathname, lazy=lazy))
                        except AttributeError:  # alien node
                            pass  # not an error
                setattr(obj, containername, relatives)
                if not cascade == 'lazy':
                    # RC -> AnalogSignal relationship will not be created later, do it now
                    if obj_type == "RecordingChannel" and containername == "analogsignals":
                        for r in relatives:
                            r.recordingchannel = obj
                    # Cannot create Many-to-Many relationship with old format, create at least One-to-Many
                    if obj_type == "RecordingChannelGroup" and not object_ref:
                        for r in relatives:
                            r.recordingchannelgroups = [obj]
            # special processor for RC -> RCG
            if obj_type == "RecordingChannel":
                if hasattr(node, '_v_parent'):
                    parent = node._v_parent
                    if hasattr(parent, '_v_parent'):
                        parent = parent._v_parent
                        if 'object_ref' in parent._v_attrs:
                            obj.recordingchannelgroups.append(self.get(
                                parent._v_pathname, lazy=lazy))
        return obj

    @_func_wrapper
    def read_all_blocks(self, lazy=False, cascade=True, **kargs):
        """
        Loads all blocks in the file that are attached to the root (which
        happens when they are saved with save() or write_block()).
        """
        blocks = []
        for n in self._data.iterNodes(self._data.root):
            if self._get_class_by_node(n) == Block:
                blocks.append(self.read_block(n._v_pathname, lazy=lazy, cascade=cascade, **kargs))
        return blocks

    @_func_wrapper
    def write_all_blocks(self, blocks, **kargs):
        """
        Writes a sequence of blocks. Just calls write_block() for each element.
        """
        for b in blocks:
            self.write_block(b)

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
        logger.info("This is a neo.HDF5 file. it contains:")
        info = {}
        info = info.fromkeys(objectnames, 0)
        for node in self._data.walkNodes():
            try:
                t = node._f_getAttr("_type")
                info[t] += 1
            except:
                # node is not of NEO type
                pass
        return info

for obj_type in NeoHdf5IO.writeable_objects:
    setattr(NeoHdf5IO, "write_" + obj_type.__name__.lower(), NeoHdf5IO._write_entity)
for obj_type in NeoHdf5IO.readable_objects:
    setattr(NeoHdf5IO, "read_" + obj_type.__name__.lower(), NeoHdf5IO._read_entity)
