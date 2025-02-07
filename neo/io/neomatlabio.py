"""
Module for reading/writing Neo objects in MATLAB format (.mat) versions
5 to 7.2.

This module is a bridge for MATLAB users who want to adopt the Neo object
representation. The nomenclature is the same but using Matlab structs and cell
arrays. With this module MATLAB users can use neo.io to read a format and
convert it to .mat.

Supported : Read/Write

Author: sgarcia, Robert Pröpper
"""

from collections.abc import Mapping
from datetime import datetime
import re

import numpy as np
import quantities as pq

from packaging.version import Version


from neo.io.baseio import BaseIO
from neo.core import (
    Block,
    Segment,
    AnalogSignal,
    IrregularlySampledSignal,
    Event,
    Epoch,
    SpikeTrain,
    Group,
    ImageSequence,
    ChannelView,
    RectangularRegionOfInterest,
    CircularRegionOfInterest,
    PolygonRegionOfInterest,
    objectnames,
    class_by_name,
    NeoReadWriteError,
)
from neo.core.regionofinterest import RegionOfInterest
from neo.core.baseneo import _container_name


def get_classname_from_container_name(container_name, struct):
    if container_name == "regionsofinterest":
        if "radius" in struct._fieldnames:
            return "CircularRegionOfInterest"
        elif "vertices" in struct._fieldnames:
            return "PolygonRegionOfInterest"
        else:
            return "RectangularRegionOfInterest"
    else:
        for classname in objectnames:
            if _container_name(classname) == container_name:
                return classname


PY_NONE = "Py_None"


class NeoMatlabIO(BaseIO):
    """
    Class for reading/writing Neo objects in MATLAB format (.mat) versions
    5 to 7.2.

    This module is a bridge for MATLAB users who want to adopt the Neo object
    representation.  The nomenclature is the same but using Matlab structs and
    cell arrays. With this module MATLAB users can use neo.io to read a format
    and convert it to .mat.

    Rules of conversion:
      * Neo classes are converted to MATLAB structs.
        e.g., a Block is a struct with attributes "name", "file_datetime", ...
      * Neo one_to_many relationships are cellarrays in MATLAB.
        e.g., ``seg.analogsignals[2]`` in Python Neo will be
        ``seg.analogsignals{3}`` in MATLAB.
      * Quantity attributes are represented by 2 fields in MATLAB.
        e.g., ``anasig.t_start = 1.5 * s`` in Python
        will be ``anasig.t_start = 1.5`` and ``anasig.t_start_unit = 's'``
        in MATLAB.
      * classes that inherit from Quantity (AnalogSignal, SpikeTrain, ...) in
        Python will have 2 fields (array and units) in the MATLAB struct.
        e.g.: ``AnalogSignal( [1., 2., 3.], 'V')`` in Python will be
        ``anasig.array = [1. 2. 3]`` and ``anasig.units = 'V'`` in MATLAB.

    1 - **Scenario 1: create data in MATLAB and read them in Python**

        This MATLAB code generates a block::

            block = struct();
            block.segments = { };
            block.name = 'my block with matlab';
            for s = 1:3
                seg = struct();
                seg.name = strcat('segment ',num2str(s));

                seg.analogsignals = { };
                for a = 1:5
                    anasig = struct();
                    anasig.signal = rand(100,1);
                    anasig.signal_units = 'mV';
                    anasig.t_start = 0;
                    anasig.t_start_units = 's';
                    anasig.sampling_rate = 100;
                    anasig.sampling_rate_units = 'Hz';
                    seg.analogsignals{a} = anasig;
                end

                seg.spiketrains = { };
                for t = 1:7
                    sptr = struct();
                    sptr.times = rand(30,1)*10;
                    sptr.times_units = 'ms';
                    sptr.t_start = 0;
                    sptr.t_start_units = 'ms';
                    sptr.t_stop = 10;
                    sptr.t_stop_units = 'ms';
                    seg.spiketrains{t} = sptr;
                end

                event = struct();
                event.times = [0, 10, 30];
                event.times_units = 'ms';
                event.labels = ['trig0'; 'trig1'; 'trig2'];
                seg.events{1} = event;

                epoch = struct();
                epoch.times = [10, 20];
                epoch.times_units = 'ms';
                epoch.durations = [4, 10];
                epoch.durations_units = 'ms';
                epoch.labels = ['a0'; 'a1'];
                seg.epochs{1} = epoch;

                block.segments{s} = seg;

            end

            save 'myblock.mat' block -V7


        This code reads it in Python::

            import neo
            r = neo.io.NeoMatlabIO(filename='myblock.mat')
            bl = r.read_block()
            print bl.segments[1].analogsignals[2]
            print bl.segments[1].spiketrains[4]


    2 - **Scenario 2: create data in Python and read them in MATLAB**

        This Python code generates the same block as in the previous scenario::

            import neo
            import quantities as pq
            from scipy import rand, array

            bl = neo.Block(name='my block with neo')
            for s in range(3):
                seg = neo.Segment(name='segment' + str(s))
                bl.segments.append(seg)
                for a in range(5):
                    anasig = neo.AnalogSignal(rand(100)*pq.mV, t_start=0*pq.s,
                                              sampling_rate=100*pq.Hz)
                    seg.analogsignals.append(anasig)
                for t in range(7):
                    sptr = neo.SpikeTrain(rand(40)*pq.ms, t_start=0*pq.ms, t_stop=10*pq.ms)
                    seg.spiketrains.append(sptr)
                ev = neo.Event([0, 10, 30]*pq.ms, labels=array(['trig0', 'trig1', 'trig2']))
                ep = neo.Epoch([10, 20]*pq.ms, durations=[4, 10]*pq.ms, labels=array(['a0', 'a1']))
                seg.events.append(ev)
                seg.epochs.append(ep)

            from neo.io.neomatlabio import NeoMatlabIO
            w = NeoMatlabIO(filename='myblock.mat')
            w.write_block(bl)


        This MATLAB code reads it::

            load 'myblock.mat'
            block.name
            block.segments{2}.analogsignals{3}.signal
            block.segments{2}.analogsignals{3}.signal_units
            block.segments{2}.analogsignals{3}.t_start
            block.segments{2}.analogsignals{3}.t_start_units


    3 - **Scenario 3: conversion**

        This Python code converts a Spike2 file to MATLAB::

            from neo import Block
            from neo.io import Spike2IO, NeoMatlabIO

            r = Spike2IO(filename='spike2.smr')
            w = NeoMatlabIO(filename='convertedfile.mat')
            blocks = r.read()
            w.write(blocks[0])

    """

    is_readable = True
    is_writable = True

    supported_objects = [
        Block,
        Segment,
        AnalogSignal,
        IrregularlySampledSignal,
        Epoch,
        Event,
        SpikeTrain,
        Group,
        ImageSequence,
        ChannelView,
        RectangularRegionOfInterest,
        CircularRegionOfInterest,
        PolygonRegionOfInterest,
    ]
    readable_objects = [Block]
    writeable_objects = [Block]

    has_header = False
    is_streameable = False
    read_params = {Block: []}
    write_params = {Block: []}

    name = "neomatlab"
    extensions = ["mat"]

    mode = "file"

    def __init__(self, filename=None):
        """
        This class read/write neo objects in matlab 5 to 7.2 format.

        Arguments:
            filename : the filename to read
        """
        import scipy.version

        if Version(scipy.version.version) < Version("0.12.0"):
            raise ImportError(
                "your scipy version is too old to support "
                + "MatlabIO, you need at least 0.12.0. "
                + f"You have {scipy.version.version}"
            )

        BaseIO.__init__(self)
        self.filename = filename
        self._refs = {}

    def read_block(self, lazy=False):
        """
        Arguments:

        """
        import scipy.io

        if lazy:
            raise NeoReadWriteError(f"This IO does not support lazy reading")

        d = scipy.io.loadmat(self.filename, struct_as_record=False, squeeze_me=True, mat_dtype=True)
        if "block" not in d:
            self.logger.exception("No block in " + self.filename)
            return None

        bl_struct = d["block"]
        bl = self.create_ob_from_struct(bl_struct, "Block")
        self._resolve_references(bl)
        bl.check_relationships()
        return bl

    def write_block(self, bl, **kargs):
        """
        Arguments:
            bl: the block to be saved
            kargs: extra keyword arguments broadcasted to scipy.io.savemat

        """
        import scipy.io

        bl_struct = self.create_struct_from_obj(bl)

        for seg in bl.segments:
            seg_struct = self.create_struct_from_obj(seg)
            bl_struct["segments"].append(seg_struct)

            for container_name in seg._child_containers:
                for child_obj in getattr(seg, container_name):
                    child_struct = self.create_struct_from_obj(child_obj)
                    seg_struct[container_name].append(child_struct)

        for group in bl.groups:
            group_structure = self.create_struct_from_obj(group)
            bl_struct["groups"].append(group_structure)

            for container_name in group._child_containers:
                for child_obj in getattr(group, container_name):
                    if isinstance(child_obj, (ChannelView, RegionOfInterest)):
                        child_struct = self.create_struct_from_view(child_obj)
                        group_structure[container_name].append(child_struct)
                    else:
                        group_structure[container_name].append(id(child_obj))

        scipy.io.savemat(self.filename, {"block": bl_struct}, oned_as="row", **kargs)

    def _get_matlab_value(self, ob, attrname):
        units = None
        if hasattr(ob, "_quantity_attr") and ob._quantity_attr == attrname:
            units = ob.dimensionality.string
            value = ob.magnitude
        else:
            try:
                value = getattr(ob, attrname)
            except AttributeError:
                value = ob[attrname]
            if isinstance(value, pq.Quantity):
                units = value.dimensionality.string
                value = value.magnitude
            elif isinstance(value, datetime):
                value = str(value)
            elif isinstance(value, Mapping):
                new_value = {}
                for key in value:
                    subvalue, subunits = self._get_matlab_value(value, key)
                    if subvalue is not None:
                        new_value[key] = subvalue
                        if subunits:
                            new_value[f"{key}_units"] = subunits
                    elif attrname == "annotations":
                        # In general we don't send None to MATLAB
                        # but we make an exception for annotations.
                        # However, we have to save then retrieve some
                        # special value as actual `None` is ignored by default.
                        new_value[key] = PY_NONE
                value = new_value
        return value, units

    def create_struct_from_obj(self, ob):
        struct = {"neo_id": id(ob)}

        # relationship
        for childname in getattr(ob, "_child_containers", []):
            supported_containers = [_container_name(subob.__name__) for subob in self.supported_objects]
            if childname in supported_containers:
                struct[childname] = []

        # attributes
        all_attrs = list(ob._all_attrs)
        if hasattr(ob, "annotations"):
            all_attrs.append(("annotations", type(ob.annotations)))

        for attr in all_attrs:
            attrname, attrtype = attr[0], attr[1]
            attr_value, attr_units = self._get_matlab_value(ob, attrname)
            if attr_value is not None:
                struct[attrname] = attr_value
                if attr_units:
                    struct[attrname + "_units"] = attr_units
        return struct

    def create_struct_from_view(self, ob):
        # for "view" objects (ChannelView and RegionOfInterest), we just store
        # a reference to the object (AnalogSignal, ImageSequence) that the view
        # points to
        struct = self.create_struct_from_obj(ob)
        obj_name = ob._necessary_attrs[0][0]  # this is fragile, better to add an attribute _view_attr
        viewed_obj = getattr(ob, obj_name)
        struct[obj_name] = id(viewed_obj)
        struct["viewed_classname"] = viewed_obj.__class__.__name__
        return struct

    def create_ob_from_struct(self, struct, classname):
        cl = class_by_name[classname]

        # ~ if is_quantity:
        if hasattr(cl, "_quantity_attr"):
            quantity_attr = cl._quantity_attr
            arr = getattr(struct, quantity_attr)
            # ~ data_complement = dict(units=str(struct.units))
            data_complement = dict(units=str(getattr(struct, quantity_attr + "_units")))
            if "sampling_rate" in (at[0] for at in cl._necessary_attrs):
                # put fake value for now, put correct value later
                data_complement["sampling_rate"] = 0 * pq.kHz
            try:
                len(arr)
            except TypeError:
                # strange scipy.io behavior: if len is 1 we get a float
                arr = np.array(arr)
                arr = arr.reshape((-1,))  # new view with one dimension
            if "t_stop" in (at[0] for at in cl._necessary_attrs):
                if len(arr) > 0:
                    data_complement["t_stop"] = arr.max()
                else:
                    data_complement["t_stop"] = 0.0
            if "t_start" in (at[0] for at in cl._necessary_attrs):
                if len(arr) > 0:
                    data_complement["t_start"] = arr.min()
                else:
                    data_complement["t_start"] = 0.0
            if "spatial_scale" in (at[0] for at in cl._necessary_attrs):
                if len(arr) > 0:
                    data_complement["spatial_scale"] = arr
                else:
                    data_complement["spatial_scale"] = 1.0

            if "times" in (at[0] for at in cl._necessary_attrs) and quantity_attr != "times":
                # handle IrregularlySampledSignal
                times = getattr(struct, "times")
                data_complement["time_units"] = getattr(struct, "times_units")
                ob = cl(times, arr, **data_complement)
            else:
                ob = cl(arr, **data_complement)
        elif cl.is_view:
            kwargs = {}
            for i, attr in enumerate(cl._necessary_attrs):
                value = getattr(struct, attr[0])
                if i == 0:
                    # this is a bit hacky, should really add an attribute _view_attr to ChannelView and RegionOfInterest
                    if not isinstance(value, int):  # object id
                        raise TypeError(f"value must be int not of type {type(value)}")
                    kwargs[attr[0]] = _Ref(identifier=value, target_class_name=struct.viewed_classname)
                else:
                    if attr[1] == np.ndarray and isinstance(value, int):
                        value = np.array([value])
                    kwargs[attr[0]] = value
            ob = cl(**kwargs)
        else:
            ob = cl()

        for attrname in struct._fieldnames:
            # check children
            if attrname in getattr(ob, "_child_containers", []):
                child_struct = getattr(struct, attrname)
                try:
                    # try must only surround len() or other errors are captured
                    child_len = len(child_struct)
                except TypeError:
                    # strange scipy.io behavior: if len is 1 there is no len()
                    child_struct = [child_struct]
                    child_len = 1

                for c in range(child_len):
                    child_class_name = get_classname_from_container_name(attrname, child_struct[c])
                    if classname == "Group":
                        if child_class_name == ("ChannelView") or "RegionOfInterest" in child_class_name:
                            child = self.create_ob_from_struct(child_struct[c], child_class_name)
                        else:
                            child = _Ref(child_struct[c], child_class_name)
                    else:
                        child = self.create_ob_from_struct(child_struct[c], child_class_name)
                    getattr(ob, attrname.lower()).append(child)
                continue

            # attributes
            if attrname.endswith("_units") or attrname == "units":
                # linked with another field
                continue

            if hasattr(cl, "_quantity_attr") and cl._quantity_attr == attrname:
                continue

            if cl.is_view and attrname in (
                "obj",
                "index",
                "image_sequence",
                "x",
                "y",
                "radius",
                "width",
                "height",
                "vertices",
            ):
                continue

            item = getattr(struct, attrname)
            attributes = cl._necessary_attrs + cl._recommended_attrs + (("annotations", dict),)
            dict_attributes = dict([(a[0], a[1:]) for a in attributes])

            if attrname in dict_attributes:
                attrtype = dict_attributes[attrname][0]
                if attrtype == datetime:
                    m = r"(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+).(\d+)"
                    r = re.findall(m, str(item))
                    if len(r) == 1:
                        item = datetime(*[int(e) for e in r[0]])
                    else:
                        item = None
                elif attrtype == np.ndarray:
                    dt = dict_attributes[attrname][2]
                    try:
                        item = item.astype(dt)
                    except AttributeError:
                        # it seems arrays of length 1 are stored as scalars
                        item = np.array([item], dtype=dt)
                elif attrtype == pq.Quantity:
                    ndim = dict_attributes[attrname][1]
                    units = str(getattr(struct, attrname + "_units"))
                    if ndim == 0:
                        item = pq.Quantity(item, units)
                    else:
                        item = pq.Quantity(item, units)
                elif attrtype == dict:
                    new_item = {}
                    for fn in item._fieldnames:
                        value = getattr(item, fn)
                        if value == PY_NONE:
                            value = None
                        new_item[fn] = value
                    item = new_item
                else:
                    item = attrtype(item)

            setattr(ob, attrname, item)

        neo_id = getattr(struct, "neo_id", None)
        if neo_id:
            setattr(ob, "_id", neo_id)
        return ob

    def _resolve_references(self, bl):
        if bl.groups:
            obj_lookup = {}
            for ob in bl.children_recur:
                if hasattr(ob, "_id"):
                    obj_lookup[ob._id] = ob
            for grp in bl.groups:
                for container_name in grp._child_containers:
                    container = getattr(grp, container_name)
                    for i, item in enumerate(container):
                        if isinstance(item, _Ref):
                            if not isinstance(item.identifier, (int, np.integer)):
                                raise TypeError(
                                    f"item.identifier must be either int or np.integer not of type {type(item.identifier)}"
                                )
                            # A reference to an object that already exists
                            container[i] = obj_lookup[item.identifier]
                        else:
                            # ChannelView and RegionOfInterest
                            if not item.is_view:
                                raise TypeError(f"`item` must be a view")
                            if not isinstance(item.obj, _Ref):
                                raise TypeError(f"`item.obj` must be a {_Ref} and is of type {type(item.obj)}")
                            item.obj = obj_lookup[item.obj.identifier]


class _Ref:
    def __init__(self, identifier, target_class_name):
        self.identifier = identifier
        if target_class_name:
            self.target_cls = class_by_name[target_class_name]
        else:
            self.target_cls = None

    @property
    def proxy_for(self):
        return self.target_cls

    @property
    def data_children_recur(self):
        return []

    @property
    def container_children_recur(self):
        return []
