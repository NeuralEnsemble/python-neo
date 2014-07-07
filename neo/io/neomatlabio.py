# -*- coding: utf-8 -*-
"""
Module for reading/writing Neo objects in MATLAB format (.mat) versions 5 to 7.2.

This module is a bridge for MATLAB users who want to adopt the Neo object representation.
The nomenclature is the same but using Matlab structs and cell arrays.
With this module MATLAB users can use neo.io to read a format and convert it to .mat.

Supported : Read/Write

Author: sgarcia
"""

from datetime import datetime
from distutils import version
import re

import numpy as np
import quantities as pq

# check scipy
try:
    import scipy.io
    import scipy.version
except ImportError as err:
    HAVE_SCIPY = False
    SCIPY_ERR = err
else:
    if version.LooseVersion(scipy.version.version) < '0.8':
        HAVE_SCIPY = False
        SCIPY_ERR = ImportError("your scipy version is too old to support " +
                                "MatlabIO, you need at least 0.8. " +
                                "You have %s" % scipy.version.version)
    else:
        HAVE_SCIPY = True
        SCIPY_ERR = None


from neo.io.baseio import BaseIO
from neo.core import (Block, Segment, AnalogSignal, EventArray, SpikeTrain,
                      objectnames, class_by_name)


classname_lower_to_upper = {}
for k in objectnames:
    classname_lower_to_upper[k.lower()] = k


class NeoMatlabIO(BaseIO):
    """
    Class for reading/writing Neo objects in MATLAB format (.mat) versions 5 to 7.2.

    This module is a bridge for MATLAB users who want to adopt the Neo object representation.
    The nomenclature is the same but using Matlab structs and cell arrays.
    With this module MATLAB users can use neo.io to read a format and convert it to .mat.

    Rules of conversion:
      * Neo classes are converted to MATLAB structs.
        e.g., a Block is a struct with attributes "name", "file_datetime", ...
      * Neo one_to_many relationships are cellarrays in MATLAB.
        e.g., ``seg.analogsignals[2]`` in Python Neo will be ``seg.analogsignals{3}`` in MATLAB.
      * Quantity attributes are represented by 2 fields in MATLAB.
        e.g., ``anasig.t_start = 1.5 * s`` in Python
        will be ``anasig.t_start = 1.5`` and ``anasig.t_start_unit = 's'`` in MATLAB.
      * classes that inherit from Quantity (AnalogSignal, SpikeTrain, ...) in Python will
        have 2 fields (array and units) in the MATLAB struct.
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
            from scipy import rand

            bl = neo.Block(name='my block with neo')
            for s in range(3):
                seg = neo.Segment(name='segment' + str(s))
                bl.segments.append(seg)
                for a in range(5):
                    anasig = neo.AnalogSignal(rand(100), units='mV', t_start=0*pq.s, sampling_rate=100*pq.Hz)
                    seg.analogsignals.append(anasig)
                for t in range(7):
                    sptr = neo.SpikeTrain(rand(30), units='ms', t_start=0*pq.ms, t_stop=10*pq.ms)
                    seg.spiketrains.append(sptr)

        w = neo.io.NeoMatlabIO(filename='myblock.mat')
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

            r = Spike2IO(filename='myspike2file.smr')
            w = NeoMatlabIO(filename='convertedfile.mat')
            seg = r.read_segment()
            bl = Block(name='a block')
            bl.segments.append(seg)
            w.write_block(bl)

    """
    is_readable        = True
    is_writable        = True

    supported_objects            = [ Block, Segment , AnalogSignal , EventArray, SpikeTrain ]
    readable_objects    = [Block, ]
    writeable_objects    = [Block, ]

    has_header         = False
    is_streameable     = False
    read_params        = { Block : [ ] }
    write_params       = { Block : [ ] }

    name               = 'neomatlab'
    extensions          = [ 'mat' ]

    mode = 'file'

    def __init__(self , filename = None) :
        """
        This class read/write neo objects in matlab 5 to 7.2 format.

        Arguments:
            filename : the filename to read
        """
        if not HAVE_SCIPY:
            raise SCIPY_ERR
        BaseIO.__init__(self)
        self.filename = filename


    def read_block(self, cascade = True, lazy = False,):
        """
        Arguments:

        """
        d = scipy.io.loadmat(self.filename, struct_as_record=False,
                             squeeze_me=True)
        assert'block' in d, 'no block in'+self.filename
        bl_struct = d['block']
        bl =  self.create_ob_from_struct(bl_struct, 'Block', cascade = cascade, lazy = lazy)
        bl.create_many_to_one_relationship()
        return bl

    def write_block(self, bl,):
        """
        Arguments::
            bl: the block to b saved
        """

        bl_struct = self.create_struct_from_obj(bl)

        for seg in bl.segments:
            seg_struct = self.create_struct_from_obj(seg)
            bl_struct['segments'].append(seg_struct)

            for anasig in seg.analogsignals:
                anasig_struct = self.create_struct_from_obj(anasig)
                seg_struct['analogsignals'].append(anasig_struct)

            for ea in seg.eventarrays:
                ea_struct = self.create_struct_from_obj(ea)
                seg_struct['eventarrays'].append(ea_struct)

            for sptr in seg.spiketrains:
                sptr_struct = self.create_struct_from_obj(sptr)
                seg_struct['spiketrains'].append(sptr_struct)

        scipy.io.savemat(self.filename, {'block':bl_struct}, oned_as = 'row')



    def create_struct_from_obj(self, ob, ):
        classname = ob.__class__.__name__
        struct = { }

        # relationship
        for childname in getattr(ob, '_single_child_containers', []):
            supported_containers = [subob.__name__.lower() + 's' for subob in
                                    self.supported_objects]
            if childname in supported_containers:
                struct[childname] = []
        # attributes
        for i, attr in enumerate(ob._all_attrs):

            attrname, attrtype = attr[0], attr[1]

            #~ if attrname =='':
                #~ struct['array'] = ob.magnitude
                #~ struct['units'] = ob.dimensionality.string
                #~ continue

            if (hasattr(ob, '_quantity_attr') and
                    ob._quantity_attr == attrname):
                struct[attrname] = ob.magnitude
                struct[attrname+'_units'] = ob.dimensionality.string
                continue


            if not(attrname in ob.annotations or hasattr(ob, attrname)): continue
            if getattr(ob, attrname) is None : continue

            if attrtype == pq.Quantity:
                #ndim = attr[2]
                struct[attrname] = getattr(ob,attrname).magnitude
                struct[attrname+'_units'] = getattr(ob,attrname).dimensionality.string
            elif attrtype ==datetime:
                struct[attrname] = str(getattr(ob,attrname))
            else:
                struct[attrname] = getattr(ob,attrname)

        return struct

    def create_ob_from_struct(self, struct, classname, cascade = True, lazy = False,):
        cl = class_by_name[classname]
        # check if hinerits Quantity
        #~ is_quantity = False
        #~ for attr in cl._necessary_attrs:
            #~ if attr[0] == '' and attr[1] == pq.Quantity:
                #~ is_quantity = True
                #~ break
        #~ is_quantiy = hasattr(cl, '_quantity_attr')

        #~ if is_quantity:
        if hasattr(cl, '_quantity_attr'):

            quantity_attr = cl._quantity_attr
            arr = getattr(struct,quantity_attr)
            #~ data_complement = dict(units=str(struct.units))
            data_complement = dict(units=str(getattr(struct,quantity_attr+'_units')))
            if "sampling_rate" in (at[0] for at in cl._necessary_attrs):
                data_complement["sampling_rate"] = 0*pq.kHz  # put fake value for now, put correct value later
            if "t_stop" in (at[0] for at in cl._necessary_attrs):
                if len(arr) > 0:
                    data_complement["t_stop"] =arr.max()
                else:
                    data_complement["t_stop"] = 0.0
            if "t_start" in (at[0] for at in cl._necessary_attrs):
                if len(arr) > 0:
                    data_complement["t_start"] =arr.min()
                else:
                    data_complement["t_start"] = 0.0
            
            if lazy:
                ob = cl([ ], **data_complement)
                ob.lazy_shape = arr.shape
            else:
                ob = cl(arr, **data_complement)
        else:
            ob = cl()
        for attrname in struct._fieldnames:
            # check children
            if attrname in getattr(ob, '_single_child_containers', []):
                try:
                    for c in range(len(getattr(struct,attrname))):
                        if cascade:
                            child = self.create_ob_from_struct(getattr(struct,attrname)[c]  , classname_lower_to_upper[attrname[:-1]],
                                                                                        cascade = cascade, lazy = lazy)
                            getattr(ob, attrname.lower()).append(child)
                except TypeError:
                    # strange behavior in scipy.io: if len is 1 so there is no len() 
                    if cascade:
                        child = self.create_ob_from_struct(getattr(struct,attrname)  , classname_lower_to_upper[attrname[:-1]],
                                                                                        cascade = cascade, lazy = lazy)
                        getattr(ob, attrname.lower()).append(child)                    
                continue

            # attributes
            if attrname.endswith('_units')  or attrname =='units' :#or attrname == 'array':
                # linked with another field
                continue
            if (hasattr(cl, '_quantity_attr') and
                    cl._quantity_attr == attrname):
                continue

            item = getattr(struct, attrname)

            attributes = cl._necessary_attrs + cl._recommended_attrs
            dict_attributes = dict([(a[0], a[1:]) for a in attributes])
            if attrname in dict_attributes:
                attrtype = dict_attributes[attrname][0]
                if attrtype == datetime:
                    m = '(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+).(\d+)'
                    r = re.findall(m, str(item))
                    if len(r)==1:
                        item = datetime( *[ int(e) for e in r[0] ] )
                    else:
                        item = None
                elif attrtype == np.ndarray:
                    dt = dict_attributes[attrname][2]
                    if lazy:
                        item = np.array([ ], dtype = dt)
                        ob.lazy_shape = item.shape
                    else:
                        item = item.astype( dt )
                elif attrtype == pq.Quantity:
                    ndim = dict_attributes[attrname][1]
                    units = str(getattr(struct, attrname+'_units'))
                    if ndim == 0:
                        item = pq.Quantity(item, units)
                    else:
                        if lazy:
                            item = pq.Quantity([ ], units)
                            item.lazy_shape = item.shape
                        else:
                            item = pq.Quantity(item, units)
                else:
                    item = attrtype(item)

            setattr(ob, attrname, item)


        return ob



