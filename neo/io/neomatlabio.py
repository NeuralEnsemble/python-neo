# encoding: utf-8
"""
Class for reading/writing neo objects in matlab format 5 to 7.2 (.mat).
This module is a bridge for matlab users who want to adopote neo object reprenstation.
Nomenclature is the same but use Matlab struct and cell arrays.
With this modules Matlab users can use neo.io to read a format and convert it to .mat.


Supported : Read/Write


Author: sgarcia

"""

from .baseio import BaseIO
from ..core import *
from .tools import create_many_to_one_relationship
import numpy as np
import quantities as pq
from .. import description
classname_lower_to_upper = { }
for k in description.class_by_name.keys():
    classname_lower_to_upper[k.lower()] = k



from datetime import datetime
import os
import re

# check if version scipy
import scipy
from distutils import version
if version.LooseVersion(scipy.version.version) < '0.8':
    raise ImportError("your scipy version is too old to support MatlabIO, you need at least 0.8 you have %s"%scipy.version.version)

from scipy import io as sio



class NeoMatlabIO(BaseIO):
    """
    Class for reading/writting neo objects in mat matlab format (.mat) 5 to 7.2.

    This module is a bridge for matlab users who want to adopote neo object reprenstation.
    Nomenclature is the same but use Matlab struct and cell arrays.
    With this modules Matlab users can use neo.io to read a format and convert it to .mat.
    
    Rules of conversion:
      * neo classes are converted to matlab struct.
        Ex: Block in neo will be a struct with name, file_datetime, ...
      * neo one_to_many relationship are cellarray in matlab.
        Ex: seg.analogsignals[2] in neo will be seg.analogsignals{3} in matlab.
        Note the one based in matlab and braket vs singleton.
      * Quantity attributes in neo in will be 2 fields in mallab?
         Ex: anasig.t_start = 1.5 * s  (pq.Quantiy) in neo
         will be anasig.t_start = 1.5 and anasig.t_start_unit = 's' in matlab
      * classes that inherits Quantity (AnalogSIgnal, SpikeTrain, ...) in neo will
         have 2 fields (array and units) in matlab struct.
         Ex: AnalogSignal( [1., 2., 3.], 'V') in neo will be
         anasig.array = [1. 2. 3] and anasig.units = 'V' in matlab
    
    1 - **Senario 1: create data in matlab and read them in neo**
      
        This matlab code generate a block::
            
            block = struct();
            block.segments = { };
            block.name = 'my block with matlab';
            for s = 1:3
                seg = struct();
                seg.name = strcat('segment ',num2str(s));
                seg.analogsignals = { };
                for a = 1:5
                    anasig = struct();
                    anasig.array = rand(100,1);
                    anasig.units = 'mV';
                    anasig.t_start = 0;
                    anasig.t_start_units = 's';
                    anasig.sampling_rate = 100;
                    anasig.sampling_rate_units = 'Hz';
                    seg.analogsignals{a} = anasig;
                end
                seg.spiketrains = { };
                for t = 1:7
                    sptr = struct();
                    sptr.array = rand(30,1)*10;
                    sptr.units = 'ms';
                    sptr.t_start = 0;
                    sptr.t_start_units = 'ms';
                    sptr.t_stop = 10;
                    sptr.t_stop_units = 'ms';
                    seg.spiketrains{t} = sptr;
                end
                
                block.segments{s} = seg;
            end
            save 'myblock.mat' block -V7

            
        This code read it in python::
        
            import neo
            r = neo.io.NeoMatlabIO(filename = 'myblock.mat')
            bl = r.read_block()
            print bl.segments[1].analogsignals[2]
            print bl.segments[1].spiketrains[4]
            

    2 - **Senario 2: create data in python with neo and read them in matlab**

        This python code generate the same block as previous (yes, it is more elegant, it is pyhton)::
        
            import neo
            import quantities as pq
            from scipy import rand
            
            bl = neo.Block(name = 'my block with neo')
            for s in range(3):
                seg = neo.Segment( name = 'segment'+str(s))
                bl.segments.append(seg)
                for a in range(5):
                    anasig = neo.AnalogSignal( rand(100), units = 'mV', t_start = 0 * pq.s, sampling_rate = 100*pq.Hz)
                    seg.analogsignals.append(anasig)
                for t in range(7):
                    sptr = neo.SpikeTrain( rand(30), units = 'ms', t_start = 0*pq.ms, t_stop = 10*pq.ms)
                    seg.spiketrains.append(sptr)
            

        w = neo.io.NeoMatlabIO(filename = 'myblock.mat')
        w.write_block(bl)
            
        
        This matlab code read it ::
            
            load 'myblock.mat'
            block.name
            block.segments{2}.analogsignals{3}.array
            block.segments{2}.analogsignals{3}.units
            block.segments{2}.analogsignals{3}.t_start
            block.segments{2}.analogsignals{3}.t_start_units
        
    
    3 - **Senario 3: convertion**
        
        This python code convert a spike2 file to matlab::
        
            from neo import *
            
            r = Spike2IO(filename = 'myspike2file.smr')
            w = NeoMatlabIO(filename ='convertedfile.mat')
            seg = r.read_segment()
            bl = Block(name = 'a block')
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
        BaseIO.__init__(self)
        self.filename = filename
    
    
    def read_block(self, cascade = True, lazy = False,):
        """
        Arguments:
            
        """
        d = sio.loadmat(self.filename, struct_as_record=False, squeeze_me=True)
        assert'block' in d, 'no block in'+self.filename
        bl_struct = d['block']
        bl =  self.create_ob_from_struct(bl_struct, 'Block', cascade = cascade, lazy = lazy)
        create_many_to_one_relationship(bl)
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
            
        sio.savemat(self.filename, {'block':bl_struct}, oned_as = 'row')



    def create_struct_from_obj(self, ob, ):
        classname = ob.__class__.__name__
        struct = { }
        
        # relationship
        rel = description.one_to_many_relationship
        if classname in rel:
            for childname in rel[classname]:
                if description.class_by_name[childname] in self.supported_objects:
                    struct[childname.lower()+'s'] = [ ]
        # attributes
        necess = description.classes_necessary_attributes[classname]
        recomm = description.classes_recommended_attributes[classname]
        attributes = necess + recomm
        for i, attr in enumerate(attributes):
            
            attrname, attrtype = attr[0], attr[1]
            
            #~ if attrname =='': 
                #~ struct['array'] = ob.magnitude
                #~ struct['units'] = ob.dimensionality.string
                #~ continue
            
            if  classname in description.classes_inheriting_quantities and \
                    description.classes_inheriting_quantities[classname] == attrname:
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
        cl = description.class_by_name[classname]
        # check if hinerits Quantity
        #~ is_quantity = False
        #~ for attr in description.classes_necessary_attributes[classname]:
            #~ if attr[0] == '' and attr[1] == pq.Quantity:
                #~ is_quantity = True
                #~ break
        #~ is_quantiy = classname in description.classes_inheriting_quantities
        
        #~ if is_quantity:
        if  classname in description.classes_inheriting_quantities:
            
            quantity_attr = description.classes_inheriting_quantities[classname]
            arr = getattr(struct,quantity_attr)
            #~ data_complement = dict(units=str(struct.units))
            data_complement = dict(units=str(getattr(struct,quantity_attr+'_units')))
            if "sampling_rate" in (at[0] for at in description.classes_necessary_attributes[classname]):
                data_complement["sampling_rate"] = 0*pq.kHz  # put fake value for now, put correct value later
            if "t_stop" in (at[0] for at in description.classes_necessary_attributes[classname]):
                if len(arr) > 0:
                    data_complement["t_stop"] =arr.max()
                else:
                    data_complement["t_stop"] = 0.0
            if lazy:    
                ob = cl([ ], **data_complement)
                ob.lazy_shape = arr.shape
            else:
                ob = cl(arr, **data_complement)
        else:
            ob = cl()
        for attrname in struct._fieldnames:
            # check children
            rel = description.one_to_many_relationship
            if classname in rel and attrname[:-1] in [ r.lower() for r in rel[classname] ]:
                for c in range(len(getattr(struct,attrname))):
                    if cascade:
                        child = self.create_ob_from_struct(getattr(struct,attrname)[c]  , classname_lower_to_upper[attrname[:-1]], 
                                                                                    cascade = cascade, lazy = lazy)
                        getattr(ob, attrname.lower()).append(child)
                continue
            
            # attributes
            if attrname.endswith('_units')  or attrname =='units' :#or attrname == 'array':
                # linked with another field
                continue
            if  classname in description.classes_inheriting_quantities and \
                    description.classes_inheriting_quantities[classname] == attrname:
                continue
            
            item = getattr(struct, attrname)
            # put the good type
            necess = description.classes_necessary_attributes[classname]
            recomm = description.classes_recommended_attributes[classname]
            attributes = necess + recomm

            dict_attributes = dict( [ (a[0], a[1:]) for a in attributes])
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



