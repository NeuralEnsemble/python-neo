# -*- coding: utf-8 -*-
"""
README
=====================================================================================
This is the implementation of the NEO IO for Elphy files.

IO dependencies:
- NEO
- types
- numpy
- quantities


Quick reference:
=====================================================================================
Class ElphyIO() with methods read_block() and write_block() are implemented. 
This classes represent the way to access and produce Elphy files 
from NEO objects.

As regards reading an existing Elphy file, start by initializing a IO class with it:

>>> import neo
>>> r = neo.io.ElphyIO( filename="Elphy.DAT" )
>>> r
<neo.io.elphyio.ElphyIO object at 0xa1e960c>

Read the file content into NEO object Block:

>>> bl = r.read_block(lazy=False, cascade=True)
>>> bl
<neo.core.block.Block object at 0x9e3d44c>

Now you can then read all Elphy data as NEO objects:

>>> b1.segments
[<neo.core.segment.Segment object at 0x9ed85cc>,
 <neo.core.segment.Segment object at 0x9ed85ec>,
 <neo.core.segment.Segment object at 0x9ed880c>,
 <neo.core.segment.Segment object at 0x9ed89cc>]

>>> bl.segments[0].analogsignals[0]
<AnalogSignal(array([ 0.        , -0.0061037 , -0.0061037 , ...,  0.        ,
       -0.0061037 , -0.01831111]) * mV, [0.0 s, 7226.2 s], sampling rate: 10.0 Hz)>

These functions return NEO objects, completely "detached" from the original Elphy file.
Changes to the runtime objects will not cause any changes in the file.

Having already existing NEO structures, it is possible to write them as an Elphy file.
For example, given a segment:

>>> s = neo.Segment()

filled with other NEO structures:

>>> import numpy as np
>>> import quantities as pq
>>> a = AnalogSignal( signal=np.random.rand(300), t_start=42*pq.ms)
>>> s.analogsignals.append( a )

and added to a newly created NEO Block:

>>> bl = neo.Block()
>>> bl.segments.append( s )

Then, it's easy to create an Elphy file:

>>> r = neo.io.ElphyIO( filename="ElphyNeoTest.DAT" )
>>> r.write_block( bl )


Author: Thierry Brizzi
        Domenico Guarino
"""

# needed for python 3 compatibility
from __future__ import absolute_import

# python commons:
from datetime import datetime
from fractions import gcd
from os import path
import re
import struct
from time import time

# note neo.core needs only numpy and quantities
import numpy as np
import quantities as pq

# I need to subclass BaseIO
from neo.io.baseio import BaseIO

# to import from core
from neo.core import (Block, Segment, RecordingChannelGroup, RecordingChannel,
                      AnalogSignal, AnalogSignalArray, EventArray, SpikeTrain)

# --------------------------------------------------------
# OBJECTS

class ElphyScaleFactor(object):
    """
    Useful to retrieve real values from integer
    ones that are stored in an Elphy file :
    
    ``scale`` : compute the actual value of a sample
    with this following formula :
    
        ``delta`` * value + ``offset`` 
    
    """
    
    def __init__(self, delta, offset):
        self.delta = delta
        self.offset = offset
        
    def scale(self, value):
        return value * self.delta + self.offset

class BaseSignal(object):
    """
    A descriptor storing main signal properties :
    
    ``layout`` : the :class:``ElphyLayout` object
    that extracts data from a file.
    
    ``episode`` : the episode in which the signal
    has been acquired.
    
    ``sampling_frequency`` : the sampling frequency
    of the analog to digital converter.
    
    ``sampling_period`` : the sampling period of the
    analog to digital converter computed from sampling_frequency.
    
    ``t_start`` : the start time of the signal acquisition.
    
    ``t_stop`` : the end time of the signal acquisition.
    
    ``duration`` : the duration of the signal acquisition
    computed from t_start and t_stop.
    
    ``n_samples`` : the number of sample acquired during the
    recording computed from the duration and the sampling period.
    
    ``name`` : a label to identify the signal.
    
    ``data`` : a property triggering data extraction.

    """
    
    def __init__(self, layout, episode, sampling_frequency, start, stop, name=None):
        self.layout = layout
        self.episode = episode
        self.sampling_frequency = sampling_frequency
        self.sampling_period = 1 / sampling_frequency
        self.t_start = start
        self.t_stop = stop
        self.duration = self.t_stop - self.t_start 
        self.n_samples = int(self.duration / self.sampling_period)
        self.name = name
    
    @property
    def data(self):
        raise NotImplementedError('must be overloaded in subclass')
    
class ElphySignal(BaseSignal):
    """
    Subclass of :class:`BaseSignal` corresponding to Elphy's analog channels :
    
    ``channel`` : the identifier of the analog channel providing the signal.
    ``units`` : an array containing x and y coordinates units.
    ``x_unit`` : a property to access the x-coordinates unit.
    ``y_unit`` : a property to access the y-coordinates unit.
    ``data`` : a property that delegate data extraction to the 
                ``get_signal_data`` function of the ```layout`` object.
    """
    def __init__(self, layout, episode, channel, x_unit, y_unit, sampling_frequency, start, stop, name=None):
        super(ElphySignal, self).__init__(layout, episode, sampling_frequency, start, stop, name)
        self.channel = channel
        self.units = [x_unit, y_unit]
    
    def __str__(self):
        return "%s ep_%s ch_%s [%s, %s]" % (self.layout.file.name, self.episode, self.channel, self.x_unit, self.y_unit)
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def x_unit(self):
        """
        Return the x-coordinate of the signal.
        """
        return self.units[0]
    
    @property
    def y_unit(self):
        """
        Return the y-coordinate of the signal.
        """
        return self.units[1]
    
    @property
    def data(self):
        return self.layout.get_signal_data(self.episode, self.channel)

class ElphyTag(BaseSignal):
    """
    Subclass of :class:`BaseSignal` corresponding to Elphy's tag channels :
    
    ``number`` : the identifier of the tag channel.
    ``x_unit`` : the unit of the x-coordinate.
    
    """
    
    def __init__(self, layout, episode, number, x_unit, sampling_frequency, start, stop, name=None):
        super(ElphyTag, self).__init__(layout, episode, sampling_frequency, start, stop, name)
        self.number = number
        self.units = [x_unit, None]
    
    def __str__(self):
        return "%s : ep_%s tag_ch_%s [%s]" % (self.layout.file.name, self.episode, self.number, self.x_unit)
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def x_unit(self):
        """
        Return the x-coordinate of the signal.
        """
        return self.units[0]
    
    @property
    def data(self):
        return self.layout.get_tag_data(self.episode, self.number)
    
    @property
    def channel(self):
        return self.number

class ElphyEvent(object):
    """
    A descriptor that store a set of events properties :
    
    ``layout`` : the :class:``ElphyLayout` object
    that extracts data from a file.
    
    ``episode`` : the episode in which the signal
    has been acquired.
    
    ``number`` : the identifier of the channel.
    
    ``x_unit`` : the unit of the x-coordinate.
    
    ``n_events`` : the number of events.
    
    ``name`` : a label to identify the event.
    
    ``times`` : a property triggering event times extraction.
    """
    
    def __init__(self, layout, episode, number, x_unit, n_events, ch_number=None, name=None):
        self.layout = layout
        self.episode = episode
        self.number = number
        self.x_unit = x_unit
        self.n_events = n_events
        self.name = name
        self.ch_number = ch_number

    def __str__(self):
        return "%s : ep_%s evt_ch_%s [%s]" % (self.layout.file.name, self.episode, self.number, self.x_unit)
    
    def __repr__(self):
        return self.__str__()

    @property
    def channel(self):
        return self.number
    
    @property
    def times(self):
        return self.layout.get_event_data(self.episode, self.number)

    @property
    def data(self):
        return self.times 

class ElphySpikeTrain(ElphyEvent):
    """
    A descriptor that store spiketrain properties :
    
    ``wf_samples`` : number of samples composing waveforms.
    
    ``wf_sampling_frequency`` : sampling frequency of waveforms.
    
    ``wf_sampling_period`` : sampling period of waveforms.
    
    ``wf_units`` : the units of the x and y coordinates of waveforms.
    
    ``t_start`` : the time before the arrival of the spike which
    corresponds to the starting time of a waveform.
    
    ``name`` : a label to identify the event.
    
    ``times`` : a property triggering event times extraction.
    
    ``waveforms`` : a property triggering waveforms extraction.
    """
    def __init__(self, layout, episode, number, x_unit, n_events, wf_sampling_frequency, wf_samples, unit_x_wf, unit_y_wf, t_start, name=None):
        super(ElphySpikeTrain, self).__init__(layout, episode, number, x_unit, n_events, name)
        self.wf_samples = wf_samples
        self.wf_sampling_frequency = wf_sampling_frequency
        assert wf_sampling_frequency, "bad sampling frequency"
        self.wf_sampling_period = 1.0 / wf_sampling_frequency
        self.wf_units = [unit_x_wf, unit_y_wf]
        self.t_start = t_start
    
    @property
    def x_unit_wf(self):
        """
        Return the x-coordinate of waveforms.
        """
        return self.wf_units[0]
    
    @property
    def y_unit_wf(self):
        """
        Return the y-coordinate of waveforms.
        """
        return self.wf_units[1]
    
    @property
    def times(self):
        return self.layout.get_spiketrain_data(self.episode, self.number)
    
    @property
    def waveforms(self):
        return self.layout.get_waveform_data(self.episode, self.number) if self.wf_samples else None




# --------------------------------------------------------
# BLOCKS


class BaseBlock(object):
    """
    Represent a chunk of file storing metadata or
    raw data. A convenient class to break down the
    structure of an Elphy file to several building
    blocks :
    
    ``layout`` : the layout containing the block.
    
    ``identifier`` : the label that identified the block.
    
    ``size`` : the size of the block.
    
    ``start`` : the file index corresponding to the starting byte of the block.
    
    ``end`` : the file index corresponding to the ending byte of the block
    
    NB : Subclassing this class is a convenient
    way to set the properties using polymorphism
    rather than a conditional structure. By this
    way each :class:`BaseBlock` type know how to
    iterate through the Elphy file and store 
    interesting data.
    """
    def __init__(self, layout, identifier, start, size):
        self.layout = layout
        self.identifier = identifier
        self.size = size
        self.start = start
        self.end = self.start + self.size - 1




class ElphyBlock(BaseBlock):
    """
    A subclass of :class:`BaseBlock`. Useful to
    store the location and size of interesting
    data within a block :
    
    ``parent_block`` : the parent block containing the block.
    
    ``header_size`` : the size of the header permitting the
    identification of the type of the block.
    
    ``data_offset`` : the file index located after the block header.
    
    ``data_size`` : the size of data located after the header.
    
    ``sub_blocks`` : the sub-blocks contained by the block.
    
    """
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="i", parent_block=None):
        super(ElphyBlock, self).__init__(layout, identifier, start, size)
        # a block may be a sub-block of another block
        self.parent_block = parent_block
        # pascal language store strings in 2 different ways 
        # ... first, if in the program the size of the string is
        # specified (fixed) then the file stores the length
        # of the string and allocate a number of bytes equal
        # to the specified size
        # ... if this size is not specified the length of the
        # string is also stored but the file allocate dynamically
        # a number of bytes equal to the actual size of the string 
        l_ident = len(self.identifier)
        if fixed_length :
            l_ident += (fixed_length - l_ident)
        self.header_size = l_ident + 1 + type_dict[size_format]
        # starting point of data located in the block
        self.data_offset = self.start + self.header_size 
        self.data_size = self.size - self.header_size
        # a block may have sub-blocks
        # it is to subclasses to initialize
        # this property
        self.sub_blocks = list()
        
    def __repr__(self):
        return "%s : size = %s, start = %s, end = %s" % (self.identifier, self.size, self.start, self.end)

    def add_sub_block(self, block):
        """
        Append a block to the sub-block list.
        """
        self.sub_blocks.append(block)




class FileInfoBlock(ElphyBlock):
    """
    Base class of all subclasses whose the purpose is to
    extract user file info stored into an Elphy file :
    
    ``header`` : the header block relative to the block.
    
    ``file`` : the file containing the block.
    
    NB : User defined metadata are not really practical.
    An Elphy script must know the order of metadata storage
    to know exactly how to retrieve these data. That's why
    it is necessary to subclass and reproduce elphy script
    commands to extract metadata relative to a protocol.
    Consequently managing a new protocol implies to refactor
    the file info extraction.  
    """
    
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="i", parent_block=None):
        super(FileInfoBlock, self).__init__(layout, identifier, start, size, fixed_length, size_format, parent_block=parent_block)
        self.header = None
        self.file = self.layout.file
    
    def get_protocol_and_version(self):
        """
        Return a tuple useful to identify the
        kind of protocol that has generated a
        file during data acquisition.
        """
        raise Exception("must be overloaded in a subclass")
    
    def get_user_file_info(self):
        """
        Return a dictionary containing all
        user file info stored in the file.
        """
        raise Exception("must be overloaded in a subclass")
    
    def get_sparsenoise_revcor(self):
        """
        Return 'REVCOR' user file info. This method is common
        to :class:`ClassicFileInfo` and :class:`MultistimFileInfo`
        because the last one is able to store this kind of metadata.
        """
        
        header = dict()
        header['n_div_x'] = read_from_char(self.file, 'h')
        header['n_div_y'] = read_from_char(self.file, 'h')
        header['gray_levels'] = read_from_char(self.file, 'h')
        header['position_x'] = read_from_char(self.file, 'ext')
        header['position_y'] = read_from_char(self.file, 'ext')
        header['length'] = read_from_char(self.file, 'ext')
        header['width'] = read_from_char(self.file, 'ext')
        header['orientation'] = read_from_char(self.file, 'ext')
        header['expansion'] = read_from_char(self.file, 'h')
        header['scotoma'] = read_from_char(self.file, 'h')
        header['seed'] = read_from_char(self.file, 'h')
        #dt_on and dt_off may not exist in old revcor formats
        rollback = self.file.tell()
        
        header['dt_on'] = read_from_char(self.file, 'ext')
        if header['dt_on'] is None :
            self.file.seek(rollback)
        
        rollback = self.file.tell()
        
        header['dt_off'] = read_from_char(self.file, 'ext')
        if header['dt_off'] is None :
            self.file.seek(rollback)
            
        return header

class ClassicFileInfo(FileInfoBlock):
    """
    Extract user file info stored into an Elphy file corresponding to
    sparse noise (revcor), moving bar and flashbar protocols.
    """
    
    def detect_protocol_from_name(self, path):
        pattern = "\d{4}(\d+|\D)\D"
        codes = {
            'r':'sparsenoise',
            'o':'movingbar',
            'f':'flashbar',
            'm':'multistim' # here just for assertion
        }
        filename = path.split(path)[1]
        match = re.search(pattern, path)
        if hasattr(match, 'end') :
            code = codes.get(path[match.end() - 1].lower(), None)
            assert code != 'm', "multistim file detected"
            return code 
        elif 'spt' in filename.lower() :
            return 'spontaneousactivity'
        else :
            return None
    
    def get_protocol_and_version(self):
        if self.layout and self.layout.info_block :
            self.file.seek(self.layout.info_block.data_offset)
            version = self.get_title()
            if version in ['REVCOR1', 'REVCOR2', 'REVCOR + PAIRING'] :
                name = "sparsenoise"
            elif version in ['BARFLASH'] :
                name = "flashbar"
            elif version in ['ORISTIM', 'ORISTM', 'ORISTM1', 'ORITUN'] :
                name = "movingbar"
            else :
                name = self.detect_protocol_from_name(self.file.name)
            self.file.seek(0)
            return name, version
        return None, None
    
    def get_title(self):
        title_length, title = struct.unpack('<B20s', self.file.read(21))
        return unicode(title[0:title_length])
    
    def get_user_file_info(self):
        header = dict()
        if self.layout and self.layout.info_block :
            self.file.seek(self.layout.info_block.data_offset)
            header['title'] = self.get_title()
            
            # test the protocol name to trigger
            # the right header extraction
            if self.layout.elphy_file.protocol == 'sparsenoise' :
                header.update(self.get_sparsenoise_revcor())
            elif self.layout.elphy_file.protocol == 'flashbar' :
                header.update(self.get_flashbar_header())
            elif self.layout.elphy_file.protocol == 'movingbar' :
                header.update(self.get_movingbar_header())

            self.file.seek(0)
        return header
    
    def get_flashbar_header(self):
        header = dict()
        orientations = list()
        tmp = self.file.tell()
        for _ in range(0, 50) :
            l, ori = struct.unpack('<B5s', self.file.read(6))
            try :
                orientations.append(float(ori[0:l]))
            except :
                return header
        header['orientations'] = orientations  if orientations else None
        self.file.seek(tmp + 50 * 6)
        _tmp = read_from_char(self.file, 'h')
        header['number_of_orientations'] = _tmp if tmp < 0 else None
        _tmp = read_from_char(self.file, 'h')
        header['number_of_repetitions'] = _tmp if tmp < 0 else None
        header['position_x'] = read_from_char(self.file, 'ext')
        header['position_y'] = read_from_char(self.file, 'ext')
        header['length'] = read_from_char(self.file, 'ext')
        header['width'] = read_from_char(self.file, 'ext')
        header['orientation'] = read_from_char(self.file, 'ext')
        header['excursion'] = read_from_char(self.file, 'i')
        header['dt_on'] = None
        return header
    
    def get_movingbar_header(self):
        header = dict()
        orientations = list()
        tmp = self.file.tell()
        for _ in range(0, 50) :
            l, ori = struct.unpack('<B5s', self.file.read(6))
            orientations.append(float(ori[0:l]))
        header['orientations'] = orientations if orientations else None
        self.file.seek(tmp + 50 * 6)
        
        _tmp = read_from_char(self.file, 'h')
        header['number_of_orientations'] = _tmp if tmp < 0 else None
        
        _tmp = read_from_char(self.file, 'h')
        header['number_of_repetitions'] = _tmp if tmp < 0 else None
        header['position_x'] = read_from_char(self.file, 'ext')
        header['position_y'] = read_from_char(self.file, 'ext')
        header['length'] = read_from_char(self.file, 'ext')
        header['width'] = read_from_char(self.file, 'ext')
        header['orientation'] = read_from_char(self.file, 'ext')
        header['excursion'] = read_from_char(self.file, 'h')
        header['speed'] = read_from_char(self.file, 'h')
        header['dim_x'] = read_from_char(self.file, 'h')
        header['dim_y'] = read_from_char(self.file, 'h')
        return header

class MultistimFileInfo(FileInfoBlock):
    
    def get_protocol_and_version(self):
        # test if there is an available info_block
        if self.layout and self.layout.info_block :
            # go to the info_block
            sub_block = self.layout.info_block
            self.file.seek(sub_block.data_offset)
            
            #get the first four parameters
            #acqLGN = read_from_char(self.file, 'i')
            center = read_from_char(self.file, 'i')
            surround = read_from_char(self.file, 'i')
            version = self.get_title()
            
            # test the type of protocol from
            # center and surround parameters
            if (surround >= 2) :
                name = None
                version = None
            else :
                if center == 2 :
                    name = "sparsenoise"
                elif center == 3 :
                    name = "densenoise"
                elif center == 4 :
                    name = "densenoise"
                elif center == 5 :
                    name = "grating"
                else :
                    name = None
                    version = None

            self.file.seek(0)
            return name, version
        return None, None
    
    def get_title(self):
        title_length = read_from_char(self.file, 'B')
        title, = struct.unpack('<%ss' % title_length, self.file.read(title_length))
        self.file.seek(self.file.tell() + 255 - title_length)
        return unicode(title)
    
    def get_user_file_info(self):
        header = dict()
        if self.layout and self.layout.info_block :
            # go to the info_block
            sub_block = self.layout.info_block
            self.file.seek(sub_block.data_offset)
            
            #get the first four parameters
            acqLGN = read_from_char(self.file, 'i')
            center = read_from_char(self.file, 'i')
            surround = read_from_char(self.file, 'i')
            #store info in the header
            header['acqLGN'] = acqLGN
            header['center'] = center
            header['surround'] = surround
            if not (header['surround'] >= 2) :
                header.update(self.get_center_header(center))
            self.file.seek(0)
        return header
    
    def get_center_header(self, code):
        #get file info corresponding 
        #to the executed protocol
        #for the center first ...
        if code == 0 :
            return self.get_sparsenoise_revcor()
        elif code == 2 :
            return self.get_sparsenoise_center()
        elif code == 3 :
            return self.get_densenoise_center(True)
        elif code == 4 :
            return self.get_densenoise_center(False)
        elif code == 5 :
            return dict()
            # return self.get_grating_center()
        else :
            return dict()
    
    def get_surround_header(self, code):
        #then the surround
        if code == 2 :
            return self.get_sparsenoise_surround()
        elif code == 3 :
            return self.get_densenoise_surround(True)
        elif code == 4 :
            return self.get_densenoise_surround(False)
        elif code == 5 :
            raise NotImplementedError()
            return self.get_grating_center()
        else :
            return dict()
    
    def get_center_surround(self, center, surround):
        header = dict()
        header['stim_center'] = self.get_center_header(center)
        header['stim_surround'] = self.get_surround_header(surround)
        return header
    
    def get_sparsenoise_center(self):
        header = dict()
        header['title'] = self.get_title()
        header['number_of_sequences'] = read_from_char(self.file, 'i')
        header['pretrigger_duration'] = read_from_char(self.file, 'ext')
        header['n_div_x'] = read_from_char(self.file, 'h')
        header['n_div_y'] = read_from_char(self.file, 'h')
        header['gray_levels'] = read_from_char(self.file, 'h')
        header['position_x'] = read_from_char(self.file, 'ext')
        header['position_y'] = read_from_char(self.file, 'ext')
        header['length'] = read_from_char(self.file, 'ext')
        header['width'] = read_from_char(self.file, 'ext')
        header['orientation'] = read_from_char(self.file, 'ext')
        header['expansion'] = read_from_char(self.file, 'h')
        header['scotoma'] = read_from_char(self.file, 'h')
        header['seed'] = read_from_char(self.file, 'h')
        header['luminance_1'] = read_from_char(self.file, 'ext')
        header['luminance_2'] = read_from_char(self.file, 'ext') 
        header['dt_count'] = read_from_char(self.file, 'i')
        dt_array = list()
        for _ in range(0, header['dt_count']) :
            dt_array.append(read_from_char(self.file, 'ext'))
        header['dt_on'] = dt_array if dt_array else None
        header['dt_off'] = read_from_char(self.file, 'ext') 
        return header
    
    def get_sparsenoise_surround(self):
        header = dict()
        header['title_surround'] = self.get_title()
        header['gap'] = read_from_char(self.file, 'ext') 
        header['n_div_x'] = read_from_char(self.file, 'h') 
        header['n_div_y'] = read_from_char(self.file, 'h') 
        header['gray_levels'] = read_from_char(self.file, 'h') 
        header['expansion'] = read_from_char(self.file, 'h')
        header['scotoma'] = read_from_char(self.file, 'h')
        header['seed'] = read_from_char(self.file, 'h')
        header['luminance_1'] = read_from_char(self.file, 'ext')
        header['luminance_2'] = read_from_char(self.file, 'ext') 
        header['dt_on'] = read_from_char(self.file, 'ext')
        header['dt_off'] = read_from_char(self.file, 'ext')
        return header 
    
    def get_densenoise_center(self, is_binary):
        header = dict()
        header['stimulus_type'] = "B" if is_binary else "T"
        header['title'] = self.get_title()
        _tmp = read_from_char(self.file, 'i')
        header['number_of_sequences'] = _tmp if _tmp < 0 else None
        rollback = self.file.tell()
        header['stimulus_duration'] = read_from_char(self.file, 'ext')
        if header['stimulus_duration'] is None :
            self.file.seek(rollback)
        header['pretrigger_duration'] = read_from_char(self.file, 'ext')
        header['n_div_x'] = read_from_char(self.file, 'h')
        header['n_div_y'] = read_from_char(self.file, 'h')          
        header['position_x'] = read_from_char(self.file, 'ext')
        header['position_y'] = read_from_char(self.file, 'ext')
        header['length'] = read_from_char(self.file, 'ext')
        header['width'] = read_from_char(self.file, 'ext')
        header['orientation'] = read_from_char(self.file, 'ext')    
        header['expansion'] = read_from_char(self.file, 'h')
        header['seed'] = read_from_char(self.file, 'h')
        header['luminance_1'] = read_from_char(self.file, 'ext')
        header['luminance_2'] = read_from_char(self.file, 'ext') 
        header['dt_on'] = read_from_char(self.file, 'ext')
        header['dt_off'] = read_from_char(self.file, 'ext')    
        return header 
    
    def get_densenoise_surround(self, is_binary):
        header = dict()
        header['title_surround'] = self.get_title()
        header['gap'] = read_from_char(self.file, 'ext')
        header['n_div_x'] = read_from_char(self.file, 'h')
        header['n_div_y'] = read_from_char(self.file, 'h')            
        header['expansion'] = read_from_char(self.file, 'h')
        header['seed'] = read_from_char(self.file, 'h')
        header['luminance_1'] = read_from_char(self.file, 'ext')
        header['luminance_2'] = read_from_char(self.file, 'ext') 
        header['dt_on'] = read_from_char(self.file, 'ext')
        header['dt_off'] = read_from_char(self.file, 'ext')    
        return header
    
    def get_grating_center(self):
        pass
    
    def get_grating_surround(self):
        pass

class Header(ElphyBlock):
    """
    A convenient subclass of :class:`Block` to store
    Elphy file header properties.
    
    NB : Subclassing this class is a convenient
    way to set the properties of the header using
    polymorphism rather than a conditional structure.
    """
    def __init__(self, layout, identifier, size, fixed_length=None, size_format="i"):
        super(Header, self).__init__(layout, identifier, 0, size, fixed_length, size_format)

class Acquis1Header(Header):
    """
    A subclass of :class:`Header` used to
    identify the 'ACQUIS1/GS/1991' format.
    Whereas more recent format, the header
    contains all data relative to episodes,
    channels and traces :
    
    ``n_channels`` : the number of acquisition channels.
    
    ``nbpt`` and ``nbptEx`` : parameters useful to compute the number of samples by episodes.  
    
    ``tpData`` : the data format identifier used to compute sample size.
    
    ``x_unit`` : the x-coordinate unit for all channels in an episode. 
    
    ``y_units`` : an array containing y-coordinate units for each channel in the episode.
    
    ``dX`` and ``X0`` : the scale factors necessary to retrieve the actual
    times relative to each sample in a channel. 
    
    ``dY_ar`` and ``Y0_ar``: arrays of scale factors necessary to retrieve
    the actual values relative to samples. 
    
    ``continuous`` : a boolean telling if the file has been acquired in
    continuous mode. 
    
    ``preSeqI`` : the size in bytes of the data preceding raw data.
    
    ``postSeqI`` : the size in bytes of the data preceding raw data.
    
    ``dat_length`` : the length in bytes of the data in the file.
    
    ``sample_size`` : the size in bytes of a sample.
    
    ``n_samples`` : the number of samples.
    
    ``ep_size`` : the size in bytes of an episode.
    
    ``n_episodes`` : the number of recording sequences store in the file.
    
    NB : 
    
    The size is read from the file,
    the identifier is a string containing
    15 characters and the size is encoded
    as small integer. 
    
    See file 'FicDefAc1.pas' to identify
    the parsed parameters. 
    """
    
    def __init__(self, layout):
        fileobj = layout.file
        super(Acquis1Header, self).__init__(layout, "ACQUIS1/GS/1991", 1024, 15, "h")
        
        #parse the header to store interesting data about episodes and channels
        fileobj.seek(18)
        
        #extract episode properties
        n_channels = read_from_char(fileobj, 'B')
        assert not ((n_channels < 1) or (n_channels > 16)), "bad number of channels"
        nbpt = read_from_char(fileobj, 'h')
        l_xu, x_unit = struct.unpack('<B3s', fileobj.read(4))
        #extract units for each channel
        y_units = list()
        for i in range(1, 7) :
            l_yu, y_unit = struct.unpack('<B3s', fileobj.read(4))
            y_units.append(y_unit[0:l_yu])
        
        #extract i1, i2, x1, x2 and compute dX and X0
        i1, i2 = struct.unpack('<hh', fileobj.read(4))
        x1 = read_from_char(fileobj, 'ext')
        x2 = read_from_char(fileobj, 'ext')
        if (i1 != i2) and (x1 != x2) :
            dX = (x2 - x1) / (i2 - i1)
            X0 = x1 - i1 * dX
        else :
            dX = None
            X0 = None
            # raise Exception("bad X-scale parameters")
        
        #extract j1 and j2, y1 and y2 and compute dY 
        j1 = struct.unpack('<hhhhhh', fileobj.read(12))
        j2 = struct.unpack('<hhhhhh', fileobj.read(12))
        y1 = list()
        for i in range(1, 7) :
            y1.append(read_from_char(fileobj, 'ext'))
        y2 = list()
        for i in range(1, 7) :
            y2.append(read_from_char(fileobj, 'ext'))
        dY_ar = list()
        Y0_ar = list()
        for i in range(0, n_channels) :
            # detect division by zero
            if (j1[i] != j2[i]) and (y1[i] != y2[i]) :
                dY_ar.append((y2[i] - y1[i]) / (j2[i] - j1[i]))
                Y0_ar.append(y1[i] - j1[i] * dY_ar[i])
            else :
                dY_ar.append(None)
                Y0_ar.append(None)
        
        NbMacq = read_from_char(fileobj, 'h')
        
        #fileobj.read(300) #Macq:typeTabMarqueAcq;   { 300 octets }
        max_mark = 100
        Macq = list()
        for i in range(0, max_mark) :
            Macq.append(list(struct.unpack('<ch', fileobj.read(3))))
        
        #Xmini,Xmaxi,Ymini,Ymaxi:array[1..6] of float; #fileobj.read(240)
        x_mini = list()
        for i in range(0, 6) :
            x_mini.append(read_from_char(fileobj, 'ext'))
        x_maxi = list()
        for i in range(0, 6) :
            x_maxi.append(read_from_char(fileobj, 'ext'))
        y_mini = list()
        for i in range(0, 6) :
            y_mini.append(read_from_char(fileobj, 'ext'))
        y_maxi = list()
        for i in range(0, 6) :
            y_maxi.append(read_from_char(fileobj, 'ext'))
        
        #modeA:array[1..6] of byte; #fileobj.read(6)
        modeA = list(struct.unpack('<BBBBBB', fileobj.read(6)))
        
        continuous = read_from_char(fileobj, '?')
        preSeqI, postSeqI = struct.unpack('<hh', fileobj.read(4))
        
        #EchelleSeqI:boolean; #fileobj.read(1)
        ep_scaled = read_from_char(fileobj, '?')
           
        nbptEx = read_from_char(fileobj, 'H')
        
        x1s, x2s = struct.unpack('<ff', fileobj.read(8))
        
        y1s = list()
        for i in range(0, 6):
            y1s.append(read_from_char(fileobj, 'f'))
        
        y2s = list()
        for i in range(0, 6):
            y2s.append(read_from_char(fileobj, 'f'))
        
        #fileobj.read(96)   # Xminis,Xmaxis,Yminis,Ymaxis:array[1..6] of single;
        x_minis = list()
        for i in range(0, 6) :
            x_minis.append(read_from_char(fileobj, 'f'))
        x_maxis = list()
        for i in range(0, 6) :
            x_maxis.append(read_from_char(fileobj, 'f'))
        y_minis = list()
        for i in range(0, 6) :
            y_minis.append(read_from_char(fileobj, 'f'))
        y_maxis = list()
        for i in range(0, 6) :
            y_maxis.append(read_from_char(fileobj, 'f'))
        
        n_ep = read_from_char(fileobj, 'h')
        tpData = read_from_char(fileobj, 'h')
        assert tpData in [3, 2, 1, 0], "bad sample size"
        no_analog_data = read_from_char(fileobj, '?')
        
        self.n_ep = n_ep
        self.n_channels = n_channels
        self.nbpt = nbpt
        self.i1 = i1
        self.i2 = i2
        self.x1 = x1
        self.x2 = x2
        self.dX = dX
        self.X0 = X0
        self.x_unit = x_unit[0:l_xu]
        self.dY_ar = dY_ar
        self.Y0_ar = Y0_ar
        self.y_units = y_units[0:n_channels]
        self.NbMacq = NbMacq
        self.Macq = Macq
        self.x_mini = x_mini[0:n_channels]
        self.x_maxi = x_maxi[0:n_channels]
        self.y_mini = y_mini[0:n_channels]
        self.y_maxi = y_maxi[0:n_channels]
        self.modeA = modeA
        self.continuous = continuous
        self.preSeqI = preSeqI
        self.postSeqI = postSeqI
        self.ep_scaled = ep_scaled
        self.nbptEx = nbptEx
        self.x1s = x1s
        self.x2s = x2s
        self.y1s = y1s
        self.y2s = y2s
        self.x_minis = x_minis[0:n_channels]
        self.x_maxis = x_maxis[0:n_channels]
        self.y_minis = y_minis[0:n_channels]
        self.y_maxis = y_maxis[0:n_channels]
        self.tpData = 2 if not tpData else tpData
        self.no_analog_data = no_analog_data
        
        self.dat_length = self.layout.file_size - self.layout.data_offset 
        self.sample_size = type_dict[types[tpData]] 
        if self.continuous :
            self.n_samples = self.dat_length / (self.n_channels * self.sample_size)
        else :
            self.n_samples = self.nbpt + self.nbptEx * 32768 
        ep_size = self.preSeqI + self.postSeqI
        if not self.no_analog_data :
            ep_size += self.n_samples * self.sample_size * self.n_channels
        self.ep_size = ep_size
        self.n_episodes = (self.dat_length / self.ep_size) if (self.n_samples != 0) else 0
              
class DAC2GSHeader(Header):
    """
    A subclass of :class:`Header` used to
    identify the 'DAC2/GS/2000' format.
    
    NB : the size is fixed to 20 bytes,
    the identifier is a string containing
    15 characters and the size is encoded
    as integer.
    """
    def __init__(self, layout):
        super(DAC2GSHeader, self).__init__(layout, "DAC2/GS/2000", 20, 15, "i")

class DAC2Header(Header):
    """
    A subclass of :class:`Header` used to
    identify the 'DAC2 objects' format.
    
    NB : the size is fixed to 18 bytes,
    the identifier is a string containing
    15 characters and the size is encoded
    as small integer.
    """
    def __init__(self, layout):
        super(DAC2Header, self).__init__(layout, "DAC2 objects", 18, 15, "h")

class DAC2GSMainBlock(ElphyBlock):
    """
    Subclass of :class:`Block` useful to store data corresponding to
    the 'Main' block stored in the DAC2/GS/2000 format :
    
    ``n_channels`` : the number of acquisition channels.
    
    ``nbpt`` : the number of samples by episodes.  
    
    ``tpData`` : the data format identifier used to compute sample size. 
    
    ``x_unit`` : the x-coordinate unit for all channels in an episode. 
    
    ``y_units`` : an array containing y-coordinate units for each channel in the episode.
    
    ``dX`` and ``X0`` : the scale factors necessary to retrieve the actual
    times relative to each sample in a channel. 
    
    ``dY_ar`` and ``Y0_ar``: arrays of scale factors necessary to retrieve
    the actual values relative to samples. 
    
    ``continuous`` : a boolean telling if the file has been acquired in
    continuous mode. 
    
    ``preSeqI`` : the size in bytes of the data preceding raw data.
    
    ``postSeqI`` : the size in bytes of the data preceding raw data.
    
    ``withTags`` : a boolean telling if tags are recorded.
    
    ``tagShift`` : the number of tag channels and the shift to apply
    to encoded values to retrieve acquired values.
    
    ``dat_length`` : the length in bytes of the data in the file.
    
    ``sample_size`` : the size in bytes of a sample.
    
    ``n_samples`` : the number of samples.
    
    ``ep_size`` : the size in bytes of an episode.
    
    ``n_episodes`` : the number of recording sequences store in the file.
    
    NB : see file 'FdefDac2.pas' to identify the other parsed parameters. 
    """
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="i"):
        super(DAC2GSMainBlock, self).__init__(layout, identifier, start, size, fixed_length, size_format)
        #parse the file to retrieve episodes and channels properties
        n_channels, nbpt, tpData = struct.unpack('<BiB', layout.file.read(6))
        l_xu, xu, dX, X0 = struct.unpack('<B10sdd', layout.file.read(27))
        y_units = list()
        dY_ar = list()
        Y0_ar = list()
        for _ in range(0, 16) :
            l_yu, yu, dY, Y0 = struct.unpack('<B10sdd', layout.file.read(27)) 
            y_units.append(yu[0:l_yu])
            dY_ar.append(dY)
            Y0_ar.append(Y0)
        preSeqI, postSeqI, continuous, varEp, withTags = struct.unpack('<ii???', layout.file.read(11))
        #some file doesn't precise the tagShift
        position = layout.file.tell()
        if position >= self.end :
            tagShift = 0
        else :
            tagShift = read_from_char(layout.file, 'B')
        #setup object properties
        self.n_channels = n_channels
        self.nbpt = nbpt
        self.tpData = tpData
        self.x_unit = xu[0:l_xu]
        self.dX = dX
        self.X0 = X0
        self.y_units = y_units[0:n_channels]
        self.dY_ar = dY_ar[0:n_channels]
        self.Y0_ar = Y0_ar[0:n_channels]
        self.continuous = continuous
        if self.continuous :
            self.preSeqI = 0
            self.postSeqI = 0
        else :
            self.preSeqI = preSeqI 
            self.postSeqI = postSeqI
        self.varEp = varEp
        self.withTags = withTags
        if not self.withTags :
            self.tagShift = 0
        else :
            if tagShift == 0 :
                self.tagShift = 4
            else :
                self.tagShift = tagShift
        self.sample_size = type_dict[types[self.tpData]]
        self.dat_length = self.layout.file_size - self.layout.data_offset
        if self.continuous :
            if self.n_channels > 0 :
                self.n_samples = self.dat_length / (self.n_channels * self.sample_size)
            else :
                self.n_samples = 0
        else :
            self.n_samples = self.nbpt
        
        self.ep_size = self.preSeqI + self.postSeqI + self.n_samples * self.sample_size * self.n_channels
        self.n_episodes = self.dat_length / self.ep_size if (self.n_samples != 0) else 0 
    
class DAC2GSEpisodeBlock(ElphyBlock):
    """
    Subclass of :class:`Block` useful to store data corresponding to
    'DAC2SEQ' blocks stored in the DAC2/GS/2000 format.
    
    ``n_channels`` : the number of acquisition channels.
    
    ``nbpt`` : the number of samples by episodes.  
    
    ``tpData`` : the data format identifier used to compute the sample size. 
    
    ``x_unit`` : the x-coordinate unit for all channels in an episode. 
    
    ``y_units`` : an array containing y-coordinate units for each channel in the episode.
    
    ``dX`` and ``X0`` : the scale factors necessary to retrieve the actual
    times relative to each sample in a channel. 
    
    ``dY_ar`` and ``Y0_ar``: arrays of scale factors necessary to retrieve
    the actual values relative to samples. 
    
    ``postSeqI`` : the size in bytes of the data preceding raw data.
    
    NB : see file 'FdefDac2.pas' to identify the parsed parameters. 
    """
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="i"):
        main = layout.main_block
        n_channels, nbpt, tpData, postSeqI = struct.unpack('<BiBi', layout.file.read(10))
        l_xu, xu, dX, X0 = struct.unpack('<B10sdd', layout.file.read(27))
        y_units = list()
        dY_ar = list()
        Y0_ar = list()
        for _ in range(0, 16) :
            l_yu, yu, dY, Y0 = struct.unpack('<B10sdd', layout.file.read(27)) 
            y_units.append(yu[0:l_yu])
            dY_ar.append(dY)
            Y0_ar.append(Y0)

        super(DAC2GSEpisodeBlock, self).__init__(layout, identifier, start, layout.main_block.ep_size, fixed_length, size_format)
        
        self.n_channels = main.n_channels
        self.nbpt = main.nbpt
        self.tpData = main.tpData
        if not main.continuous :
            self.postSeqI = postSeqI
            self.x_unit = xu[0:l_xu]
            self.dX = dX
            self.X0 = X0
            self.y_units = y_units[0:n_channels]
            self.dY_ar = dY_ar[0:n_channels]
            self.Y0_ar = Y0_ar[0:n_channels]
        else :
            self.postSeqI = 0
            self.x_unit = main.x_unit
            self.dX = main.dX
            self.X0 = main.X0
            self.y_units = main.y_units
            self.dY_ar = main.dY_ar
            self.Y0_ar = main.Y0_ar


class DAC2EpisodeBlock(ElphyBlock):
    """
    Subclass of :class:`Block` useful to store data corresponding to
    'B_Ep' blocks stored in the last version of Elphy format :
    
    ``ep_block`` : a shortcut the the 'Ep' sub-block.
    
    ``ch_block`` : a shortcut the the 'Adc' sub-block. 
    
    ``ks_block`` : a shortcut the the 'KSamp' sub-block. 
    
    ``kt_block`` : a shortcut the the 'Ktype' sub-block.
    
    """
    
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="l"):
        super(DAC2EpisodeBlock, self).__init__(layout, identifier, start, size, fixed_length, size_format)
        self.ep_block = None
        self.ch_block = None
        self.ks_block = None
        self.kt_block = None

    def set_episode_block(self):
        blocks = self.layout.get_blocks_of_type('Ep', target_blocks=self.sub_blocks)
        self.ep_block = blocks[0] if blocks else None 
        
    def set_channel_block(self):
        blocks = self.layout.get_blocks_of_type('Adc', target_blocks=self.sub_blocks)
        self.ch_block = blocks[0] if blocks else None
    
    def set_sub_sampling_block(self):
        blocks = self.layout.get_blocks_of_type('Ksamp', target_blocks=self.sub_blocks)
        self.ks_block = blocks[0] if blocks else None 
    
    def set_sample_size_block(self):
        blocks = self.layout.get_blocks_of_type('Ktype', target_blocks=self.sub_blocks)
        self.kt_block = blocks[0] if blocks else None 
    

class DummyDataBlock(BaseBlock):
    """
    Subclass of :class:`BaseBlock` useful to
    identify chunk of blocks that are actually
    corresponding to acquired data.
    """
    pass


class DAC2RDataBlock(ElphyBlock):
    """
    Subclass of :class:`Block` useful to store data corresponding to
    'RDATA' blocks stored in the last version of Elphy format :
    
    ``data_start`` : the starting point of raw data.
    
    NB : This kind of block is preceeded by a structure which size is encoded
    as a 2 bytes unsigned short. Consequently, data start at data_offset plus
    the size. 
    """
    
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="l"):
        super(DAC2RDataBlock, self).__init__(layout, identifier, start, size, fixed_length, size_format)
        self.data_start = self.data_offset + read_from_char(layout.file, 'H')

class DAC2CyberTagBlock(ElphyBlock):
    """
    Subclass of :class:`Block` useful to store data corresponding to
    'RCyberTag' blocks stored in the last version of Elphy format :
    
    ``data_start`` : the starting point of raw data.
    
    NB : This kind of block is preceeded by a structure which size is encoded
    as a 2 bytes unsigned short. Consequently, data start at data_offset plus
    the size. 
    """
    
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="l"):
        super(DAC2CyberTagBlock, self).__init__(layout, identifier, start, size, fixed_length, size_format)
        self.data_start = self.data_offset + read_from_char(layout.file, 'H')

class DAC2EventBlock(ElphyBlock):
    """
    Subclass of :class:`Block` useful to store
    data corresponding to 'REVT' blocks stored
    in the last version of Elphy format :
    
    ``data_start`` : the starting point of raw data.
    
    ``n_evt_channels`` : the number of channels used to acquire events.
    
    ``n_events`` : an array containing the number of events for each event channel.
    
    """
    
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="l"):
        super(DAC2EventBlock, self).__init__(layout, identifier, start, size, fixed_length, size_format)
        fileobj = self.layout.file
        jump = self.data_offset + read_from_char(fileobj, 'H')
        fileobj.seek(jump)
        
        #extract the number of event channel
        self.n_evt_channels = read_from_char(fileobj, 'i')
        
        # extract for each event channel
        # the corresponding number of events
        n_events = list()
        for _ in range(0, self.n_evt_channels) :
            n_events.append(read_from_char(fileobj, 'i'))
        self.n_events = n_events
        
        self.data_start = fileobj.tell()


class DAC2SpikeBlock(DAC2EventBlock):
    """
    Subclass of :class:`DAC2EventBlock` useful
    to identify 'RSPK' and make the distinction
    with 'REVT' blocks stored in the last version
    of Elphy format. 
    """
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="l"):
        super(DAC2SpikeBlock, self).__init__(layout, identifier, start, size, fixed_length, size_format)
        fileobj = self.layout.file
        jump = self.data_offset
        fileobj.seek(jump) # go to SpikeBlock
        jump = self.data_offset + read_from_char(fileobj, 'h')
        fileobj.seek(jump)
        #extract the number of event channel
        self.n_evt_channels = read_from_char(fileobj, 'i')
        # extract for each event channel
        # the corresponding number of events
        n_events = list()
        for _ in range(0, self.n_evt_channels) :
            n_events.append(read_from_char(fileobj, 'i'))
        self.n_events = n_events
        self.data_start = fileobj.tell()



class DAC2WaveFormBlock(ElphyBlock):
    """
    Subclass of :class:`Block` useful to store data corresponding to
    'RspkWave' blocks stored in the last version of Elphy format :
    
    ``data_start`` : the starting point of raw data.
    
    ``n_spk_channels`` : the number of channels used to acquire spiketrains.
    
    ``n_spikes`` : an array containing the number of spikes for each spiketrain.
    
    ``pre_trigger`` : the number of samples of a waveform arriving before a spike.
    
    ``wavelength`` : the number of samples in a waveform.
    
    """
    
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="l"):
        super(DAC2WaveFormBlock, self).__init__(layout, identifier, start, size, fixed_length, size_format)
        fileobj = self.layout.file
        jump = self.data_offset + read_from_char(fileobj, 'H')
        fileobj.seek(jump)
        self.wavelength = read_from_char(fileobj, 'i')
        self.pre_trigger = read_from_char(fileobj, 'i')
        self.n_spk_channels = read_from_char(fileobj, 'i')
        n_spikes = list()
        for _ in range(0, self.n_spk_channels) :
            n_spikes.append(read_from_char(fileobj, 'i'))
        self.n_spikes = n_spikes
        self.data_start = fileobj.tell()



class DAC2EpSubBlock(ElphyBlock):
    """
    Subclass of :class:`Block` useful to retrieve data corresponding
    to a 'Ep' sub-block stored in the last version of Elphy format :
    
    ``n_channels`` : the number of acquisition channels.
    
    ``nbpt`` : the number of samples by episodes 
    
    ``tpData`` : the data format identifier used to store signal samples. 
    
    ``x_unit`` : the x-coordinate unit for all channels in an episode. 
    
    ``dX`` and ``X0`` : the scale factors necessary to retrieve the actual
    times relative to each sample in a channel. 
    
    ``continuous`` : a boolean telling if the file has been acquired in
    continuous mode. 
    
    ``tag_mode`` : identify the way tags are stored in a file. 
    
    ``tag_shift`` : the number of bits that tags occupy in a 16-bits sample
    and the shift necessary to do to retrieve the value of the sample. 
    
    ``dX_wf`` and ``X0_wf``: the scale factors necessary to retrieve the actual
    times relative to each waveforms. 
    
    ``dY_wf`` and ``Y0_wf``: the scale factors necessary to retrieve the actual
    values relative to waveform samples.  
    
    ``x_unit_wf`` and ``y_unit_wf``: the unit of x and y coordinates for all waveforms in an episode. 
    """
    
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="l", parent_block=None):
        super(DAC2EpSubBlock, self).__init__(layout, identifier, start, size, fixed_length, size_format, parent_block=parent_block)
        fileobj = self.layout.file
        n_channels, nbpt, tpData, l_xu, x_unit, dX, X0 = struct.unpack('<BiBB10sdd', fileobj.read(33))
        continuous, tag_mode, tag_shift = struct.unpack('<?BB', fileobj.read(3))
        DxuSpk, X0uSpk, nbSpk, DyuSpk, Y0uSpk, l_xuspk, unitXSpk, l_yuspk, unitYSpk = struct.unpack('<ddiddB10sB10s', fileobj.read(58))
        cyber_time, pc_time = struct.unpack('<dI', fileobj.read(12))
        # necessary properties to reconstruct
        # signals stored into the file
        self.n_channels = n_channels
        self.nbpt = nbpt
        self.tpData = tpData
        self.x_unit = x_unit[0:l_xu]
        self.dX = dX
        self.X0 = X0
        self.continuous = continuous
        self.tag_mode = tag_mode
        self.tag_shift = tag_shift if self.tag_mode == 1 else 0
        # following properties are valid
        # when using multielectrode system
        # named BlackRock / Cyberkinetics
        #if fileobj.tell() < self.end :
        self.dX_wf = DxuSpk
        self.X0_wf = X0uSpk
        self.n_spikes = nbSpk
        self.dY_wf = DyuSpk
        self.Y0_wf = Y0uSpk
        self.x_unit_wf = unitXSpk[0:l_xuspk]
        self.y_unit_wf = unitYSpk[0:l_yuspk]
        self.cyber_time = cyber_time
        self.pc_time = pc_time

        

class DAC2AdcSubBlock(ElphyBlock):
    """
    Subclass of :class:`SubBlock` useful to retrieve data corresponding
    to a 'Adc' sub-block stored in the last version of Elphy format : 
    
    ``y_units`` : an array containing all y-coordinates for each channel.
    
    ``dY_ar`` and ``Y0_ar`` : arrays containing scaling factors  for each
    channel useful to compute the actual value of a signal sample.
    
    """
    
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="l", parent_block=None):
        super(DAC2AdcSubBlock, self).__init__(layout, identifier, start, size, fixed_length, size_format, parent_block=parent_block)
        fileobj = self.layout.file
        #fileobj.seek(start + len(identifier) + 1)
        ep_block, = [k for k in self.parent_block.sub_blocks if k.identifier.startswith('Ep')]
        n_channels = ep_block.n_channels
        self.y_units = list()
        self.dY_ar = list()
        self.Y0_ar = list()
        for _ in range(0, n_channels) :
            l_yu, y_unit, dY, Y0 = struct.unpack('<B10sdd', fileobj.read(27))
            self.y_units.append(y_unit[0:l_yu])
            self.dY_ar.append(dY)
            self.Y0_ar.append(Y0)


class DAC2KSampSubBlock(ElphyBlock):
    """
    Subclass of :class:`SubBlock` useful to retrieve data corresponding
    to a 'Ksamp' sub-block stored in the last version of Elphy format :
    
    ``k_sampling`` : an array containing all sub-sampling factors
    corresponding to each acquired channel. If a factor is equal to
    zero, then the channel has been converted into an event channel.
    
    """
    
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="l", parent_block=None):
        super(DAC2KSampSubBlock, self).__init__(layout, identifier, start, size, fixed_length, size_format, parent_block=parent_block)
        fileobj = self.layout.file
        ep_block, = [k for k in self.parent_block.sub_blocks if k.identifier.startswith('Ep')]
        n_channels = ep_block.n_channels
        k_sampling = list()
        for _ in range(0, n_channels) :
            k_sampling.append(read_from_char(fileobj, "H"))
        self.k_sampling = k_sampling


        
class DAC2KTypeSubBlock(ElphyBlock):
    """
    Subclass of :class:`SubBlock` useful to retrieve data corresponding
    to a 'Ktype' sub-block stored in the last version of Elphy format :
    
    ``k_types`` : an array containing all data formats identifier used
    to compute sample size.
    """
    
    def __init__(self, layout, identifier, start, size, fixed_length=None, size_format="l", parent_block=None):
        super(DAC2KTypeSubBlock, self).__init__(layout, identifier, start, size, fixed_length, size_format, parent_block=parent_block)
        fileobj = self.layout.file
        ep_block, = [k for k in self.parent_block.sub_blocks if k.identifier.startswith('Ep')]
        n_channels = ep_block.n_channels
        k_types = list()
        for _ in range(0, n_channels) :
            k_types.append(read_from_char(fileobj, "B"))
        self.k_types = k_types






# --------------------------------------------------------
# UTILS

#symbols of types that could 
#encode a value in an elphy file
types = (
    'B',
    'b',
    'h',
    'H',
    'l',
    'f',
    'real48',
    'd',
    'ext',
    's_complex',
    'd_complex',
    'complex',
    'none'
)

#a dictionary linking python.struct
#formats to their actual size in bytes
type_dict = {
    'c':1,
    'b':1,
    'B':1,
    '?':1,
    'h':2,
    'H':2,
    'i':4,
    'I':4,
    'l':4,
    'L':4,
    'q':8,
    'Q':8,
    'f':4,
    'd':8,
    'H+l':6,
    'ext':10,
    'real48':6,
    's_complex':8,
    'd_complex':16,
    'complex':20,
    'none':0
}

#a dictionary liking python.struct
#formats to numpy formats
numpy_map = {
    'b':np.int8,
    'B':np.uint8,
    'h':np.int16,
    'H':np.uint16,
    'i':np.int32,
    'I':np.uint32,
    'l':np.int32,
    'L':np.uint32,
    'q':np.int64,
    'Q':np.uint64,
    'f':np.float32,
    'd':np.float64,
    'H+l':6,
    'ext':10,
    'real48':6,
    'SComp':8,
    'DComp':16,
    'Comp':20,
    'none':0
}

def read_from_char(data, type_char):
    """
    Return the value corresponding
    to the specified character type.
    """
    n_bytes = type_dict[type_char]
    ascii = data.read(n_bytes) if isinstance(data, file) else data
    if type_char != 'ext':
        try :
            value = struct.unpack('<%s' % type_char, ascii)[0]
        except :
            # the value could not been read
            # because the value is not compatible
            # with the specified type
            value = None    
    else :
        try :
            value = float(ascii)
        except :
            value = None
    return value

def least_common_multiple(a, b):
    """
    Return the value of the least common multiple.
    """
    return (a * b) / gcd(a, b)



# --------------------------------------------------------
# LAYOUT

b_float = 'f8'
b_int = 'i2'

class ElphyLayout(object):
    """
    A convenient class to know how data
    are organised into an Elphy file :
    
    ``elphy_file`` : a :class:`ElphyFile`
    asking file introspection.
    
    ``blocks`` : a set of :class:``BaseBlock`
    objects partitioning  a file and extracting
    some useful metadata.
    
    ``ìnfo_block`` : a shortcut to a :class:`FileInfoBlock`
    object containing metadata describing a recording
    protocol (sparsenoise, densenoise, movingbar or flashbar)
    
    ``data_blocks`` : a shortcut to access directly
    blocks containing raw data.
    
    NB : Subclassing this class is a convenient
    way to retrieve blocks constituting a file,
    their relative information and location of
    raw data using polymorphism rather than a
    conditional structure.
    """
    
    def __init__(self, elphy_file):
        self.elphy_file = elphy_file
        self.blocks = list()
        self.info_block = None
        self.data_blocks = None
    
    @property
    def file(self):
        return self.elphy_file.file
    
    @property
    def file_size(self):
        return self.elphy_file.file_size
    
    def is_continuous(self):
        return self.is_continuous()
    
    def add_block(self, block):
        self.blocks.append(block)
    
    @property
    def header(self):
        return self.blocks[0]
    
    def get_blocks_of_type(self, identifier, target_blocks=None):
        blocks = self.blocks if target_blocks is None else target_blocks
        return [k for k in blocks if (k.identifier == identifier)]
    
    def set_info_block(self):
        raise NotImplementedError('must be overloaded in a subclass')
    
    def set_data_blocks(self):
        raise NotImplementedError('must be overloaded in a subclass')
    
    def get_tag(self, episode, tag_channel):
        raise NotImplementedError('must be overloaded in a subclass')

    @property
    def n_episodes(self):
        raise NotImplementedError('must be overloaded in a subclass')
    
    def n_channels(self, episode):
        raise NotImplementedError('must be overloaded in a subclass')
    
    def n_tags(self, episode):
        raise NotImplementedError('must be overloaded in a subclass')
    
    def n_samples(self, episode, channel):
        raise NotImplementedError('must be overloaded in a subclass')
    
    def sample_type(self, ep, ch):
        raise NotImplementedError('must be overloaded in a subclass')
    
    def sample_size(self, ep, ch):
        symbol = self.sample_symbol(ep, ch)
        return type_dict[symbol]
    
    def sample_symbol(self, ep, ch):
        tp = self.sample_type(ep, ch)
        try:
            return types[tp]
        except :
            return 'h'

    def sampling_period(self, ep, ch):
        raise NotImplementedError('must be overloaded in a subclass')

    def x_scale_factors(self, ep, ch):
        raise NotImplementedError('must be overloaded in a subclass')

    def y_scale_factors(self, ep, ch):
        raise NotImplementedError('must be overloaded in a subclass')
    
    def x_tag_scale_factors(self, ep):
        raise NotImplementedError('must be overloaded in a subclass')
    
    def x_unit(self, ep, ch):
        raise NotImplementedError('must be overloaded in a subclass')

    def y_unit(self, ep, ch):
        raise NotImplementedError('must be overloaded in a subclass')

    def tag_shift(self, ep):
        raise NotImplementedError('must be overloaded in a subclass')
    
    def get_channel_for_tags(self, ep):
        raise NotImplementedError('must be overloaded in a subclass')
    
    def get_signal(self, episode, channel):
        """
        Return the signal description relative
        to the specified episode and channel.
        """
        assert episode in range(1, self.n_episodes + 1)
        assert channel in range(1, self.n_channels(episode) + 1)
        t_start = 0
        sampling_period = self.sampling_period(episode, channel)
        t_stop = sampling_period * self.n_samples(episode, channel)
        return ElphySignal(
            self,
            episode,
            channel,
            self.x_unit(episode, channel),
            self.y_unit(episode, channel),
            1 / sampling_period,
            t_start,
            t_stop
        )
    
    def create_channel_mask(self, ep):
        """
        Return the minimal pattern of channel numbers
        representing the succession of channels in the
        multiplexed data. It is necessary to do the mapping
        between a sample stored in the file and its relative
        channel.
        """
        raise NotImplementedError('must be overloaded in a subclass')
    
    def get_data_blocks(self, ep):
        """
        Return a set of :class:`DummyDataBlock` instances
        that defined the actual location of samples in blocks
        encapsulating raw data.
        """
        raise NotImplementedError('must be overloaded in a subclass')
    
    def create_bit_mask(self, ep, ch):
        """
        Build a mask to apply on the entire episode
        in order to only keep values corresponding
        to the specified channel.
        """
        ch_mask = self.create_channel_mask(ep)
        _mask = list()
        for _ch in ch_mask :
            size = self.sample_size(ep, _ch)
            val = 1 if _ch == ch else 0 
            for _ in xrange(0, size) :
                _mask.append(val)
        return np.array(_mask)
    
    def load_bytes(self, data_blocks, dtype='<i1', start=None, end=None, expected_size=None):
        """
        Return list of bytes contained
        in the specified set of blocks.
        
        NB : load all data as files cannot exceed 4Gb 
             find later other solutions to spare memory.
        """
        chunks = list()
        raw = ''
        # keep only data blocks having
        # a size greater than zero
        blocks = [k for k in data_blocks if k.size > 0]
        for data_block in blocks :
            self.file.seek(data_block.start)
            raw = self.file.read(data_block.size)[0:expected_size]
            databytes = np.frombuffer(raw, dtype=dtype)
            chunks.append(databytes)
        # concatenate all chunks and return
        # the specified slice
        if len(chunks)>0 :
            databytes = np.concatenate(chunks)
            return databytes[start:end]
        else :
            return np.array([])
        
    
    def reshape_bytes(self, databytes, reshape, datatypes, order='<'):
        """
        Reshape a numpy array containing a set of databytes.
        """
        assert datatypes and len(datatypes) == len(reshape), "datatypes are not well defined"
        
        l_bytes = len(databytes)
        
        #create the mask for each shape
        shape_mask = list()
        for shape in reshape :
            for _ in xrange(1, shape + 1) :
                shape_mask.append(shape)
        
        #create a set of masks to extract data
        bit_masks = list()
        for shape in reshape :
            bit_mask = list()
            for value in shape_mask :
                bit = 1 if (value == shape) else 0
                bit_mask.append(bit)
            bit_masks.append(np.array(bit_mask))
         
        #extract data
        
        n_samples = l_bytes / np.sum(reshape)
        data = np.empty([len(reshape), n_samples], dtype=(int, int))
        for index, bit_mask in enumerate(bit_masks) :
            tmp = self.filter_bytes(databytes, bit_mask)
            tp = '%s%s%s' % (order, datatypes[index], reshape[index])
            data[index] = np.frombuffer(tmp, dtype=tp)
        
        return data.T

    def filter_bytes(self, databytes, bit_mask):
        """
        Detect from a bit mask which bits
        to keep to recompose the signal.
        """
        n_bytes = len(databytes)
        mask = np.ones(n_bytes, dtype=int)
        np.putmask(mask, mask, bit_mask)
        to_keep = np.where(mask > 0)[0]
        return databytes.take(to_keep)
    
    def load_channel_data(self, ep, ch):
        """
        Return a numpy array containing the
        list of bytes corresponding to the
        specified episode and channel.
        """
        #memorise the sample size and symbol
        sample_size = self.sample_size(ep, ch)
        sample_symbol = self.sample_symbol(ep, ch)
        
        #create a bit mask to define which
        #sample to keep from the file
        bit_mask = self.create_bit_mask(ep, ch)
        
        #load all bytes contained in an episode
        data_blocks = self.get_data_blocks(ep)
        databytes = self.load_bytes(data_blocks)
        raw = self.filter_bytes(databytes, bit_mask)
        
        #reshape bytes from the sample size
        dt = np.dtype(numpy_map[sample_symbol])
        dt.newbyteorder('<')
        return np.frombuffer(raw.reshape([len(raw) / sample_size, sample_size]), dt)
    
    def apply_op(self, np_array, value, op_type):
        """
        A convenient function to apply an operator
        over all elements of a numpy array.
        """
        if op_type == "shift_right" :
            return np_array >> value
        elif op_type == "shift_left" :
            return np_array << value
        elif op_type == "mask" :
            return np_array & value
        else :
            return np_array
    
    def get_tag_mask(self, tag_ch, tag_mode):
        """
        Return a mask useful to retrieve
        bits that encode a tag channel.
        """
        if  tag_mode == 1 :
            tag_mask = 0b01 if (tag_ch == 1) else 0b10
        elif tag_mode in [2, 3] :
            ar_mask = np.zeros(16, dtype=int)
            ar_mask[tag_ch - 1] = 1
            st = "0b" + ''.join(np.array(np.flipud(ar_mask), dtype=str))
            tag_mask = eval(st)
        return tag_mask
        
    def load_encoded_tags(self, ep, tag_ch):
        """
        Return a numpy array containing
        bytes corresponding to the specified
        episode and channel.
        """
        tag_mode = self.tag_mode(ep)
        tag_mask = self.get_tag_mask(tag_ch, tag_mode)
        if tag_mode in [1, 2] :
            #digidata or itc mode
            #available for all formats
            ch = self.get_channel_for_tags(ep)
            raw = self.load_channel_data(ep, ch)
            return self.apply_op(raw, tag_mask, "mask")
        elif tag_mode == 3 :
            #cyber k mode
            #only available for DAC2 objects format
            
            #store bytes corresponding to the blocks
            #containing tags in a numpy array and reshape
            #it to have a set of tuples (time, value)
            ck_blocks = self.get_blocks_of_type(ep, 'RCyberTag')
            databytes = self.load_bytes(ck_blocks)
            raw = self.reshape_bytes(databytes, reshape=(4, 2), datatypes=('u', 'u'), order='<')
            
            #keep only items that are compatible
            #with the specified tag channel
            raw[:, 1] = self.apply_op(raw[:, 1], tag_mask, "mask")
            
            #computing numpy.diff is useful to know
            #how many times a value is maintained
            #and necessary to reconstruct the 
            #compressed signal ... 
            repeats = np.array(np.diff(raw[:, 0]), dtype=int)
            data = np.repeat(raw[:-1, 1], repeats, axis=0)
            
            # ... note that there is always
            #a transition at t=0 for synchronisation
            #purpose, consequently it is not necessary
            #to complete with zeros when the first
            #transition arrive ...
            
            return data
    
    def load_encoded_data(self, ep, ch):
        """
        Get encoded value of raw data from the elphy file.
        """
        tag_shift = self.tag_shift(ep)
        data = self.load_channel_data(ep, ch)
        if tag_shift :
            return  self.apply_op(data, tag_shift, "shift_right")
        else :
            return data
    
    def get_signal_data(self, ep, ch):
        """
        Return a numpy array containing all samples of a
        signal, acquired on an Elphy analog channel, formatted
        as a list of (time, value) tuples.
        """
        #get data from the file
        y_data = self.load_encoded_data(ep, ch)
        x_data = np.arange(0, len(y_data))
        
        #create a recarray
        data = np.recarray(len(y_data), dtype=[('x', b_float), ('y', b_float)])
        
        #put in the recarray the scaled data
        x_factors = self.x_scale_factors(ep, ch)
        y_factors = self.y_scale_factors(ep, ch)
        data['x'] = x_factors.scale(x_data)
        data['y'] = y_factors.scale(y_data)
        
        return data

    def get_tag_data(self, ep, tag_ch):
        """
        Return a numpy array containing all samples of a
        signal, acquired on an Elphy tag channel, formatted
        as a list of (time, value) tuples.
        """
        #get data from the file
        y_data = self.load_encoded_tags(ep, tag_ch)
        x_data = np.arange(0, len(y_data))
        
        #create a recarray
        data = np.recarray(len(y_data), dtype=[('x', b_float), ('y', b_int)])
        
        #put in the recarray the scaled data
        factors = self.x_tag_scale_factors(ep)
        data['x'] = factors.scale(x_data)
        data['y'] = y_data
        return data
    
class Acquis1Layout(ElphyLayout):
    """
    A subclass of :class:`ElphyLayout` to know
    how the 'ACQUIS1/GS/1991' format is organised.
    Extends :class:`ElphyLayout` to store the
    offset used to retrieve directly raw data :
    
    ``data_offset`` : an offset to jump directly
    to the raw data.
 
    """
    
    def __init__(self, fileobj, data_offset):
        super(Acquis1Layout, self).__init__(fileobj)
        self.data_offset = data_offset
        self.data_blocks = None
        
    def get_blocks_end(self):
        return self.data_offset
    
    def is_continuous(self):
        return self.header.continuous
    
    def get_episode_blocks(self):
        raise NotImplementedError()
    
    def set_info_block(self):
        i_blks = self.get_blocks_of_type('USER INFO')
        assert len(i_blks) < 2, 'too many info blocks'
        if len(i_blks) :
            self.info_block = i_blks[0]
    
    def set_data_blocks(self):
        data_blocks = list()
        size = self.header.n_samples * self.header.sample_size * self.header.n_channels
        for ep in range(0, self.header.n_episodes) :
            start = self.data_offset + ep * self.header.ep_size + self.header.preSeqI
            data_blocks.append(DummyDataBlock(self, 'Acquis1Data', start, size))
        self.data_blocks = data_blocks
    
    def get_data_blocks(self, ep):
        return [self.data_blocks[ep - 1]]
    
    @property
    def n_episodes(self):
        return self.header.n_episodes
    
    def n_channels(self, episode):
        return self.header.n_channels
    
    def n_tags(self, episode):
        return 0
    
    def tag_mode(self, ep):
        return 0
    
    def tag_shift(self, ep):
        return 0
    
    def get_channel_for_tags(self, ep):
        return None
    
    @property
    def no_analog_data(self):
        return True if (self.n_episodes == 0) else self.header.no_analog_data
    
    def sample_type(self, ep, ch):
        return self.header.tpData
    
    def sampling_period(self, ep, ch):
        return self.header.dX
    
    def n_samples(self, ep, ch):
        return self.header.n_samples
    
    def x_tag_scale_factors(self, ep):
        return ElphyScaleFactor(
            self.header.dX,
            self.header.X0
        )
    
    def x_scale_factors(self, ep, ch):
        return ElphyScaleFactor(
            self.header.dX,
            self.header.X0
        )
    
    def y_scale_factors(self, ep, ch):
        dY = self.header.dY_ar[ch - 1]
        Y0 = self.header.Y0_ar[ch - 1]
        # TODO: see why this kind of exception exists
        if dY is None or Y0 is None :
            raise Exception('bad Y-scale factors for episode %s channel %s' % (ep, ch))
        return ElphyScaleFactor(dY, Y0)
    
    def x_unit(self, ep, ch):
        return self.header.x_unit

    def y_unit(self, ep, ch):
        return self.header.y_units[ch - 1]
    
    @property
    def ep_size(self):
        return self.header.ep_size
    
    @property
    def file_duration(self):
        return self.header.dX * self.n_samples
    
    def get_tag(self, episode, tag_channel):
        return None
    
    def create_channel_mask(self, ep):
        return np.arange(1, self.header.n_channels + 1)

class DAC2GSLayout(ElphyLayout):
    """
    A subclass of :class:`ElphyLayout` to know
    how the 'DAC2 / GS / 2000' format is organised.
    Extends :class:`ElphyLayout` to store the
    offset used to retrieve directly raw data :
    
    ``data_offset`` : an offset to jump directly
    after the 'MAIN' block where 'DAC2SEQ' blocks
    start.
    
    ``main_block```: a shortcut to access 'MAIN' block.
    
    ``episode_blocks`` : a shortcut to access blocks
    corresponding to episodes.
    """
    
    def __init__(self, fileobj, data_offset):
        super(DAC2GSLayout, self).__init__(fileobj)
        self.data_offset = data_offset
        self.main_block = None
        self.episode_blocks = None
    
    def get_blocks_end(self):
        return self.file_size #data_offset
    
    def is_continuous(self):
        main_block = self.main_block
        return main_block.continuous if main_block else False
    
    def get_episode_blocks(self):
        raise NotImplementedError()
    
    def set_main_block(self):
        main_block = self.get_blocks_of_type('MAIN')
        self.main_block = main_block[0] if main_block else None
    
    def set_episode_blocks(self):
        ep_blocks = self.get_blocks_of_type('DAC2SEQ')
        self.episode_blocks = ep_blocks if ep_blocks else None
    
    def set_info_block(self):
        i_blks = self.get_blocks_of_type('USER INFO')
        assert len(i_blks) < 2, "too many info blocks"
        if len(i_blks) :
            self.info_block = i_blks[0]
    
    def set_data_blocks(self):
        data_blocks = list()
        identifier = 'DAC2GSData'
        size = self.main_block.n_samples * self.main_block.sample_size * self.main_block.n_channels
        if not self.is_continuous() :
            blocks = self.get_blocks_of_type('DAC2SEQ')
            for block in blocks :
                start = block.start + self.main_block.preSeqI
                data_blocks.append(DummyDataBlock(self, identifier, start, size))
        else :
            start = self.blocks[-1].end + 1 + self.main_block.preSeqI
            data_blocks.append(DummyDataBlock(self, identifier, start, size))
        self.data_blocks = data_blocks
    
    def get_data_blocks(self, ep):
        return [self.data_blocks[ep - 1]]
    
    def episode_block(self, ep):
        return self.main_block if self.is_continuous() else self.episode_blocks[ep - 1]
    
    def tag_mode(self, ep):
        return 1 if self.main_block.withTags else 0
     
    def tag_shift(self, ep):
        return self.main_block.tagShift
    
    def get_channel_for_tags(self, ep):
        return 1
    
    def sample_type(self, ep, ch):
        return self.main_block.tpData
    
    def sample_size(self, ep, ch):
        size = super(DAC2GSLayout, self).sample_size(ep, ch)
        assert size == 2, "sample size is always 2 bytes for DAC2/GS/2000 format"
        return size
    
    def sampling_period(self, ep, ch):
        block = self.episode_block(ep)
        return block.dX
    
    def x_tag_scale_factors(self, ep):
        block = self.episode_block(ep)
        return ElphyScaleFactor(
            block.dX,
            block.X0,
        )
    
    def x_scale_factors(self, ep, ch):
        block = self.episode_block(ep)
        return ElphyScaleFactor(
            block.dX,
            block.X0,
        )
    
    def y_scale_factors(self, ep, ch):
        block = self.episode_block(ep)
        return ElphyScaleFactor(
            block.dY_ar[ch - 1],
            block.Y0_ar[ch - 1]
        )
    
    def x_unit(self, ep, ch):
        block = self.episode_block(ep)
        return block.x_unit

    def y_unit(self, ep, ch):
        block = self.episode_block(ep)
        return block.y_units[ch - 1]
    
    def n_samples(self, ep, ch):
        return self.main_block.n_samples
    
    def ep_size(self, ep):
        return self.main_block.ep_size
    
    @property
    def n_episodes(self):
        return self.main_block.n_episodes
    
    def n_channels(self, episode):
        return self.main_block.n_channels
    
    def n_tags(self, episode):
        return 2 if self.main_block.withTags else 0
    
    @property
    def file_duration(self):
        return self.main_block.dX * self.n_samples
    
    def get_tag(self, episode, tag_channel):
        assert episode in range(1, self.n_episodes + 1)
        # there are none or 2 tag channels
        if self.tag_mode(episode) == 1 :
            assert tag_channel in range(1, 3), "DAC2/GS/2000 format support only 2 tag channels"
            block = self.episode_block(episode)
            t_stop = self.main_block.n_samples * block.dX
            return ElphyTag(self, episode, tag_channel, block.x_unit, 1.0 / block.dX, 0, t_stop)
        else :
            return None
    
    def n_tag_samples(self, ep, tag_channel):
        return self.main_block.n_samples
    
    def get_tag_data(self, episode, tag_channel):
        #memorise some useful properties
        block = self.episode_block(episode)
        sample_size = self.sample_size(episode, tag_channel)
        sample_symbol = self.sample_symbol(episode, tag_channel)
        
        #create a bit mask to define which
        #sample to keep from the file
        channel_mask = self.create_channel_mask(episode)
        bit_mask = self.create_bit_mask(channel_mask, 1)
        
        #get bytes from the file
        data_block = self.data_blocks[episode - 1]
        n_bytes = data_block.size
        self.file.seek(data_block.start)
        databytes = np.frombuffer(self.file.read(n_bytes), '<i1')
        
        #detect which bits keep to recompose the tag
        ep_mask = np.ones(n_bytes, dtype=int)
        np.putmask(ep_mask, ep_mask, bit_mask)
        to_keep = np.where(ep_mask > 0)[0]
        raw = databytes.take(to_keep)
        raw = raw.reshape([len(raw) / sample_size, sample_size])
        
        #create a recarray containing data
        dt = np.dtype(numpy_map[sample_symbol])
        dt.newbyteorder('<')
        tag_mask = 0b01 if (tag_channel == 1) else 0b10
        y_data = np.frombuffer(raw, dt) & tag_mask
        x_data = np.arange(0, len(y_data)) * block.dX + block.X0
        data = np.recarray(len(y_data), dtype=[('x', b_float), ('y', b_int)])
        data['x'] = x_data
        data['y'] = y_data
        
        return data
    
    def create_channel_mask(self, ep):
        return np.arange(1, self.main_block.n_channels + 1)



class DAC2Layout(ElphyLayout):
    """
    A subclass of :class:`ElphyLayout` to know
    how the Elphy format is organised.
    Whereas other formats storing raw data at the
    end of the file, 'DAC2 objects' format spreads
    them over multiple blocks :
    
    ``episode_blocks`` : a shortcut to access blocks
    corresponding to episodes.
    """
    
    def __init__(self, fileobj):
        super(DAC2Layout, self).__init__(fileobj)
        self.episode_blocks = None
        
    def get_blocks_end(self):
        return self.file_size
    
    def is_continuous(self):
        ep_blocks = [k for k in self.blocks if k.identifier.startswith('B_Ep')]
        if ep_blocks :
            ep_block = ep_blocks[0]
            ep_sub_block = ep_block.sub_blocks[0]
            return ep_sub_block.continuous
        else :
            return False
    
    def set_episode_blocks(self):
        self.episode_blocks = [k for k in self.blocks if k.identifier.startswith('B_Ep')]
    
    def set_info_block(self):
        #in fact the file info are contained into a single sub-block with an USR identifier
        i_blks = self.get_blocks_of_type('B_Finfo')
        assert len(i_blks) < 2, "too many info blocks"
        if len(i_blks) :
            i_blk = i_blks[0]
            sub_blocks = i_blk.sub_blocks
            if len(sub_blocks) :
                self.info_block = sub_blocks[0]
    
    def set_data_blocks(self):
        data_blocks = list()
        blocks = self.get_blocks_of_type('RDATA')
        for block in blocks :
            start = block.data_start
            size = block.end + 1 - start
            data_blocks.append(DummyDataBlock(self, 'RDATA', start, size))
        self.data_blocks = data_blocks
    
    def get_data_blocks(self, ep):
        return self.group_blocks_of_type(ep, 'RDATA')

    def group_blocks_of_type(self, ep, identifier):
        ep_blocks = list()
        blocks = [k for k in self.get_blocks_stored_in_episode(ep) if k.identifier == identifier]
        for block in blocks :
            start = block.data_start
            size = block.end + 1 - start
            ep_blocks.append(DummyDataBlock(self, identifier, start, size))
        return ep_blocks
    
    def get_blocks_stored_in_episode(self, ep):
        data_blocks = [k for k in self.blocks if k.identifier == 'RDATA']
        n_ep = self.n_episodes
        blk_1 = self.episode_block(ep)
        blk_2 = self.episode_block((ep + 1) % n_ep)
        i_1 = self.blocks.index(blk_1)
        i_2 = self.blocks.index(blk_2)
        if (blk_1 == blk_2) or (i_2 < i_1)  :
            return [k for k in data_blocks if self.blocks.index(k) > i_1]    
        else :
            return [k for k in data_blocks if self.blocks.index(k) in xrange(i_1, i_2)]
    
    def set_cyberk_blocks(self):
        ck_blocks = list()
        blocks = self.get_blocks_of_type('RCyberTag')
        for block in blocks :
            start = block.data_start
            size = block.end + 1 - start
            ck_blocks.append(DummyDataBlock(self, 'RCyberTag', start, size))
        self.ck_blocks = ck_blocks
    
    def episode_block(self, ep):
        return self.episode_blocks[ep - 1]
    
    @property
    def n_episodes(self):
        return len(self.episode_blocks)
    
    def analog_index(self, episode):
        """
        Return indices relative to channels
        used for analog signals.
        """
        block = self.episode_block(episode)
        tag_mode = block.ep_block.tag_mode
        an_index = np.where(np.array(block.ks_block.k_sampling) > 0)
        if tag_mode == 2 :
            an_index = an_index[:-1]
        return an_index
    
    def n_channels(self, episode):
        """
        Return the number of channels used
        for analog signals but also events.
        
        NB : in Elphy this 2 kinds of channels
        are not differenciated.
        """
        block = self.episode_block(episode)
        tag_mode = block.ep_block.tag_mode
        n_channels = len(block.ks_block.k_sampling)
        return n_channels if tag_mode != 2 else n_channels - 1
    
    def n_tags(self, episode):
        block = self.episode_block(episode)
        tag_mode = block.ep_block.tag_mode
        tag_map = {0:0, 1:2, 2:16, 3:16}
        return tag_map.get(tag_mode, 0)
    
    def n_events(self, episode):
        """
        Return the number of channels
        dedicated to events.
        """
        block = self.episode_block(episode)
        return block.ks_block.k_sampling.count(0)
    
    def n_spiketrains(self, episode):
        spk_blocks = [k for k in self.blocks if k.identifier == 'RSPK']
        return spk_blocks[0].n_evt_channels if spk_blocks else 0
    
    def sub_sampling(self, ep, ch):
        """
        Return the sub-sampling factor for
        the specified episode and channel.
        """
        block = self.episode_block(ep)
        return block.ks_block.k_sampling[ch - 1] if block.ks_block else 1
    
    def aggregate_size(self, block, ep):
        ag_count = self.aggregate_sample_count(block)
        ag_size = 0
        for ch in range(1, ag_count + 1) :
            if (block.ks_block.k_sampling[ch - 1] != 0) :
                ag_size += self.sample_size(ep, ch)
        return ag_size
    
    def n_samples(self, ep, ch):
        block = self.episode_block(ep)
        if not block.ep_block.continuous :
            return block.ep_block.nbpt / self.sub_sampling(ep, ch)
        else :
            # for continuous case there isn't any place
            # in the file that contains the number of 
            # samples unlike the episode case ...
            data_blocks = self.get_data_blocks(ep)
            total_size = np.sum([k.size for k in data_blocks])
            
            # count the number of samples in an
            # aggregate and compute its size in order
            # to determine the size of an aggregate
            ag_count = self.aggregate_sample_count(block)
            ag_size = self.aggregate_size(block, ep)
            n_ag = total_size / ag_size
            
            # the number of samples is equal
            # to the number of aggregates ...
            n_samples = n_ag 
            n_chunks = total_size % ag_size
            
            # ... but not when there exists
            # a incomplete aggregate at the
            # end of the file, consequently
            # the preeceeding computed number
            # of samples must be incremented
            # by one only if the channel map
            # to a sample in the last aggregate
            
            # ... maybe this last part should be
            # deleted because the n_chunks is always
            # null in continuous mode
            if n_chunks :
                last_ag_size = total_size - n_ag * ag_count
                size = 0
                for i in range(0, ch) :
                    size += self.sample_size(ep, i + 1)
                if size <= last_ag_size :
                    n_samples += 1
            
            return n_samples
    
    def sample_type(self, ep, ch):
        block = self.episode_block(ep)
        return block.kt_block.k_types[ch - 1] if block.kt_block else block.ep_block.tpData
    
    def sampling_period(self, ep, ch):
        block = self.episode_block(ep)
        return block.ep_block.dX * self.sub_sampling(ep, ch)
    
    def x_tag_scale_factors(self, ep):
        block = self.episode_block(ep)
        return ElphyScaleFactor(
            block.ep_block.dX,
            block.ep_block.X0
        )
    
    def x_scale_factors(self, ep, ch):
        block = self.episode_block(ep)
        return ElphyScaleFactor(
            block.ep_block.dX * block.ks_block.k_sampling[ch - 1],
            block.ep_block.X0,
        )
    
    def y_scale_factors(self, ep, ch):
        block = self.episode_block(ep)
        return ElphyScaleFactor(
            block.ch_block.dY_ar[ch - 1],
            block.ch_block.Y0_ar[ch - 1]
        )
    
    def x_unit(self, ep, ch):
        block = self.episode_block(ep)
        return block.ep_block.x_unit
    
    def y_unit(self, ep, ch):
        block = self.episode_block(ep)
        return block.ch_block.y_units[ch - 1]
    
    def tag_mode(self, ep):
        block = self.episode_block(ep)
        return block.ep_block.tag_mode
    
    def tag_shift(self, ep):
        block = self.episode_block(ep)
        return block.ep_block.tag_shift
    
    def get_channel_for_tags(self, ep):
        block = self.episode_block(ep)
        tag_mode = self.tag_mode(ep)
        if tag_mode == 1 :
            ks = np.array(block.ks_block.k_sampling)
            mins = np.where(ks == ks.min())[0] + 1
            return mins[0]
        elif tag_mode == 2 :
            return block.ep_block.n_channels
        else :
            return None
    
    def aggregate_sample_count(self, block):
        """
        Return the number of sample in an aggregate.
        """
        
        # compute the least common multiple
        # for channels having block.ks_block.k_sampling[ch] > 0
        lcm0 = 1
        for i in range(0, block.ep_block.n_channels) :
            if block.ks_block.k_sampling[i] > 0 :
                lcm0 = least_common_multiple(lcm0, block.ks_block.k_sampling[i])
        
        # sum quotients lcm / KSampling
        count = 0
        for i in range(0, block.ep_block.n_channels) :
            if block.ks_block.k_sampling[i] > 0 :
                count += lcm0 / block.ks_block.k_sampling[i]
        
        return count
    
    def create_channel_mask(self, ep):
        """
        Return the minimal pattern of channel numbers
        representing the succession of channels in the
        multiplexed data. It is useful to do the mapping
        between a sample stored in the file and its relative
        channel.
        
        NB : This function has been converted from the
        'TseqBlock.BuildMask' method of the file 'ElphyFormat.pas'
        stored in Elphy source code.
        """
        
        block = self.episode_block(ep)
        ag_count = self.aggregate_sample_count(block)
        mask_ar = np.zeros(ag_count, dtype='i')
        ag_size = 0
        i = 0
        k = 0
        while k < ag_count :
            for j in range(0, block.ep_block.n_channels) :
                if (block.ks_block.k_sampling[j] != 0) and (i % block.ks_block.k_sampling[j] == 0) :
                    mask_ar[k] = j + 1
                    ag_size += self.sample_size(ep, j + 1)
                    k += 1
                    if k >= ag_count :
                        break
            i += 1
        return mask_ar
    
    def get_signal(self, episode, channel):
        block = self.episode_block(episode)
        k_sampling = np.array(block.ks_block.k_sampling)
        evt_channels = np.where(k_sampling == 0)[0]
        if not channel in evt_channels :
            return super(DAC2Layout, self).get_signal(episode, channel)
        else :
            k_sampling[channel - 1] = -1
            return self.get_event(episode, channel, k_sampling)
    
    def get_tag(self, episode, tag_channel):
        """
        Return a :class:`ElphyTag` which is a
        descriptor of the specified event channel.
        """
        assert episode in range(1, self.n_episodes + 1)
        
        # there are none, 2 or 16 tag 
        # channels depending on tag_mode
        tag_mode = self.tag_mode(episode)
        if tag_mode :
            block = self.episode_block(episode)
            x_unit = block.ep_block.x_unit
            
            # verify the validity of the tag channel
            if tag_mode == 1 :
                assert tag_channel in range(1, 3), "Elphy format support only 2 tag channels for tag_mode == 1"
            elif tag_mode == 2 :
                assert tag_channel in range(1, 17), "Elphy format support only 16 tag channels for tag_mode == 2"
            elif tag_mode == 3 :
                assert tag_channel in range(1, 17), "Elphy format support only 16 tag channels for tag_mode == 3"
            
            smp_period = block.ep_block.dX
            smp_freq = 1.0 / smp_period
            if tag_mode != 3 : 
                ch = self.get_channel_for_tags(episode)
                n_samples = self.n_samples(episode, ch)
                t_stop = (n_samples - 1) * smp_freq
            else :  
                # get the max of n_samples multiplied by the sampling
                # period done on every analog channels in order to avoid
                # the selection of a channel without concrete signals
                t_max = list()
                for ch in self.analog_index(episode) :
                    n_samples = self.n_samples(episode, ch)
                    factors = self.x_scale_factors(episode, ch)
                    chtime = n_samples * factors.delta
                    t_max.append(chtime)
                time_max = max(t_max)
                
                # as (n_samples_tag - 1) * dX_tag
                # and time_max = n_sample_tag * dX_tag
                # it comes the following duration
                t_stop = time_max - smp_period
                
            return ElphyTag(self, episode, tag_channel, x_unit, smp_freq, 0, t_stop)
        else :
            return None
    
    def get_event(self, ep, ch, marked_ks):
        """
        Return a :class:`ElphyEvent` which is a
        descriptor of the specified event channel.
        """
        assert ep in range(1, self.n_episodes + 1)
        assert ch in range(1, self.n_channels + 1)
        
        # find the event channel number
        evt_channel = np.where(marked_ks == -1)[0][0]
        assert evt_channel in range(1, self.n_events(ep) + 1)
        
        block = self.episode_block(ep)
        ep_blocks = self.get_blocks_stored_in_episode(ep)
        evt_blocks = [k for k in ep_blocks if k.identifier == 'REVT']
        n_events = np.sum([k.n_events[evt_channel - 1] for k in evt_blocks], dtype=int)
        x_unit = block.ep_block.x_unit
        
        return ElphyEvent(self, ep, evt_channel, x_unit, n_events, ch_number=ch)
    
    def load_encoded_events(self, episode, evt_channel, identifier):
        """
        Return times stored as a 4-bytes integer
        in the specified event channel.
        """
        data_blocks = self.group_blocks_of_type(episode, identifier)
        ep_blocks = self.get_blocks_stored_in_episode(episode)
        evt_blocks = [k for k in ep_blocks if k.identifier == identifier]
        
        #compute events on each channel
        n_events = np.sum([k.n_events for k in evt_blocks], dtype=int, axis=0)
        pre_events = np.sum(n_events[0:evt_channel - 1], dtype=int)
        start = pre_events
        end = start + n_events[evt_channel - 1]
        expected_size = 4 * np.sum(n_events, dtype=int)
        return self.load_bytes(data_blocks, dtype='<i4', start=start, end=end, expected_size=expected_size)
    
    def load_encoded_spikes(self, episode, evt_channel, identifier):
        """
        Return times stored as a 4-bytes integer
        in the specified spike channel.
        NB: it is meant for Blackrock-type, having an additional byte for each event time as spike sorting label.
            These additiona bytes are appended trailing the times.
        """
        # to load the requested spikes for the specified episode and event channel:
        # get all the elphy blocks having as identifier 'RSPK' (or whatever)
        all_rspk_blocks = [k for k in self.blocks if k.identifier == identifier]
        rspk_block = all_rspk_blocks[episode-1]
        # RDATA(h?dI) REVT(NbVeV:I, NbEv:256I ... spike data are 4byte integers
        rspk_header = 4*( rspk_block.size - rspk_block.data_size-2 + len(rspk_block.n_events))
        pre_events = np.sum(rspk_block.n_events[0:evt_channel-1], dtype=int, axis=0)
        # the real start is after header, preceeding events (which are 4byte) and preceeding labels (1byte)
        start = rspk_header + (4*pre_events) + pre_events
        end = start + 4*rspk_block.n_events[evt_channel-1]
        raw = self.load_bytes( [rspk_block], dtype='<i1', start=start, end=end, expected_size=rspk_block.size )
        # re-encoding after reading byte by byte
        res = np.frombuffer(raw[0:(4*rspk_block.n_events[evt_channel-1])], dtype='<i4')
        res.sort() # sometimes timings are not sorted
        #print "load_encoded_data() - spikes:",res
        return res

    def get_episode_name( self, episode ):
        episode_name = "episode %s" % episode
        names = [k for k in self.blocks if k.identifier == 'COM']
        if len(names) > 0 :
            name = names[episode-1]
            start = name.size+1 - name.data_size+1
            end = name.end - name.start+1
            chars = self.load_bytes([name], dtype='uint8', start=start, end=end, expected_size=name.size ).tolist()
            #print "chars[%s:%s]: %s" % (start,end,chars)
            episode_name = ''.join([chr(k) for k in chars])
        return episode_name
 
    def get_event_data(self, episode, evt_channel):
        """
        Return times contained in the specified event channel.
        This function is triggered when the 'times' property of 
        an :class:`ElphyEvent` descriptor instance is accessed.  
        """
        times = self.load_encoded_events(episode, evt_channel, "REVT")
        block = self.episode_block(episode)
        return times * block.ep_block.dX / len(block.ks_block.k_sampling)
    
    def get_spiketrain(self, episode, electrode_id):
        """
        Return a :class:`Spike` which is a
        descriptor of the specified spike channel.
        """
        assert episode in range(1, self.n_episodes + 1)
        assert electrode_id in range(1, self.n_spiketrains(episode) + 1)
        # get some properties stored in the episode sub-block
        block = self.episode_block(episode)
        x_unit = block.ep_block.x_unit
        x_unit_wf = getattr(block.ep_block, 'x_unit_wf', None)
        y_unit_wf = getattr(block.ep_block, 'y_unit_wf', None)
        # number of spikes in the entire episode
        spk_blocks = [k for k in self.blocks if k.identifier == 'RSPK']
        n_events = np.sum([k.n_events[electrode_id - 1] for k in spk_blocks], dtype=int)
        # number of samples in a waveform
        wf_sampling_frequency = 1.0 / block.ep_block.dX
        wf_blocks = [k for k in self.blocks if k.identifier == 'RspkWave']
        if wf_blocks :
            wf_samples = wf_blocks[0].wavelength
            t_start = wf_blocks[0].pre_trigger * block.ep_block.dX
        else:
            wf_samples = 0
            t_start = 0
        return ElphySpikeTrain(self, episode, electrode_id, x_unit, n_events, wf_sampling_frequency, wf_samples, x_unit_wf, y_unit_wf, t_start)
    
    def get_spiketrain_data(self, episode, electrode_id):
        """
        Return times contained in the specified spike channel.
        This function is triggered when the 'times' property of 
        an :class:`Spike` descriptor instance is accessed.
        
        NB : The 'RSPK' block is not actually identical to the 'EVT' one,
        because all units relative to a time are stored directly after all
        event times, 1 byte for each. This function doesn't return these
        units. But, they could be retrieved from the 'RspkWave' block with
        the 'get_waveform_data function'
        """
        block = self.episode_block(episode)
        times = self.load_encoded_spikes(episode, electrode_id, "RSPK")
        return times * block.ep_block.dX
    
    def load_encoded_waveforms(self, episode, electrode_id):
        """
        Return times on which waveforms are defined
        and a numpy recarray containing all the data
        stored in the RspkWave block.
        """
        # load data corresponding to the RspkWave block
        identifier = "RspkWave"
        data_blocks = self.group_blocks_of_type(episode, identifier)
        databytes = self.load_bytes(data_blocks)
        
        # select only data corresponding
        # to the specified spk_channel
        ep_blocks = self.get_blocks_stored_in_episode(episode)
        wf_blocks = [k for k in ep_blocks if k.identifier == identifier]
        wf_samples = wf_blocks[0].wavelength
        events = np.sum([k.n_spikes for k in wf_blocks], dtype=int, axis=0)
        n_events = events[electrode_id - 1]
        pre_events = np.sum(events[0:electrode_id - 1], dtype=int)
        start = pre_events
        end = start + n_events
        
        # data must be reshaped before
        dtype = [
            # the time of the spike arrival
            ('elphy_time', 'u4', (1,)),
            ('device_time', 'u4', (1,)),
            
            # the identifier of the electrode
            # would also be the 'trodalness'
            # but this tetrode devices are not
            # implemented in Elphy
            ('channel_id', 'u2', (1,)),
            
            # the 'category' of the waveform
            ('unit_id', 'u1', (1,)),
            
            #do not used
            ('dummy', 'u1', (13,)),
            
            # samples of the waveform
            ('waveform', 'i2', (wf_samples,))
        ]
        x_start = wf_blocks[0].pre_trigger
        x_stop = wf_samples - x_start
        return np.arange(-x_start, x_stop), np.frombuffer(databytes, dtype=dtype)[start:end]
    
    def get_waveform_data(self, episode, electrode_id):
        """
        Return waveforms corresponding to the specified
        spike channel. This function is triggered when the
        ``waveforms`` property of an :class:`Spike` descriptor
        instance is accessed.
        """
        block = self.episode_block(episode)
        times, databytes = self.load_encoded_waveforms(episode, electrode_id)
        n_events, = databytes.shape
        wf_samples = databytes['waveform'].shape[1]
        dtype = [
            ('time', float),
            ('electrode_id', int),
            ('unit_id', int),
            ('waveform', float, (wf_samples, 2))
        ]
        data = np.empty(n_events, dtype=dtype)
        data['electrode_id'] = databytes['channel_id'][:, 0]
        data['unit_id'] = databytes['unit_id'][:, 0]
        data['time'] = databytes['elphy_time'][:, 0] * block.ep_block.dX
        data['waveform'][:, :, 0] = times * block.ep_block.dX
        data['waveform'][:, :, 1] = databytes['waveform'] * block.ep_block.dY_wf + block.ep_block.Y0_wf
        return data
    
    def get_rspk_data(self, spk_channel):
        """
        Return times stored as a 4-bytes integer
        in the specified event channel.
        """
        evt_blocks = self.get_blocks_of_type('RSPK')
        #compute events on each channel
        n_events = np.sum([k.n_events for k in evt_blocks], dtype=int, axis=0)
        pre_events = np.sum(n_events[0:spk_channel], dtype=int) # sum of array values up to spk_channel-1!!!!
        start = pre_events + (7 + len(n_events))# rspk header
        end = start + n_events[spk_channel]
        expected_size = 4 * np.sum(n_events, dtype=int) # constant
        return self.load_bytes(evt_blocks, dtype='<i4', start=start, end=end, expected_size=expected_size)


# ---------------------------------------------------------
# factories.py

class LayoutFactory(object):
    """
    Generate base elements composing the layout of a file.
    """
    
    def __init__(self, elphy_file):
        self.elphy_file = elphy_file
        self.pattern = "\d{4}(\d+|\D)\D"
        self.block_subclasses = dict()
    
    @property
    def file(self):
        return self.elphy_file.file
    
    def create_layout(self):
        """
        Return the actual :class:`ElphyLayout` subclass
        instance used in an :class:`ElphyFile` object.
        """
        raise Exception('must be overloaded in a subclass')
    
    def create_header(self, layout):
        """
        Return the actual :class:`Header` instance used
        in an :class:`ElphyLayout` subclass object.
        """
        raise Exception('must be overloaded in a subclass')

    def create_block(self, layout):
        """
        Return a :class:`Block` instance composing
        the :class:`ElphyLayout` subclass instance.
        """
        raise Exception('must be overloaded in a subclass')
    
    def create_sub_block(self, block, sub_offset):
        """
        Return a set of sub-blocks stored
        in DAC2 objects format files.
        """
        self.file.seek(sub_offset)
        sub_ident_size = read_from_char(self.file, 'B')
        sub_identifier, = struct.unpack('<%ss' % sub_ident_size, self.file.read(sub_ident_size))
        sub_data_size = read_from_char(self.file, 'H')
        sub_data_offset = sub_offset + sub_ident_size + 3
        size_format = "H"
        if sub_data_size == 0xFFFF :
            _ch = 'l'
            sub_data_size = read_from_char(self.file, _ch)
            size_format += "+%s" % (_ch)
            sub_data_offset += 4
        sub_size = len(sub_identifier) + 1 + type_dict[size_format] + sub_data_size
        if sub_identifier == 'Ep' :
            block_type = DAC2EpSubBlock
        elif sub_identifier == 'Adc' :
            block_type = DAC2AdcSubBlock
        elif sub_identifier == 'Ksamp' :
            block_type = DAC2KSampSubBlock
        elif sub_identifier == 'Ktype' :
            block_type = DAC2KTypeSubBlock
        elif sub_identifier == 'USR' :
            block_type = self.select_file_info_subclass()
        else :
            block_type = ElphyBlock
        block = block_type(block.layout, sub_identifier, sub_offset, sub_size, size_format=size_format, parent_block=block)
        self.file.seek(self.file.tell() + sub_data_size)
        return block
    
    def create_episode(self, block):
        raise Exception('must be overloaded in a subclass')
    
    def create_channel(self, block):
        raise Exception('must be overloaded in a subclass')
    
    def is_multistim(self, path):
        """
        Return a boolean telling if the 
        specified file is a multistim one.
        """
        match = re.search(self.pattern, path)
        return hasattr(match, 'end') and path[match.end() - 1] in ['m', 'M']
    
    def select_file_info_subclass(self):
        """
        Detect the type of a file from its nomenclature
        and return its relative :class:`ClassicFileInfo` or 
        :class:`MultistimFileInfo` class. Useful to transparently
        access to user file info stored in an Elphy file.
        """
        if not self.is_multistim(self.file.name) :
            return ClassicFileInfo 
        else :
            return MultistimFileInfo
    
    def select_block_subclass(self, identifier):
        return self.block_subclasses.get(identifier, ElphyBlock)
        
class Acquis1Factory(LayoutFactory):
    """
    Subclass of :class:`LayoutFactory` useful to
    generate base elements composing the layout
    of Acquis1 file format.
    """
    
    def __init__(self, elphy_file):
        super(Acquis1Factory, self).__init__(elphy_file)
        self.file.seek(16)
        self.data_offset = read_from_char(self.file, 'h')
        self.file.seek(0)
        
        # the set of interesting blocks useful
        # to retrieve data stored in a file
        self.block_subclasses = {
            "USER INFO" : self.select_file_info_subclass()
        }
    
    def create_layout(self):
        return Acquis1Layout(self.elphy_file, self.data_offset)
    
    def create_header(self, layout):
        return Acquis1Header(layout)
    
    def create_block(self, layout, offset):
        self.file.seek(offset)
        ident_size, identifier = struct.unpack('<B15s', self.file.read(16))
        identifier = identifier[0:ident_size]
        size = read_from_char(self.file, 'h')
        block_type = self.select_block_subclass(identifier)
        block = block_type(layout, identifier, offset, size, fixed_length=15, size_format='h')
        self.file.seek(0)
        return block

class DAC2GSFactory(LayoutFactory):
    """
    Subclass of :class:`LayoutFactory` useful to
    generate base elements composing the layout
    of DAC2/GS/2000 file format.
    """
    
    def __init__(self, elphy_file):
        super(DAC2GSFactory, self).__init__(elphy_file)
        self.file.seek(16)
        self.data_offset = read_from_char(self.file, 'i')
        self.file.seek(0)
        
        # the set of interesting blocks useful
        # to retrieve data stored in a file
        self.block_subclasses = {
            "USER INFO" : self.select_file_info_subclass(),
            "DAC2SEQ" : DAC2GSEpisodeBlock,
            'MAIN' : DAC2GSMainBlock,
        }
    
    def create_layout(self):
        return DAC2GSLayout(self.elphy_file, self.data_offset)
    
    def create_header(self, layout):
        return DAC2GSHeader(layout)
    
    def create_block(self, layout, offset):
        self.file.seek(offset)
        ident_size, identifier = struct.unpack('<B15s', self.file.read(16))
        # block title size is 7 or 15 bytes
        # 7 is for sequence blocs
        if identifier.startswith('DAC2SEQ') :
            self.file.seek(self.file.tell() - 8)
            length = 7
        else : 
            length = 15
        identifier = identifier[0:ident_size]
        size = read_from_char(self.file, 'i')
        block_type = self.select_block_subclass(identifier)
        block = block_type(layout, identifier, offset, size, fixed_length=length, size_format='i') 
        self.file.seek(0)
        return block



class DAC2Factory(LayoutFactory):
    """
    Subclass of :class:`LayoutFactory` useful to
    generate base elements composing the layout
    of DAC2 objects file format.
    """
    
    def __init__(self, elphy_file):
        super(DAC2Factory, self).__init__(elphy_file)
        
        # the set of interesting blocks useful
        # to retrieve data stored in a file
        self.block_subclasses = {
            "B_Ep" : DAC2EpisodeBlock,
            "RDATA" : DAC2RDataBlock,
            "RCyberTag" : DAC2CyberTagBlock,
            "REVT" : DAC2EventBlock,
            "RSPK" : DAC2SpikeBlock,
            "RspkWave" : DAC2WaveFormBlock
        }
    
    def create_layout(self):
        return DAC2Layout(self.elphy_file)
    
    def create_header(self, layout):
        return DAC2Header(layout)
    
    def create_block(self, layout, offset):
        self.file.seek(offset)
        size = read_from_char(self.file, 'l')
        ident_size = read_from_char(self.file, 'B')
        identifier, = struct.unpack('<%ss' % ident_size, self.file.read(ident_size))
        block_type = self.select_block_subclass(identifier)
        block = block_type(layout, identifier, offset, size, size_format='l')
        self.file.seek(0)
        return block




#caching all available layout factories

factories = {
    "ACQUIS1/GS/1991" : Acquis1Factory,
    "DAC2/GS/2000" : DAC2GSFactory,
    "DAC2 objects" : DAC2Factory
}


# --------------------------------------------------------
# ELPHY FILE


"""
Classes useful to retrieve data from the
three major Elphy formats, i.e : Acquis1, DAC2/GS/2000, DAC2 objects.

The :class:`ElphyFile` class is useful to access raw data and user info
that stores protocol metadata. Internally, It uses a subclass :class:`ElphyLayout`
to handle each kind of file format : :class:`Acquis1Layout`, :class:`DAC2GSLayout`
and :class:`DAC2Layout`.

These layouts decompose the file structure into several blocks of data, inheriting
from the :class:`BaseBlock`, corresponding for example to the header of the file,
the user info, the raw data, the episode or channel properties. Each subclass of
:class:`BaseBlock` map to a file chunk and is responsible to store metadata contained
in this chunk. These metadata could be also useful to reconstruct raw data.

Consequently, when an :class:`ElphyLayout` layout is requested by its relative
:class:`ElphyFile`, It iterates through :class:`BaseBlock` objects to retrieve
asked data.

NB : The reader is not able to read Acquis1 and DAC2/GS/2000 event channels.
"""


class ElphyFile(object):
    """
    A convenient class useful to read Elphy files.
    It acts like a file reader that wraps up a python
    file opened in 'rb' mode in order to retrieve
    directly from an Elphy file raw data and metadata
    relative to protocols.
    
    ``path`` : the path of the elphy file.
    
    ``file`` : the python file object that iterates
    through the elphy file.
    
    ``file_size`` : the size of the elphy file on the
    hard disk drive.
    
    ``nomenclature`` : the label that identifies the
    kind of elphy format, i.e. 'Acquis1', 'DAC2/GS/2000',
    'DAC2 objects'.
    
    ``factory`` : the :class:`LayoutFactory` object which
    generates the base component of the elphy file layout.
    
    ``layout`` : the :class:`ElphyLayout` object which
    decomposes the file structure into several blocks of
    data (:class:`BaseBlock` objects). The :class:`ElphyFile`
    object do requests to this layout which iterates through
    this blocks before returning asked data.
     
    ``protocol`` : the acquisition protocol which has generated
    the file.
    
    ``version`` : the variant of the acquisition protocol.
    
    NB : An elphy file could store several kind of data :
    
    (1) 'User defined' metadata which are stored in a block
    called 'USER INFO' ('Acquis1' and 'DAC2/GS/2000') or 'USR'
    ('DAC2 objects') of the ``layout``. They could be used for
    example to describe stimulation parameters.
    
    (2) Raw data acquired on separate analog channels. Data
    coming from each channel are multiplexed in blocks dedicated
    to raw data storage : 
    
        - For Acquis1 format, raw data are stored directly
        after the file header.
        
        - For DAC2/GS/2000, in continuous mode they are stored
        after all blocks composing the file else they are stored
        in a 'DAC2SEQ' block.
        
        - For 'DAC2 objects' they are stored in 'RDATA' blocks.
        In continuous mode raw data could be spread other multiple
        'RDATA' blocks. Whereas in episode mode there is a single
        'RDATA' block for each episode.
    
    These raw data are placed under the 'channels' node of a
    TDataFile object listed in Elphy's "Inspect" tool.
    
    (3) ElphyEvents dedicated to threshold detection in analog
    channels. ElphyEvents are only available for 'DAC2 objects'
    format. For 'Acquis1' and 'DAC2/GS/2000' these events are
    in fact stored in another kind of file format called
    'event' format with the '.evt' extension which is opened
    by Elphy as same time as the '.dat' file. This 'event'
    format is not yet implemented because it seems that it
    was not really used. 
    
    These events are also placed under the 'channels' node
    of a TDataFile object in Elphy's "Inspect" tool.
    
    (4) ElphyTags that appeared after 'DAC2/GS/2000' release. They
    are also present in 'DAC2 objects' format. Each, tag occupies
    a channel called 'tag' channel. Their encoding depends on the
    kind of acquisition card :
    
        - For 'digidata' cards (``tag_mode``=1) and if tags are acquired,
        they are directly encoded in 2 (digidata 1322) or 4 (digidata 1200)
        significant bits of 16-bits samples coming from an analog channel.
        In all cases they are only 2 bits encoding the tag channels. The
        sample value could be encoded on 16, 14 or 12 bits and retrieved by
        applying a shift equal to ``tag_shift`` to the right. 
    
        - For ITC cards (``tag_mode``=2), tags are transmitted by a channel
        fully dedicated to 'tag channels' providing 16-bits samples. In this
        case, each bit corresponds to a 'tag channel'.
        
        - For Blackrock/Cyberkinetics devices (``tag_mode``=3), tags are also
        transmitted by a channel fully dedicated to tags, but the difference is
        that only transitions are stored in 'RCyberTag' blocks. This case in only
        available in 'DAC2 objects' format.
    
    These tags are placed under the 'Vtags' node of a TDataFile
    object in Elphy's "Inspect" tool.
    
    (5) Spiketrains coming from an electrode of a Blackrock/Cyberkinetics
    multi-electrode device. These data are only available in 'DAC2 objects'
    format.
    
    These spiketrains are placed under the 'Vspk' node of a TDataFile
    object in Elphy's "Inspect" tool.
    
    (6) Waveforms relative to each time of a spiketrain. These data are only
    available in 'DAC2 objects' format. These waveforms are placed under the
    'Wspk' node of a TDataFile object in Elphy's "Inspect" tool.
    """
    
    def __init__(self, file_path) :
        self.path = file_path
        self.folder, self.filename = path.split(self.path)
        self.file = None
        self.file_size = None
        self.nomenclature = None
        self.factory = None
        self.layout = None
        # writing support
        self.header_size = None


    def __del__(self):
        """
        Trigger closing of the file.
        """
        self.close()
        # super(ElphyFile, self).__del__()


    def open(self):
        """
        Setup the internal structure.
        
        NB : Call this function before 
        extracting data from a file.
        """
        if self.file :
            self.file.close()
        try :
            self.file = open(self.path, 'rb')
        except Exception as e:
            raise Exception("python couldn't open file %s : %s" % (self.path, e))
        self.file_size = path.getsize(self.file.name)
        self.creation_date = datetime.fromtimestamp(path.getctime(self.file.name))
        self.modification_date = datetime.fromtimestamp(path.getmtime(self.file.name))
        self.nomenclature = self.get_nomenclature()
        self.factory = self.get_factory()
        self.layout = self.create_layout()


    def close(self):
        """
        Close the file.
        """
        if self.file :
            self.file.close()


    def get_nomenclature(self):
        """
        Return the title of the file header
        giving the actual file format. This
        title is encoded as a pascal string
        containing 15 characters and stored
        as 16 bytes of binary data.  
        """
        self.file.seek(0)
        length, title = struct.unpack('<B15s', self.file.read(16))
        self.file.seek(0)
        title = title[0:length]
        if not title in factories :
            title = "format is not implemented ('%s' not in %s)" % (title, str(factories.keys()))
        return title
    
    
    def set_nomenclature(self):
        """
        As in get_nomenclature, but set the title of the file header
        in the file, encoded as a pascal string containing 
        15 characters and stored as 16 bytes of binary data.  
        """
        self.file.seek(0)
        title = 'DAC2 objects'
        st = struct.Struct( '<B15sH' )
        header_rec = [len(title), title, 18] # constant header
        header_chr = st.pack( *header_rec )
        self.header_size = len( header_chr )
        self.file.write( header_chr )


    def get_factory(self):
        """
        Return a subclass of :class:`LayoutFactory`
        useful to build the file layout depending
        on header title.
        """
        return factories[self.nomenclature](self)


      
    def write(self, data):
        """
        Assume the blocks are already filled.
        It is able to write several types of block: B_Ep, RDATA, ...
        and subBlock: Adc, Ksamp, Ktype, dataRecord, ...
        In the following shape:
        B_Ep
            |_ Ep
            |_ Adc
            |_ Adc
            |_ ...
            |_ Ktype
        RDATA
            |_ dataRecord+data
        """
        # close if open and reopen for writing
        if self.file :
            self.file.close()
        try :
            self.file = open(self.path, 'wb')
        except Exception as e:
            raise Exception("python couldn't open file %s : %s" % (self.path, e))
        self.file_size = 0
        self.creation_date = datetime.now()
        self.modification_date = datetime.now()
        self.set_nomenclature()
        # then call ElphyFile writing routines to write the serialized string
        self.file.write( data ) # actual writing
        # close file
        self.close()


      
    def create_layout(self):
        """
        Build the :class:`Layout` object corresponding
        to the file format and configure properties of
        itself and then its blocks and sub-blocks.
        
        NB : this function must be called before all kind
        of requests on the file because it is used also to setup
        the internal properties of the :class:`ElphyLayout`
        object or some :class:`BaseBlock` objects. Consequently,
        executing some function corresponding to a request on
        the file has many chances to lead to bad results. 
        """
        # create the layout
        layout = self.factory.create_layout()
        
        # create the header block and
        # add it to the list of blocks
        header = self.factory.create_header(layout)
        layout.add_block(header)
        
        # set the position of the cursor
        # in order to be after the header
        # block and then compute its last
        # valid position to know when stop
        # the iteration through the file
        offset = header.size
        offset_stop = layout.get_blocks_end()
        
        # in continuous mode DAC2/GS/2000 raw data are not stored
        # into several DAC2SEQ blocks, they are stored after all 
        # available blocks, that's why it is necessary to limit the
        # loop to data_offset when it is a DAC2/GS/2000 format
        is_continuous = False
        detect_continuous = False
        detect_main = False
        while (offset < offset_stop) and not (is_continuous  and (offset >= layout.data_offset)) :
            block = self.factory.create_block(layout, offset)
            
            # create the sub blocks if it is DAC2 objects format
            # this is only done for B_Ep and B_Finfo blocks for
            # DAC2 objects format, maybe it could be useful to
            # spread this to other block types.
            #if isinstance(header, DAC2Header) and (block.identifier in ['B_Ep']) :
            if isinstance(header, DAC2Header) and (block.identifier in ['B_Ep', 'B_Finfo']) :
                sub_offset = block.data_offset
                while sub_offset < block.start + block.size :
                    sub_block = self.factory.create_sub_block(block, sub_offset)
                    block.add_sub_block(sub_block)
                    sub_offset += sub_block.size
                # set up some properties of some DAC2Layout sub-blocks
                if isinstance(sub_block, (DAC2EpSubBlock, DAC2AdcSubBlock, DAC2KSampSubBlock, DAC2KTypeSubBlock)) :
                    block.set_episode_block()
                    block.set_channel_block()
                    block.set_sub_sampling_block()
                    block.set_sample_size_block()

            # SpikeTrain
            #if isinstance(header, DAC2Header) and (block.identifier in ['RSPK']) :
                #print "\nElphyFile.create_layout() - RSPK"
                #print "ElphyFile.create_layout() - n_events",block.n_events
                #print "ElphyFile.create_layout() - n_evt_channels",block.n_evt_channels
            
            layout.add_block(block)
            offset += block.size
            
            # set up as soon as possible the shortcut 
            # to the main block of a DAC2GSLayout
            if not detect_main and isinstance(layout, DAC2GSLayout) and isinstance(block, DAC2GSMainBlock) :
                layout.set_main_block()
                detect_main = True
 
            # detect if the file is continuous when
            # the 'MAIN' block has been parsed
            if not detect_continuous :
                is_continuous = isinstance(header, DAC2GSHeader) and layout.is_continuous()
        
        # set up the shortcut to blocks corresponding
        # to episodes, only available for DAC2Layout
        # and also DAC2GSLayout if not continuous 
        if isinstance(layout, DAC2Layout) or (isinstance(layout, DAC2GSLayout) and not layout.is_continuous()) :
            layout.set_episode_blocks()
        
        layout.set_data_blocks()
        
        # finally set up the user info block of the layout
        layout.set_info_block()

        self.file.seek(0)
        return layout

    def is_continuous(self):
        return self.layout.is_continuous()
    
    @property
    def n_episodes(self):
        """
        Return the number of recording sequences.
        """
        return self.layout.n_episodes
    
    def n_channels(self, episode):
        """
        Return the number of recording
        channels involved in data acquisition
        and relative to the specified episode :
        
        ``episode`` : the recording sequence identifier. 
        """
        return self.layout.n_channels(episode)
    
    def n_tags(self, episode):
        """
        Return the number of tag channels
        relative to the specified episode :
        
        ``episode`` : the recording sequence identifier. 
        """
        return self.layout.n_tags(episode)
    
    def n_events(self, episode):
        """
        Return the number of event channels
        relative to the specified episode :
        
        ``episode`` : the recording sequence identifier. 
        """
        return self.layout.n_events(episode)
    
    def n_spiketrains(self, episode):
        """
        Return the number of event channels
        relative to the specified episode :
        
        ``episode`` : the recording sequence identifier. 
        """
        return self.layout.n_spiketrains(episode)
    
    def n_waveforms(self, episode):
        """
        Return the number of waveform channels :
        """
        return self.layout.n_waveforms(episode)
    
    def get_signal(self, episode, channel):
        """
        Return the signal or event descriptor relative
        to the specified episode and channel :
        
        ``episode`` : the recording sequence identifier.
        ``channel`` : the analog channel identifier.
        
        NB : For 'DAC2 objects' format, it could
        be also used to retrieve events.
        """
        return self.layout.get_signal(episode, channel)
    
    def get_tag(self, episode, tag_channel):
        """
        Return the tag descriptor relative to
        the specified episode and tag channel :
        
        ``episode`` : the recording sequence identifier.
        ``tag_channel`` : the tag channel identifier.
        
        NB : There isn't any tag channels for
        'Acquis1' format. ElphyTag channels appeared
        after 'DAC2/GS/2000' release. They are
        also present in 'DAC2 objects' format.
        
        """
        return self.layout.get_tag(episode, tag_channel)

    def get_event(self, episode, evt_channel):
        """
        Return the event relative the
        specified episode and event channel.
        
        `episode`` : the recording sequence identifier.
        ``tag_channel`` : the tag channel identifier.
        """
        return self.layout.get_event(episode, evt_channel)
    
    def get_spiketrain(self, episode, electrode_id):
        """
        Return the spiketrain relative to the
        specified episode and electrode_id.
        
        ``episode`` : the recording sequence identifier.
        ``electrode_id`` : the identifier of the electrode providing the spiketrain.
        
        NB : Available only for 'DAC2 objects' format.
        This descriptor can return the times of a spiketrain
        and waveforms relative to each of these times.
        """
        return self.layout.get_spiketrain(episode, electrode_id)

    @property
    def comments(self):
        raise NotImplementedError()
    
    def get_user_file_info(self):
        """
        Return user defined file metadata.
        """ 
        if not self.layout.info_block :
            return dict()
        else :
            return self.layout.info_block.get_user_file_info()
    
    @property
    def episode_info(self, ep_number):
        raise NotImplementedError()
    
    def get_signals(self):
        """
        Get all available analog or event channels stored into an Elphy file.
        """
        signals = list()
        for ep in range(1, self.n_episodes + 1) :
            for ch in range(1, self.n_channels(ep) + 1) :
                signal = self.get_signal(ep, ch)
                signals.append(signal)
        return signals
    
    def get_tags(self):
        """
        Get all available tag channels stored into an Elphy file.
        """
        tags = list()
        for ep in range(1, self.n_episodes + 1) :
            for tg in range(1, self.n_tags(ep) + 1) :
                tag = self.get_tag(ep, tg)
                tags.append(tag)
        return tags

    def get_spiketrains(self):
        """
        Get all available spiketrains stored into an Elphy file.
        """
        spiketrains = list()
        for ep in range(1, self.n_episodes + 1) :
            for ch in range(1, self.n_spiketrains(ep) + 1) :
                spiketrain = self.get_spiketrain(ep, ch)
                spiketrains.append(spiketrain)
        return spiketrains
    
    def get_rspk_spiketrains(self):
        """
        Get all available spiketrains stored into an Elphy file.
        """
        spiketrains = list()
        spk_blocks = self.layout.get_blocks_of_type('RSPK')
        for bl in spk_blocks :
            #print "ElphyFile.get_spiketrains() - identifier:",bl.identifier
            for ch in range(0,bl.n_evt_channels) :
                spiketrain = self.layout.get_rspk_data(ch)
                spiketrains.append(spiketrain)
        return spiketrains

    def get_names( self ) :
        com_blocks = list()
        com_blocks = self.layout.get_blocks_of_type('COM')
        return com_blocks



# --------------------------------------------------------



class ElphyIO(BaseIO):
    """
    Class for reading from and writing to an Elphy file.

    It enables reading:
    - :class:`Block`
    - :class:`Segment`
    - :class:`RecordingChannel`
    - :class:`RecordingChannelGroup`
    - :class:`EventArray`
    - :class:`SpikeTrain`

    Usage:
        >>> from neo import io
        >>> r = io.ElphyIO(filename='ElphyExample.DAT')
        >>> seg = r.read_block(lazy=False, cascade=True)
        >>> print(seg.analogsignals)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        >>> print(seg.spiketrains)    # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        >>> print(seg.eventarrays)    # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        >>> print(anasig._data_description)
        >>> anasig = r.read_analogsignal(lazy=False, cascade=False)

        >>> bl = Block()
        >>> # creating segments, their contents and append to bl
        >>> r.write_block( bl )
    """
    is_readable = True # This class can read data
    is_writable = False # This class can write data
    # This class is able to directly or indirectly handle the following objects
    supported_objects  = [ Block, Segment, AnalogSignalArray, SpikeTrain ] #, AnalogSignal
    # This class can return a Block
    readable_objects    = [ Block ]
    # This class is not able to write objects
    writeable_objects   = [ ]
    has_header         = False
    is_streameable     = False
    # This is for GUI stuff : a definition for parameters when reading.
    # This dict should be keyed by object (`Block`). Each entry is a list
    # of tuple. The first entry in each tuple is the parameter name. The
    # second entry is a dict with keys 'value' (for default value),
    # and 'label' (for a descriptive name).
    # Note that if the highest-level object requires parameters,
    # common_io_test will be skipped.
    read_params = {
    }
    # do not supported write so no GUI stuff
    write_params       = {
    }
    name               = 'Elphy IO'
    extensions         = [ 'DAT' ]
    # mode can be 'file' or 'dir' or 'fake' or 'database'
    mode = 'file' 
    # internal serialized representation of neo data
    serialized = None
    

    def __init__(self , filename = None) :
        """
        Arguments:
            filename : the filename to read
        """
        BaseIO.__init__(self)
        self.filename = filename
        self.elphy_file = ElphyFile(self.filename)


    def read_block(self,
                   # the 2 first key arguments are imposed by neo.io API
                   lazy = False,
                   cascade = True
                   ):
        """
        Return :class:`Block` filled or not depending on 'cascade' parameter.

        Parameters:
             lazy : postpone actual reading of the file.
             cascade : normally you want this True, otherwise method will only ready Block label.
        """
        # basic
        block = Block(name=None)
        # laziness
        if lazy:
            return block
        else:
            # get analog and tag channels
            try :
                self.elphy_file.open()
            except Exception as e:
                self.elphy_file.close()
                raise Exception("cannot open file %s : %s" % (self.filename, e))
        # cascading
        #print "\n\n==========================================\n"
        #print "read_block() - n_episodes:",self.elphy_file.n_episodes
        if cascade:
            # create a segment containing all analog,
            # tag and event channels for the episode
            if self.elphy_file.n_episodes == None :
                print("File '%s' appears to have no episodes" % (self.filename))
                return block
            for episode in range(1, self.elphy_file.n_episodes+1) :
                segment = self.read_segment(episode)
                segment.block = block
                block.segments.append(segment)
        # close file
        self.elphy_file.close()
        # result    
        return block
        



    def write_block( self, block ):
        """
        Write a given Neo Block to an Elphy file, its structure being, for example:
        Neo                        ->  Elphy
        --------------------------------------------------------------
        Block                          File
            Segment                        Episode Block (B_Ep)
                AnalogSignalArray              Episode Descriptor (Ep + Adc + Ksamp + Ktype)
                    multichannel           RDATA (with a ChannelMask multiplexing channels)
                    2D NumPy Array                     
                    ...
                AnalogSignalArray
                    AnalogSignal
                    AnalogSignal
                    ...
                ...
                SpikeTrain                 Event Block (RSPK) 
                SpikeTrain
                ...

        Arguments::
            block: the block to be saved
        """
        # Serialize Neo structure into Elphy file
        # each analog signal will be serialized as elphy Episode Block (with its subblocks)
        # then all spiketrains will be serialized into an Rspk Block (an Event Block with addons).
        # Serialize (and size) all Neo structures before writing them to file
        # Since to write each Elphy Block is required to know in advance its size,
        # which includes that of its subblocks, it is necessary to 
        # serialize first the lowest structures.
        # Iterate over block structures
        elphy_limit = 256
        All = ''
        #print "\n\n--------------------------------------------\n"
        #print "write_block() - n_segments:",len(block.segments)
        for seg in block.segments:
            analogsignals = 0 # init
            nbchan = 0
            nbpt = 0
            chls = 0
            Dxu = 1e-8 #0.0000001
            Rxu = 1e+8 #10000000.0
            X0uSpk = 0.0
            CyberTime = 0.0
            aa_units = []
            NbEv = []
            serialized_analog_data = ''
            serialized_spike_data = ''
            # AnalogSignals
            # Neo signalarrays are 2D numpy array where each row is an array of samples for a channel:
            # signalarray A = [[  1,  2,  3,  4 ],
            #                  [  5,  6,  7,  8 ]]
            # signalarray B = [[  9, 10, 11, 12 ],
            #                  [ 13, 14, 15, 16 ]]
            # Neo Segments can have more than one signalarray.
            # To be converted in Elphy analog channels they need to be all in a 2D array, not in several 2D arrays.
            # Concatenate all analogsignalarrays into one and then flatten it.
            # Elphy RDATA blocks contain Fortran styled samples:
            # 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16
            # AnalogSignalArrays -> analogsignals
            # get the first to have analogsignals with the right shape
            # Annotations for analogsignals array come as a list of int being source ids
            # here, put each source id on a separate dict entry in order to have a matching afterwards
            idx = 0
            annotations = dict( )
            # get all the others
            #print "write_block() - n_analogsignals:",len(seg.analogsignals)
            #print "write_block() - n_analogsignalarrays:",len(seg.analogsignalarrays)
            for asigar in seg.analogsignalarrays :
                idx,annotations = self.get_annotations_dict( annotations, "analogsignal", asigar.annotations.items(), asigar.name, idx )
                # array structure
                _,chls = asigar.shape
                # units
                for _ in range(chls) :
                    aa_units.append( asigar.units )
                Dxu = asigar.sampling_period
                Rxu = asigar.sampling_rate
                if isinstance(analogsignals, np.ndarray) :
                    analogsignals = np.hstack( (analogsignals,asigar) )
                else :
                    analogsignals = asigar # first time
            # collect and reshape all analogsignals
            if isinstance(analogsignals, np.ndarray) :
                # transpose matrix since in Neo channels are column-wise while in Elphy are row-wise
                analogsignals = analogsignals.T
                # get dimensions
                nbchan,nbpt = analogsignals.shape
                # serialize AnalogSignal
                analog_data_fmt = '<' + str(analogsignals.size) + 'f'
                # serialized flattened numpy channels in 'F'ortran style
                analog_data_64 = analogsignals.flatten('F')
                # elphy normally uses float32 values (for performance reasons)
                analog_data = np.array( analog_data_64, dtype=np.float32 )
                serialized_analog_data += struct.pack( analog_data_fmt, *analog_data )
            # SpikeTrains
            # Neo spiketrains are stored as a one-dimensional array of times
            # [ 0.11, 1.23, 2.34, 3.45, 4.56, 5.67, 6.78, 7.89 ... ]
            # These are converted into Elphy Rspk Block which will contain all of them
            # RDATA + NbVeV:integer for the number of channels (spiketrains) 
            # + NbEv:integer[] for the number of event per channel
            # followed by the actual arrays of integer containing spike times
            #spiketrains = seg.spiketrains
            # ... but consider elphy loading limitation:
            NbVeV = len( seg.spiketrains )
            #print "write_block() - n_spiketrains:",NbVeV
            if len(seg.spiketrains) > elphy_limit :
                NbVeV = elphy_limit
            # serialize format
            spiketrain_data_fmt = '<'
            spiketrains = []
            for idx,train in enumerate(seg.spiketrains[:NbVeV]) :
                #print "write_block() - train.size:", train.size,idx
                #print "write_block() - train:", train
                fake,annotations = self.get_annotations_dict( annotations,"spiketrain", train.annotations.items(), '', idx )
                #annotations.update( dict( [("spiketrain-"+str(idx),train.annotations['source_id'])] ) )
                #print "write_block() - train[%s].annotation['source_id']:%s" % (idx,train.annotations['source_id'])
                # total number of events format + blackrock sorting mark (0 for neo)
                spiketrain_data_fmt += str(train.size) + "i" + str(train.size) + "B"
                # get starting time
                X0uSpk = train.t_start.item()
                CyberTime = train.t_stop.item()
                # count number of events per train
                NbEv.append( train.size )
                # multiply by sampling period
                train = train * Rxu
                # all flattened spike train
                # blackrock acquisition card also adds a byte for each event to sort it
                spiketrains.extend( [spike.item() for spike in train] + [0 for _ in range(train.size)])
            # Annotations
            #print annotations
            # using DBrecord elphy block, they will be available as values in elphy environment
            # separate keys and values in two separate serialized strings
            ST_sub = ''
            st_fmt = ''
            st_data = []
            BUF_sub = ''
            serialized_ST_data = '' 
            serialized_BUF_data = ''
            for key in sorted(annotations.iterkeys()) :
                # take all values, get their type and concatenate
                fmt = ''
                data = []
                value = annotations[key]
                if isinstance( value, (int,np.int32,np.int64) ) :
                    # elphy type 2
                    fmt = '<Bq'
                    data = [2, value] 
                elif type( value ) == str :
                    # elphy type 4
                    str_len = len(value)
                    fmt = '<BI'+str(str_len)+'s'
                    data = [4, str_len, value] 
                else :
                    print("ElphyIO.write_block() - unknown annotation type: %s" % type(value))
                    continue
                # last, serialization
                # BUF values 
                serialized_BUF_data += struct.pack( fmt, *data )
                # ST values 
                # take each key and concatenate using 'crlf'
                st_fmt += str(len(key))+'s2s'
                st_data.extend( [ key, "\r\n" ] )
            # ST keys
            serialized_ST_data = struct.pack( st_fmt, *st_data )
            # SpikeTrains
            # serialized spike trains
            serialized_spike_data += struct.pack( spiketrain_data_fmt, *spiketrains )
            # ------------- Elphy Structures to be filled --------------
            # 'Ep'
            data_format = '<BiBB10sdd?BBddiddB10sB10sdI'
            # setting values
            uX = 'ms        '
            pc_time = datetime.now()
            pc_time = pc_time.microsecond * 1000
            data_values = [
                nbchan, # nbchan : byte
                nbpt,   # nbpt : integer - nominal number of samples per channel
                0,      # tpData : byte - not used
                10,     # uX length
                uX,     # uX : string - time units
                Dxu,    # Dxu : double - sampling rate, scaling parameters on time axis 
                0.0,    # X0u : double - starting, scaling parameters on time axis
                False,  # continuous : boolean
                0,      # TagMode : byte - 0: not a tag channel
                0,      # TagShift : byte
                Dxu,    # DxuSpk : double
                X0uSpk, # X0uSpk : double
                NbVeV,  # nbSpk : integer
                0.0,    # DyuSpk : double
                0.0,    # Y0uSpk : double
                10,     # uX length
                uX,     # unitXSpk : string
                10,     # uX length
                '          ',    # unitYSpk : string
                CyberTime,    # CyberTime : double
                pc_time # PCtime : longword - time in milliseconds
            ]
            Ep_chr = self.get_serialized( data_format, data_values )
            Ep_sub = self.get_serialized_subblock( 'Ep', Ep_chr )
            # 'Adc'
            # Then, one or more (nbchan) Analog/Digital Channel will be, having their fixed data format 
            data_format = "<B10sdd"
            # when Ep.tpdata is an integer type, Dyu nad Y0u are parameters such that
            # for an adc value j, the real value is y = Dyu*j + Y0u
            Adc_chrl = ""
            for dc in aa_units :
                # create 
                Adc_chr = [] # init
                Dyu,UnitY = '{}'.format(dc).split()
                data_values = [
                    10, # size
                    UnitY+'        ',   # uY string : vertical units
                    float(Dyu),  # Dyu double : scaling parameter
                    0.0   # Y0u double : scaling parameter            
                ]
                Adc_chr = self.get_serialized( data_format, data_values )
                Adc_chrl += Adc_chr
            Adc_sub = self.get_serialized_subblock( 'Adc', Adc_chrl )
            #print "Adc size:",len(Adc_sub)
            # 'Ksamp'
            # subblock containing an array of nbchan bytes
            # data_format = '<h...' # nbchan times Bytes
            # data_values = [ 1, 1, ... ] # nbchan times 1
            data_format = "<" + ("h" * nbchan)
            data_values = [ 1 for _ in range(nbchan) ]
            Ksamp_chr = self.get_serialized( data_format, data_values )
            Ksamp_sub = self.get_serialized_subblock( 'Ksamp', Ksamp_chr )
            #print "Ksamp size: %s" % (len(Ksamp_sub))
            # 'Ktype'
            # subblock containing an array of nbchan bytes
            # data_format = '<B...' # nbchan times Bytes
            # data_values = [ 2, ... ] # nbchan times ctype
            # Possible values are:
            #  0: byte 
            #  1: short 
            #  2: smallint 
            #  3: word 
            #  4: longint 
            #  5: single 
            #  6: real48 
            #  7: double 
            #  8: extended DATA
            # array of nbchan bytes specifying type of data forthcoming
            ctype = 5 # single float
            data_format = "<" + ("B" * nbchan)
            data_values = [ ctype for n in range(nbchan) ]
            Ktype_chr = self.get_serialized( data_format, data_values )
            Ktype_sub = self.get_serialized_subblock( 'Ktype', Ktype_chr )
            #print "Ktype size: %s" % (len(Ktype_sub))
            # Episode data serialization:
            # concatenate all its data strings under a block
            Ep_data = Ep_sub + Adc_sub + Ksamp_sub + Ktype_sub
            #print "\n---- Finishing:\nEp subs size: %s" % (len(Ep_data))
            Ep_blk = self.get_serialized_block( 'B_Ep', Ep_data )
            #print "B_Ep size: %s" % (len(Ep_blk))
            # 'RDATA'
            # It produces a two part (header+data) content coming from analog/digital inputs.
            pctime = time()
            data_format = "<h?dI"
            data_values = [ 15, True, pctime, 0 ]
            RDATA_chr = self.get_serialized( data_format, data_values, serialized_analog_data )
            RDATA_blk = self.get_serialized_block( 'RDATA', RDATA_chr )
            #print "RDATA size: %s" % (len(RDATA_blk))
            # 'Rspk'
            # like an REVT block + addons
            # It starts with a RDATA header, after an integer with the number of events, 
            # then the events per channel and finally all the events one after the other
            data_format = "<h?dII" + str(NbVeV) + "I"
            data_values = [ 15, True, pctime, 0, NbVeV ]
            data_values.extend(NbEv)
            Rspk_chr = self.get_serialized( data_format, data_values, serialized_spike_data )
            Rspk_blk = self.get_serialized_block( 'RSPK', Rspk_chr )
            #print "RSPK size: %s" % (len(Rspk_blk))
            # 'DBrecord'
            # like a block + subblocks
            # serializzation
            ST_sub = self.get_serialized_subblock( 'ST', serialized_ST_data )
            #print "ST size: %s" % (len(ST_sub))
            BUF_sub = self.get_serialized_subblock( 'BUF', serialized_BUF_data )
            #print "BUF size: %s" % (len(BUF_sub))
            annotations_data = ST_sub + BUF_sub
            #data_format = "<h?dI"
            #data_values = [ 15, True, pctime, 0 ]
            #DBrec_chr = self.get_serialized( data_format, data_values, annotations_data )
            DBrec_blk = self.get_serialized_block( 'DBrecord', annotations_data )
            #print "DBrecord size: %s" % (len(DBrec_blk))
            # 'COM'
            #print "write_block() - segment name:", seg.name
            # name of the file - NEO Segment name
            data_format = '<h'+str(len(seg.name))+'s'
            data_values = [ len(seg.name), seg.name ]
            SEG_COM_chr = self.get_serialized( data_format, data_values )
            SEG_COM_blk = self.get_serialized_block( 'COM', SEG_COM_chr )
            # Complete data serialization: concatenate all data strings
            All += Ep_blk + RDATA_blk + Rspk_blk + DBrec_blk + SEG_COM_blk
        # ElphyFile (open, write and close)
        self.elphy_file.write( All )



    def get_serialized( self, data_format, data_values, ext_data='' ):
        data_chr = struct.pack( data_format, *data_values )
        return data_chr + ext_data



    def get_serialized_block( self, ident, data ):
        """
        Generic Block Header
        This function (without needing a layout and the rest) creates a binary serialized version of
        the block containing the format string and the actual data for the following 
        Elphy Block Header structure:
        size: longint   // 4-byte integer 
        ident: string[XXX]; // a Pascal variable-length string 
        data: array[1..YYY] of byte; 

        For example:
        '<IB22s' followed by an array of bytes as specified
        """
        #        endian 4byte        ident 
        data_format = "<IB" + str(len(ident))+"s"
        data_size = 4 + 1 + len(ident) + len(data) # all: <IBs...data...
        data_values = [ data_size, len(ident), ident ]
        data_chr = struct.pack( data_format, *data_values )
        return data_chr + data



    def get_serialized_subblock( self, ident, data ):
        """
        Generic Sub-Block Header
        This function (without needing a layout and the rest) creates a binary serialized version of
        the block containing the format string and the actual data for the following 
        Elphy Sub-Block Header structure:
        id: string[XXX]; // a Pascal variable-length string 
        size1: word      // 2-byte unsigned integer 
        data: array[1..YYY] of byte; 

        For example:
        '<B22sH4522L' followed by an array of bytes as specified
        """
        data_size = len( data )
        #            endian  size+string         2byte   array of data_size bytes
        data_format = "<B" + str(len(ident))+"s" + "h" 
        data_values = [ len(ident), ident, data_size ]
        data_chr = struct.pack( data_format, *data_values )
        return data_chr + data



    def get_annotations_dict( self, annotations, prefix, items, name='', idx=0 ) :
        """
        Helper function to retrieve annotations in a dictionary to be serialized as Elphy DBrecord
        """
        for (key,value) in items :
            #print "get_annotation_dict() - items[%s]" % (key)
            if isinstance( value, (list,tuple,np.ndarray) ) :
                for element in value :
                    annotations.update( dict( [(prefix+"-"+name+"-"+key+"-"+str(idx), element)] ) )
                    idx = idx+1
            else :
                annotations.update( dict( [(prefix+"-"+key+"-"+str(idx),value)] ) )
        return (idx,annotations)



    def read_segment( self, episode ):
        """
        Internal method used to return :class:`Segment` data to the main read method.
        Parameters:
            elphy_file : is the elphy object.
            episode : number of elphy episode, roughly corresponding to a segment
        """
        #print "name:",self.elphy_file.layout.get_episode_name(episode)
        episode_name = self.elphy_file.layout.get_episode_name(episode)
        name = episode_name if len(episode_name)>0 else "episode %s" % str(episode + 1)
        segment = Segment( name=name )
        # create an analog signal for
        # each channel in the episode
        for channel in range(1, self.elphy_file.n_channels(episode)+1) :
            signal = self.elphy_file.get_signal(episode, channel)
            analog_signal = AnalogSignal(
                signal.data['y'],
                units = signal.y_unit,
                t_start = signal.t_start * getattr(pq, signal.x_unit.strip()),
                t_stop = signal.t_stop * getattr(pq, signal.x_unit.strip()),
                #sampling_rate = signal.sampling_frequency * pq.kHz,
                sampling_period = signal.sampling_period * getattr(pq, signal.x_unit.strip()),
                channel_name="episode %s, channel %s" % ( int(episode+1), int(channel+1) )
            )
            analog_signal.segment = segment
            segment.analogsignals.append(analog_signal)
        # create a spiketrain for each
        # spike channel in the episode
        # in case of multi-electrode
        # acquisition context
        n_spikes = self.elphy_file.n_spiketrains(episode)
        #print "read_segment() - n_spikes:",n_spikes
        if n_spikes>0 :
            for spk in range(1, n_spikes+1) :
                spiketrain = self.read_spiketrain(episode, spk)
                spiketrain.segment = segment
                segment.spiketrains.append( spiketrain )
        # segment
        return segment




    def read_recordingchannelgroup( self, episode ):
        """
        Internal method used to return :class:`RecordingChannelGroup` info.

        Parameters:
            elphy_file : is the elphy object.
            episode : number of elphy episode, roughly corresponding to a segment
        """
        n_spikes = self.elphy_file.n_spikes
        group = RecordingChannelGroup(
            name="episode %s, group of %s electrodes" % (episode, n_spikes)
        )
        for spk in range(0, n_spikes) :
            channel = self.read_recordingchannel(episode, spk)
            group.recordingchannels.append(channel)
        return group
            



    def read_recordingchannel( self, episode, chl ):
        """
        Internal method used to return a :class:`RecordingChannel` label.

        Parameters:
            elphy_file : is the elphy object.
            episode : number of elphy episode, roughly corresponding to a segment.
            chl : electrode number.
        """
        channel = RecordingChannel(
            name="episode %s, electrodes %s" % (episode, chl)
        )
        return channel
        



    def read_eventarray( self, episode, evt ):
        """
        Internal method used to return a list of elphy :class:`EventArray` acquired from event channels.

        Parameters:
            elphy_file : is the elphy object.
            episode : number of elphy episode, roughly corresponding to a segment.
            evt : index of the event.
        """
        event = self.elphy_file.get_event(episode, evt)
        event_array = EventArray(
            times=event.times * pq.s,
            channel_name="episode %s, event channel %s" % (episode + 1, evt + 1)
        )
        return event_array
    



    def read_spiketrain( self, episode, spk ):
        """
        Internal method used to return an elphy object :class:`SpikeTrain`.

        Parameters:
            elphy_file : is the elphy object.
            episode : number of elphy episode, roughly corresponding to a segment.
            spk : index of the spike array.
        """
        block = self.elphy_file.layout.episode_block(episode)
        spike = self.elphy_file.get_spiketrain(episode, spk)
        spikes = spike.times * pq.s
        #print "read_spiketrain() - spikes: %s" % (len(spikes))
        #print "read_spiketrain() - spikes:",spikes
        dct = {
            'times':spikes,
            't_start': block.ep_block.X0_wf if block.ep_block.X0_wf < spikes[0] else spikes[0], #check
            't_stop': block.ep_block.cyber_time if block.ep_block.cyber_time > spikes[-1] else spikes[-1],
            'units':'s',
            # special keywords to identify the 
            # electrode providing the spiketrain
            # event though it is redundant with
            # waveforms
            'label':"episode %s, electrode %s" % (episode, spk),
            'electrode_id':spk
        }
        # new spiketrain
        return SpikeTrain(**dct)
  
