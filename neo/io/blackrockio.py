# -*- coding: utf-8 -*-
"""Module for reading binary file from Blackrock format.
"""

from .baseio import BaseIO
from ..core import *
import numpy as np
import struct
import quantities as pq  
from neo.io import tools


import logging

class BlackrockIO(BaseIO):
    """
    Class for reading/writing data in a BlackRock Neuroshare ns5 files.
    """    
    # Class variables demonstrating capabilities of this IO
    is_readable        = True # This a only reading class
    is_writable        = True # write is not supported
    
    # This IO can only manipulate continuous data, not spikes or events
    supported_objects  = [Block, Segment, AnalogSignal, RecordingChannelGroup, RecordingChannel]
    
    # Keep things simple by always returning a block
    readable_objects    = [Block]
    
    # And write a block
    writeable_objects   = [Block]

    # Not sure what these do, if anything
    has_header         = False
    is_streameable     = False
    
    # The IO name and the file extensions it uses
    name               = 'Blackrock'    
    extensions          = ['ns5']
    
    # Operates on *.ns5 files
    mode = 'file'  
    
    # GUI defaults for reading
    # Most information is acquired from the file header.
    read_params = { 
        Block: [
            #('rangemin' , { 'value' : -10 } ),
            #('rangemax' , { 'value' : 10 } ),
            ]
        }

    # GUI defaults for writing (not supported)
    write_params       = None

    
    def __init__(self, filename, full_range=8192.*pq.mV) :
        """Initialize Blackrock reader.
        
        **Arguments**
            filename: string, the filename to read
            full_range: Quantity, the full-scale analog range of the data.
                This is set by your digitizing hardware. It should be in
                volts or millivolts.
        """
        BaseIO.__init__(self)
        self.filename = filename
        self.full_range = full_range
    
    # The reading methods. The `lazy` and `cascade` parameters are imposed
    # by neo.io API    
    def read_block(self, lazy=False, cascade=True, 
        n_starts=None, n_stops=None, channel_list=None):
        """Reads the file and returns contents as a Block.
        
        The Block contains one Segment for each entry in zip(n_starts,
        n_stops). If these parameters are not specified, the default is
        to store all data in one Segment.
        
        The Block also contains one RecordingChannelGroup for all channels.
        
        n_starts: list or array of starting times of each Segment in
            samples from the beginning of the file.
        n_stops: similar, stopping times of each Segment
        channel_list: list of channel numbers to get. The neural data channels
            are 1 - 128. The analog inputs are 129 - 144. The default
            is to acquire all channels.
        
        Returns: Block object containing the data.
        """


        # Create block
        block = Block(file_origin=self.filename)
        
        if not cascade:
            return block
        
        self.loader = Loader(self.filename)
        self.loader.load_file()
        self.header = self.loader.header
        
        # If channels not specified, get all
        if channel_list is None:
            channel_list = self.loader.get_neural_channel_numbers()
        
        # If not specified, load all as one Segment
        if n_starts is None:
            n_starts = [0]
            n_stops = [self.loader.header.n_samples]
        
        #~ # Add channel hierarchy
        #~ rcg = RecordingChannelGroup(name='allchannels',
            #~ description='group of all channels', file_origin=self.filename)
        #~ block.recordingchannelgroups.append(rcg)
        #~ self.channel_number_to_recording_channel = {}

        #~ # Add each channel at a time to hierarchy
        #~ for ch in channel_list:            
            #~ ch_object = RecordingChannel(name='channel%d' % ch,
                #~ file_origin=self.filename, index=ch)
            #~ rcg.channel_indexes.append(ch_object.index)
            #~ rcg.channel_names.append(ch_object.name)
            #~ rcg.recordingchannels.append(ch_object)
            #~ self.channel_number_to_recording_channel[ch] = ch_object

        # Iterate through n_starts and n_stops and add one Segment
        # per each.
        for n, (t1, t2) in enumerate(zip(n_starts, n_stops)):
            # Create segment and add metadata
            seg = self.read_segment(n_start=t1, n_stop=t2, chlist=channel_list,
                lazy=lazy, cascade=cascade)
            seg.name = 'Segment %d' % n
            seg.index = n
            t1sec = t1 / self.loader.header.f_samp
            t2sec = t2 / self.loader.header.f_samp
            seg.description = 'Segment %d from %f to %f' % (n, t1sec, t2sec)
            
            # Link to block
            block.segments.append(seg)
        
        # Create hardware view, and bijectivity
        tools.populate_RecordingChannel(block)
        tools.create_many_to_one_relationship(block)        
        
        return block
    
    def read_segment(self, n_start, n_stop, chlist=None, lazy=False, cascade=True):
        """Reads a Segment from the file and stores in database.
        
        The Segment will contain one AnalogSignal for each channel
        and will go from n_start to n_stop (in samples).
        
        Arguments:
            n_start : time in samples that the Segment begins
            n_stop : time in samples that the Segment ends
        
        Python indexing is used, so n_stop is not inclusive.
        
        Returns a Segment object containing the data.
        """
        # If no channel numbers provided, get all of them
        if chlist is None:
            chlist = self.loader.get_neural_channel_numbers()
        
        # Conversion from bits to full_range units
        conversion = self.full_range / 2**(8*self.header.sample_width)
        
        # Create the Segment
        seg = Segment(file_origin=self.filename)
        t_start = float(n_start) / self.header.f_samp
        t_stop = float(n_stop) / self.header.f_samp        
        seg.annotate(t_start=t_start)
        seg.annotate(t_stop=t_stop)
        
        # Load data from each channel and store
        for ch in chlist:
            if lazy:
                sig = np.array([]) * conversion
            else:
                # Get the data from the loader
                sig = np.array(\
                    self.loader._get_channel(ch)[n_start:n_stop]) * conversion

            # Create an AnalogSignal with the data in it
            anasig = AnalogSignal(signal=sig,
                sampling_rate=self.header.f_samp*pq.Hz,
                t_start=t_start*pq.s, file_origin=self.filename,
                description='Channel %d from %f to %f' % (ch, t_start, t_stop),
                channel_index=ch)
            
            if lazy:
                anasig.lazy_shape = n_stop-n_start
                
            
            # Link the signal to the segment
            seg.analogsignals.append(anasig)
            
            # Link the signal to the recording channel from which it came
            #rc = self.channel_number_to_recording_channel[ch]
            #rc.analogsignals.append(anasig)
        
        return seg


    def write_block(self, block):
        """Writes block to `self.filename`.
        
        *.ns5 BINARY FILE FORMAT
        The following information is contained in the first part of the header
        file.
        The size in bytes, the variable name, the data type, and the meaning are
        given below. Everything is little-endian.

        8B. File_Type_ID. char. Always "NEURALSG"
        16B. File_Spec. char. Always "30 kS/s\0"
        4B. Period. uint32. Always 1.
        4B. Channel_Count. uint32. Generally 32 or 34.
        Channel_Count*4B. uint32. Channel_ID. One uint32 for each channel.

        Thus the total length of the header is 8+16+4+4+Channel_Count*4.
        Immediately after this header, the raw data begins.
        Each sample is a 2B signed int16.
        For our hardware, the conversion factor is 4096.0 / 2**16 mV/bit.
        The samples for each channel are interleaved, so the first Channel_Count
        samples correspond to the first sample from each channel, in the same
        order as the channel id's in the header.

        Variable names are consistent with the Neuroshare specification.
        """                
        fi = file(self.filename, 'wb')
        self._write_header(block, fi)
        
        # Write each segment in order
        for seg in block.segments:
            # Create a 2d numpy array of analogsignals converted to bytes
            all_signals = np.array([
                np.rint(sig * 2**16 / self.full_range)
                for sig in seg.analogsignals],
                dtype=np.int)
            
            # Write to file. We transpose because channel changes faster
            # than time in this format.
            for vals in all_signals.transpose():
                fi.write(struct.pack('<%dh' % len(vals), *vals))

        fi.close()
        
    
    def _write_header(self, block, fi):
        """Write header info about block to fi"""
        if len(block.segments) > 0:
            channel_indexes = channel_indexes_in_segment(block.segments[0])
        else:
            channel_indexes = []
            seg = block.segments[0]

        # type of file
        fi.write('NEURALSG')
        
        # sampling rate, in text and integer
        fi.write('30 kS/s\0')
        for n in range(8): fi.write('\0')
        fi.write(struct.pack('<I', 1))
        
        # channel count: one for each analogsignal, and then also for
        # each column in each analogsignalarray        
        fi.write(struct.pack('<I', len(channel_indexes)))
        for chidx in channel_indexes:
            fi.write(struct.pack('<I', chidx))
    
def channel_indexes_in_segment(seg):
    """List channel indexes of analogsignals and analogsignalarrays"""
    channel_indices = []
    for sig in seg.analogsignals:
        channel_indices.append(sig.recordingchannel.index)
    
    for asa in seg.analogsignalarrays:
        channel_indices.append(asa.recordingchannelgroup.channel_indexes)
    
    return channel_indices

class HeaderInfo:
    """Holds information from the ns5 file header about the file."""
    pass       

class Loader(object):
    """Object to load data from binary ns5 files.
    
    Methods
    -------
    load_file : actually create links to file on disk
    load_header : load header info and store in self.header
    get_channel_as_array : Returns 1d numpy array of the entire recording
        from requested channel.
    get_analog_channel_as_array : Same as get_channel_as_array, but works
        on analog channels rather than neural channels.
    get_analog_channel_ids : Returns an array of analog channel numbers
        existing in the file.    
    get_neural_channel_ids : Returns an array of neural channel numbers
        existing in the file.
    regenerate_memmap : Deletes and restores the underlying memmap, which
        may free up memory.
    
    Issues
    ------
    Memory leaks may exist
    Not sure that regenerate_memmap actually frees up any memory.
    """
    def __init__(self, filename=None):
        """Creates a new object to load data from the ns5 file you specify.

        filename : path to ns5 file
        Call load_file() to actually get data from the file.
        """
        self.filename = filename
        
        
        
        self._mm = None
        self.file_handle = None

    def load_file(self, filename=None):
        """Loads an ns5 file, if not already done.
        
        *.ns5 BINARY FILE FORMAT
        The following information is contained in the first part of the header
        file.
        The size in bytes, the variable name, the data type, and the meaning are
        given below. Everything is little-endian.

        8B. File_Type_ID. char. Always "NEURALSG"
        16B. File_Spec. char. Always "30 kS/s\0"
        4B. Period. uint32. Always 1.
        4B. Channel_Count. uint32. Generally 32 or 34.
        Channel_Count*4B. uint32. Channel_ID. One uint32 for each channel.

        Thus the total length of the header is 8+16+4+4+Channel_Count*4.
        Immediately after this header, the raw data begins.
        Each sample is a 2B signed int16.
        For our hardware, the conversion factor is 4096.0 / 2**16 mV/bit.
        The samples for each channel are interleaved, so the first Channel_Count
        samples correspond to the first sample from each channel, in the same
        order as the channel id's in the header.

        Variable names are consistent with the Neuroshare specification.
        """
        # If filename specified, use it, else use previously specified
        if filename is not None: self.filename = filename
        
        # Load header info into self.header
        self.load_header()
        
        # build an internal memmap linking to the data on disk
        self.regenerate_memmap()
    
    def load_header(self, filename=None):
        """Reads ns5 file header and writes info to self.header"""
        # (Re-)initialize header
        self.header = HeaderInfo()
        
        # the width of each sample is always 2 bytes
        self.header.sample_width = 2
        
        # If filename specified, use it, else use previously specified
        if filename is not None: self.filename = filename
        self.header.filename = self.filename
        
        # first load the binary in directly
        self.file_handle = open(self.filename, 'rb') # buffering=?

        # Read File_Type_ID and check compatibility
        # If v2.2 is used, this value will be 'NEURALCD', which uses a slightly
        # more complex header. Currently unsupported.
        self.header.File_Type_ID = [chr(ord(c)) \
            for c in self.file_handle.read(8)]
        if "".join(self.header.File_Type_ID) != 'NEURALSG':
            logging.info( "Incompatible ns5 file format. Only v2.1 is supported.\nThis will probably not work.")
        
        
        # Read File_Spec and check compatibility.
        self.header.File_Spec = [chr(ord(c)) \
            for c in self.file_handle.read(16)]
        if "".join(self.header.File_Spec[:8]) != '30 kS/s\0':
            logging.info( "File_Spec seems to indicate you did not sample at 30KHz.")
        
        
        #R ead Period and verify that 30KHz was used. If not, the code will
        # still run but it's unlikely the data will be useful.
        self.header.period, = struct.unpack('<I', self.file_handle.read(4))
        if self.header.period != 1:
            logging.info( "Period seems to indicate you did not sample at 30KHz.")
        self.header.f_samp = self.header.period * 30000.0


        # Read Channel_Count and Channel_ID
        self.header.Channel_Count, = struct.unpack('<I',
            self.file_handle.read(4))
        self.header.Channel_ID = [struct.unpack('<I',
            self.file_handle.read(4))[0]
            for n in xrange(self.header.Channel_Count)]
        
        # Compute total header length
        self.header.Header = 8 + 16 + 4 + 4 + \
            4*self.header.Channel_Count # in bytes

        # determine length of file
        self.file_handle.seek(0, 2) # last byte
        self.header.file_total_size = self.file_handle.tell()
        self.header.n_samples = \
            (self.header.file_total_size - self.header.Header) / \
            self.header.Channel_Count / self.header.sample_width
        self.header.Length = np.float64(self.header.n_samples) / \
            self.header.Channel_Count
        if self.header.sample_width * self.header.Channel_Count * \
            self.header.n_samples + \
            self.header.Header != self.header.file_total_size:
            logging.info( "I got header of %dB, %d channels, %d samples, \
                but total file size of %dB" % (self.header.Header, 
                self.header.Channel_Count, self.header.n_samples, 
                self.header.file_total_size))

        # close file
        self.file_handle.close()

    
    def regenerate_memmap(self):
        """Delete internal memmap and create a new one, to save memory."""
        try:
            del self._mm
        except AttributeError: 
            pass
        
        self._mm = np.memmap(\
            self.filename, dtype='h', mode='r', 
            offset=self.header.Header, 
            shape=(self.header.n_samples, self.header.Channel_Count))
    
    def __del__(self):
        # this deletion doesn't free memory, even though del l._mm does!
        if '_mm' in self.__dict__: del self._mm
        #else: logging.info( "gracefully skipping")
    
    def _get_channel(self, channel_number):
        """Returns slice into internal memmap for requested channel"""
        try:
            mm_index = self.header.Channel_ID.index(channel_number)
        except ValueError:
            logging.info( "Channel number %d does not exist" % channel_number)
            return np.array([])
        
        self.regenerate_memmap()
        return self._mm[:, mm_index]
    
    def get_channel_as_array(self, channel_number):
        """Returns data from requested channel as a 1d numpy array."""
        data = np.array(self._get_channel(channel_number))
        self.regenerate_memmap()
        return data

    def get_analog_channel_as_array(self, analog_chn):
        """Returns data from requested analog channel as a numpy array.
        
        Simply adds 128 to the channel number to convert to ns5 number.
        This is just the way Cyberkinetics numbers its channels.
        """
        return self.get_channel_as_array(analog_chn + 128)    

    def get_audio_channel_numbers(self):
        """Deprecated, use get_analog_channel_ids"""
        return self.get_analog_channel_ids()
    
    def get_analog_channel_ids(self):
        """Returns array of analog channel ids existing in the file.
        
        These can then be loaded by calling get_analog_channel_as_array(chn).
        """
        return np.array(filter(lambda x: (x > 128) and (x <= 144), 
            self.header.Channel_ID)) - 128

    def get_neural_channel_numbers(self):
        return np.array(filter(lambda x: x <= 128, self.header.Channel_ID))
