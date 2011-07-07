# -*- coding: utf-8 -*-
"""Class for reading binary file from Blackrock format.
"""

from baseio import BaseIO
#import OpenElectrophy as OE
from ..core import *
import numpy as np
import struct
    



class BlackrockIO(BaseIO):
    """
    Class for reading/writing data in a BlackRock Neuroshare ns5 files.
   
    **Usage**

    **Example**
    
    """
    
    is_readable        = True
    is_writable        = False

    supported_objects  = [Block, Segment, AnalogSignal]
    readable_objects    = [ Segment]
    writeable_objects   = []
    
    # The file has a header, is that what this means?
    has_header         = True
    is_streameable     = False
    
    # GUI defaults for reading
    # Most information is acquired from the file header.
    # Blocks will start at time zero.
    # Segments can be read starting from any particular point in the file.
    read_params = { 
        Block : [
            ('rangemin' , { 'value' : -10 } ),
            ('rangemax' , { 'value' : 10 } ),
            ],
        Segment : [
            ('t_start', { 'value' : 0. } ),
            ('t_stop', { 'value' : 1. } ),
            ('rangemin', { 'value' : -10. } ),
            ('rangemax', { 'value' : 10. } ),
            ]
        }

    # GUI defaults for writing (not supported)
    write_params       = None
                        
    name               = 'Blackrock'
    extensions          = [ 'ns5' ]
    
    
    mode = 'file'
    
    def __init__(self , filename) :
        """This class read a binary Blackrock file.
        
        **Arguments**
            filename : the filename to read
        """
        BaseIO.__init__(self)
        self.filename = filename
        self.loader = Loader(filename)
        self.loader.load_file()
        self.header = self.loader.header
        
    
    def read(self , **kargs):
        """Read the file and return contents as a Block.

        See read_block for detail.
        """
        return self.read_block( **kargs)
    
    def read_block(self, full_range=2., 
        t_starts=[], t_stops=[], chlist=None):
        """Reads the file and returns contents as a Block.
        
        The Block will contain Segment sliced based on the times in
        t_starts and t_stops.
        
        Returns a Block object containing the data.
        """
        # Create block
        block = Block(fileOrigin=self.filename)
        
        # If channels not specified, get all
        if chlist is None:
            chlist = self.loader.get_neural_channel_numbers()

        # Iterate through t_starts and t_stops and add one Segment
        # per each.
        for n, (t1, t2) in enumerate(zip(t_starts, t_stops)):
            seg = self.read_segment(t_start=t1, 
                t_stop=t2, full_range=full_range, chlist=chlist)
            seg.name = 'Segment %d' % n
            seg.fileOrigin = self.filename
            block._segments.append(seg)
        
        return block
    
    def read_segment(self, t_start, t_stop, full_range=2., chlist=None):
        """Reads a Segment from the file and stores in database.
        
        The Segment will contain one AnalogSignal for each channel
        and will go from t_start to t_stop.
        
        Arguments:
            t_start : time in seconds that the Segment begins
            t_stop : time in seconds that the Segment ends
        
        Returns a Segment object containing the data.
        """
        conversion = np.float(full_range) / 2**16
        if chlist is None:
            chlist = self.loader.get_neural_channel_numbers()
        
        # Create the Segment
        seg = Segment(fileOrigin=self.filename)
        
        for ch in chlist:
            ts = t_start / self.header.f_samp
            sig = np.array(\
                self.loader._get_channel(ch)[t_start:t_stop]) * conversion

            anasig = AnalogSignal(signal=sig,
                sampling_rate=self.header.f_samp,
                t_start=ts,
                channel=ch)
            seg._analogsignals.append(anasig)
        
        return seg





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
            print "Incompatible ns5 file format. Only v2.1 is supported.\n" + \
                "This will probably not work."          
        
        
        # Read File_Spec and check compatibility.
        self.header.File_Spec = [chr(ord(c)) \
            for c in self.file_handle.read(16)]
        if "".join(self.header.File_Spec[:8]) != '30 kS/s\0':
            print "File_Spec seems to indicate you did not sample at 30KHz."
        
        
        #R ead Period and verify that 30KHz was used. If not, the code will
        # still run but it's unlikely the data will be useful.
        self.header.period, = struct.unpack('<I', self.file_handle.read(4))
        if self.header.period != 1:
            print "Period seems to indicate you did not sample at 30KHz."
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
            print "I got header of %dB, %d channels, %d samples, \
                but total file size of %dB" % (self.header.Header, 
                self.header.Channel_Count, self.header.n_samples, 
                self.header.file_total_size)

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
        #else: print "gracefully skipping"
    
    def _get_channel(self, channel_number):
        """Returns slice into internal memmap for requested channel"""
        try:
            mm_index = self.header.Channel_ID.index(channel_number)
        except ValueError:
            print "Channel number %d does not exist" % channel_number
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
