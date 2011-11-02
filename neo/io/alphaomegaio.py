# encoding: utf-8
"""

Class for reading data from Alpha Omega .map files.

This class is an experimental reader with important limitations.
See the source code for details of the limitations.
The code of this reader is of alpha quality and received very limited testing.    

This code is written from the incomplete file specifications available in:

[1] AlphaMap Data Acquisition System User's Manual Version 10.1.1
Section 5 APPENDIX B: ALPHAMAP FILE STRUCTURE, pages 120-140
Edited by ALPHA OMEGA Home Office: P.O. Box 810, Nazareth Illit 17105, Israel
http://www.alphaomega-eng.com/

and from the source code of a C software for conversion of .map files to 
.eeg elan software files :

[2] alphamap2eeg 1.0, 12/03/03, Anne CHEYLUS - CNRS ISC UMR 5015

Supported : Read

@author : sgarcia, Florent Jaillet

"""

# NOTE: For some specific types of comments, the following convention is used:
# "TODO:" Desirable future evolution
# "WARNING:" Information about code that is based on broken or missing 
# specifications and that might be wrong 


# Main limitations of this reader: 
# - The reader is only able to load data stored in data blocks of type 5 
#   (data block for one channel). In particular it means that it doesn't  
#   support signals stored in blocks of type 7 (data block for multiple 
#   channels).
#   For more details on these data blocks types, see 5.4.1 and 5.4.2 p 127 in 
#   [1].
# - Rather than supporting all the neo objects types that could be extracted 
#   from the file, all read data are returned in AnalogSignal objects, even for 
#   digital channels or channels containing spiking informations.
# - Digital channels are not converted to events or events array as they
#   should.
# - Loading multichannel signals as AnalogSignalArrays is not supported. 
# - Many data or metadata that are avalaible in the file and that could be 
#   represented in some way in the neo model are not extracted. In particular 
#   scaling of the data and extraction of the units of the signals are not 
#   supported.
# - It received very limited testing, exlusively using python 2.6.6. In 
#   particular it has not been tested using Python 3.x.
#
# These limitations are mainly due to the following reasons:
# - Incomplete, unclear and in some places innacurate specifications of the 
#   format in [1].
# - Lack of test files containing all the types of data blocks of interest 
#   (in particular no file with type 7 data block for multiple channels where 
#   available when writing this code).
# - Lack of knowledge of the Alphamap software and the associated data models.
# - Lack of time (especially as the specifications are incomplete, a lot of 
#   reverse engineering and testing is required, which makes the development of 
#   this IO very painful and long).


from __future__ import absolute_import, division
from .baseio import BaseIO
from ..core import Block, Segment, AnalogSignal
from .tools import create_many_to_one_relationship, populate_RecordingChannel

# note neo.core need only numpy and quantities
import numpy as np
import quantities as pq

# specific imports
import struct, os
import datetime

# file no longer exists in Python3
try:
    file
except NameError:
    import io
    file = io.BufferedReader

class AlphaOmegaIO(BaseIO):
    """
    Class for reading data from Alpha Omega .map files (experimental)

    This class is an experimental reader with important limitations.
    See the source code for details of the limitations.
    The code of this reader is of alpha quality and received very limited 
    testing.    
    
    Usage:
        >>> from neo import io
        >>> r = io.AlphaOmegaIO( filename = 'File_AlphaOmega_1.map')
        >>> blck = r.read_block(lazy = False, cascade = True)
        >>> print blck.segments[0].analogsignals

    """
    
    is_readable        = True  # This is a reading only class
    is_writable        = False # writting is not supported
    
    # This class is able to directly or inderectly read the following kind of 
    # objects
    supported_objects  = [ Block, Segment , AnalogSignal] 
    # TODO: Add support for other objects that should be extractable from .map
    # files (AnalogSignalArray, Event, EventArray, Epoch?, Epoch Array?, 
    # Spike?, SpikeTrain?)
    
    # This class can only return a Block
    readable_objects   = [ Block ]
    # TODO : create readers for different type of objects (Segment, 
    # AnalogSignal,...)
        
    # This class is not able to write objects
    writeable_objects  = [ ]
    
    # This is for GUI stuff : a definition for parameters when reading. 
    read_params        = { Block : [ ] }
    
    # Writing is not supported, so no GUI stuff
    write_params       = None
    
    name               = 'AlphaOmega'
    extensions         = [ 'map' ]
    
    mode = 'file'
    
    
    def __init__(self , filename = None) :
        """
        
        Arguments:
            filename : the .map Alpha Omega file name

        """
        BaseIO.__init__(self)
        self.filename = filename


    def read(self , **kargs):
        """
        Read a file.
        Return a neo.Block
        See read_block for more details.
         
        """
        # the higher level of my IO is Block so:
        return self.read_block( **kargs)

    
    # write is not supported so I do not overload write method from BaseIO

    def read_block(self, 
                   # the 2 first keyword arguments are imposed by neo.io API
                   lazy = False,
                   cascade = True):
        """
        Return a Block.
        
        """

        def count_samples(m_length):
            """
            Count the number of signal samples available in a type 5 data block 
            of length m_length
            
            """
            
            # for information about type 5 data block, see [1]
            count = int((m_length-6)/2-2)
            # -6 corresponds to the header of block 5, and the -2 take into 
            # account the fact that last 2 values are not available as the 4 
            # corresponding bytes are coding the time stamp of the beginning 
            # of the block
            return count

        # create the neo Block that will be returned at the end  
        blck = Block(file_origin = os.path.basename(self.filename))
        blck.file_origin = os.path.basename(self.filename)
        
        fid = open(self.filename, 'rb')

        # NOTE: in the following, the word "block" is used in the sense used in 
        # the alpha-omega specifications (ie a data chunk in the file), rather 
        # than in the sense of the usual Block object in neo

        # step 1: read the headers of all the data blocks to load the file 
        # structure

        pos_block = 0 # position of the current block in the file        
        file_blocks = [] # list of data blocks available in the file        
        
        if not cascade:
            # we read only the main header          
            
            m_length, m_TypeBlock = struct.unpack('Hcx' , fid.read(4))
            # m_TypeBlock should be 'h', as we read the first block
            block = HeaderReader(fid, 
                                 dict_header_type.get(m_TypeBlock,
                                                      Type_Unknown)).read_f()
            block.update({'m_length': m_length,
                          'm_TypeBlock': m_TypeBlock, 
                          'pos': pos_block})
            file_blocks.append(block)
            
        else: # cascade == True 
        
            seg = Segment(file_origin = os.path.basename(self.filename))
            seg.file_origin = os.path.basename(self.filename)
            blck.segments.append(seg)
            
            while True:
                first_4_bytes = fid.read(4)
                if len(first_4_bytes) < 4:
                    # we have reached the end of the file
                    break
                else:
                    m_length, m_TypeBlock = struct.unpack('Hcx', first_4_bytes)
 
                block = HeaderReader(fid, 
                                dict_header_type.get(m_TypeBlock, 
                                                     Type_Unknown)).read_f()
                block.update({'m_length': m_length,
                              'm_TypeBlock': m_TypeBlock,
                              'pos': pos_block})
                
                if m_TypeBlock == '2':
                    # The beggining of the block of type '2' is identical for 
                    # all types of channels, but the following part depends on 
                    # the type of channel. So we need a special case here.
                    
                    # WARNING: How to check the type of channel is not 
                    # described in the documentation. So here I use what is 
                    # proposed in the C code [2].
                    # According to this C code, it seems that the 'm_isAnalog' 
                    # is used to distinguished analog and digital channels, and
                    # 'm_Mode' encodes the type of analog channel:
                    # 0 for continuous, 1 for level, 2 for external trigger.
                    # But in some files, I found channels that seemed to be 
                    # continuous channels with 'm_Modes' = 128 or 192. So I 
                    # decided to consider every channel with 'm_Modes' 
                    # different from 1 or 2 as continuous. I also couldn't
                    # check that values of 1 and 2 are really for level and
                    # external trigger as I had no test files containing data 
                    # of this types.
                    
                    type_subblock = 'unknown_channel_type(m_Mode=' \
                                    + str(block['m_Mode'])+ ')'
                    description = Type2_SubBlockUnknownChannels
                    block.update({'m_Name': 'unknown_name'})
                    if block['m_isAnalog'] == 0:
                        # digital channel
                        type_subblock = 'digital'
                        description = Type2_SubBlockDigitalChannels
                    elif block['m_isAnalog'] == 1:
                        # analog channel
                        if block['m_Mode'] == 1:
                            # level channel
                            type_subblock = 'level'
                            description = Type2_SubBlockLevelChannels
                        elif block['m_Mode'] == 2:
                            # external trigger channel
                            type_subblock = 'external_trigger'
                            description = Type2_SubBlockExtTriggerChannels                
                        else: 
                            # continuous channel
                            type_subblock = 'continuous(Mode' \
                                            + str(block['m_Mode']) +')'
                            description = Type2_SubBlockContinuousChannels 
                        
                    subblock = HeaderReader(fid, description).read_f()
                    
                    block.update(subblock)
                    block.update({'type_subblock': type_subblock})

                file_blocks.append(block)
                pos_block += m_length
                fid.seek(pos_block)
            
            # step 2: find the available channels                
            list_chan = [] # list containing indexes of channel blocks
            for ind_block, block in enumerate(file_blocks):
                if block['m_TypeBlock'] == '2':
                    list_chan.append(ind_block)
            
            # step 3: find blocks containing data for the available channels
            list_data = [] # list of lists of indexes of data blocks 
                           # corresponding to each channel
            for ind_chan, chan in enumerate(list_chan):
                list_data.append([])
                num_chan = file_blocks[chan]['m_numChannel']
                for ind_block, block in enumerate(file_blocks):
                    if block['m_TypeBlock'] == '5':
                        if block['m_numChannel'] == num_chan:
                            list_data[ind_chan].append(ind_block)
     
            
            # step 4: compute the length (number of samples) of the channels
            chan_len = np.zeros(len(list_data), dtype = np.int)       
            for ind_chan, list_blocks in enumerate(list_data):
                for ind_block in list_blocks:
                    chan_len[ind_chan] += count_samples(
                                          file_blocks[ind_block]['m_length'])
            
            # step 5: find channels for which data are available
            ind_valid_chan = np.nonzero(chan_len)[0]
                   
            # step 6: load the data
            # TODO give the possibility to load data as AnalogSignalArrays
            for ind_chan in ind_valid_chan:
                list_blocks = list_data[ind_chan]            
                ind = 0 # index in the data vector
                
                # read time stamp for the beginning of the signal
                form = '<l' # reading format
                ind_block = list_blocks[0]
                count = count_samples(file_blocks[ind_block]['m_length'])
                fid.seek(file_blocks[ind_block]['pos']+6+count*2)
                buf = fid.read(struct.calcsize(form))
                val = struct.unpack(form , buf)
                start_index = val[0]
                                
                # WARNING: in the following blocks are read supposing taht they
                # are all contiguous and sorted in time. I don't know if it's 
                # always the case. Maybe we should use the time stamp of each 
                # data block to choose where to put the read data in the array.
                if not lazy:
                    temp_array = np.empty(chan_len[ind_chan], dtype = np.int16)
                    # NOTE: we could directly create an empty AnalogSignal and 
                    # load the data in it, but it is much faster to load data 
                    # in a temporary numpy array and create the AnalogSignals 
                    # from this temporary array
                    for ind_block in list_blocks:
                        count = count_samples(
                                file_blocks[ind_block]['m_length'])
                        fid.seek(file_blocks[ind_block]['pos']+6)          
                        temp_array[ind:ind+count] = \
                            np.fromfile(fid, dtype = np.int16, count = count)                           
                        ind += count            
                
                sampling_rate = \
                    file_blocks[list_chan[ind_chan]]['m_SampleRate'] * pq.kHz
                t_start = (start_index / sampling_rate).simplified
                if lazy:
                    ana_sig = AnalogSignal([],
                                           sampling_rate = sampling_rate,
                                           t_start = t_start,
                                           name = file_blocks\
                                               [list_chan[ind_chan]]['m_Name'],
                                           file_origin = \
                                               os.path.basename(self.filename),
                                           units = pq.dimensionless)
                    ana_sig.lazy_shape = chan_len[ind_chan]
                else:
                    ana_sig = AnalogSignal(temp_array, 
                                           sampling_rate = sampling_rate,
                                           t_start = t_start,
                                           name = file_blocks\
                                               [list_chan[ind_chan]]['m_Name'],
                                           file_origin = \
                                               os.path.basename(self.filename),
                                           units = pq.dimensionless)
                
                ana_sig.annotate(channel_index = \
                            file_blocks[list_chan[ind_chan]]['m_numChannel'])
                ana_sig.annotate(channel_name = \
                            file_blocks[list_chan[ind_chan]]['m_Name'])
                ana_sig.annotate(channel_type = \
                            file_blocks[list_chan[ind_chan]]['type_subblock'])
                seg.analogsignals.append(ana_sig)

        fid.close()        
        
        if file_blocks[0]['m_TypeBlock'] == 'h': # this should always be true
            blck.rec_datetime = datetime.datetime(\
                file_blocks[0]['m_date_year'], 
                file_blocks[0]['m_date_month'],
                file_blocks[0]['m_date_day'],
                file_blocks[0]['m_time_hour'],
                file_blocks[0]['m_time_minute'],
                file_blocks[0]['m_time_second'],
                10000 * file_blocks[0]['m_time_hsecond'])
                # the 10000 is here to convert m_time_hsecond from centisecond
                # to microsecond
            version = file_blocks[0]['m_version']                
            blck.annotate(alphamap_version = version)
            if cascade:
                seg.rec_datetime = blck.rec_datetime.replace()
                # I couldn't find a simple copy function for datetime, 
                # using replace without arguments is a twisted way to make a
                # copy
                seg.annotate(alphamap_version = version)
        if cascade:
            populate_RecordingChannel(blck, remove_from_annotation = True)
            create_many_to_one_relationship(blck)             
        
        return blck
    


"""
Information for special types in [1]:

_dostime_t type definition:
struct dos_time_t
{
 unsigned char hour; /* hours (0-23)*/
 unsigned char minute; /* minutes (0-59)*/
 unsigned char second; /* seconds (0-59) */
 unsigned char hsecond; /* seconds/ 100 (0-99)*/
}

_dosdate_t type definition:
struct _dosdate_t
{
 unsigned char day;       /* day of month( 1-31) */
 unsigned char month;     /* month (1-12) */
 unsigned int year;       /* year (1980-2099) */
 unsigned char dayofweek; /* day of week (0 = Sunday) */
}

WINDOWPLACEMENT16 type definition (according to WINE source code):
typedef struct
{
    UINT16   length;
    UINT16   flags;
    UINT16   showCmd;
    POINT16  ptMinPosition;
    POINT16  ptMaxPosition;
    RECT16   rcNormalPosition;
} WINDOWPLACEMENT16,*LPNONCLIENTMETRICS16;

"""

max_string_len = '32s' # maximal length of variable length strings in the file
# WARNING: I don't know what is the real value here. According to [1] p 139 
# it seems that it could be 20. Some tests would be needed to check this.

# WARNING: A cleaner way to handle strings reading is suitable. Currently I 
# read a buffer of max_string_len bytes and look for the C "end of string" 
# character ('\x00'). It would be better either to read characters until 
# reaching '\x00' or to read the exact number of characters needed, if the 
# length of a string can be deduced from the lentgh of the block and the number
# of bytes already read (it seems possible, at least for certain block types). 

# WARNING: Some test files contains data blocks of type 'b' and they are not 
# described in the documentation.

# The name of the keys in the folowing dicts are chosen to match as closely as 
# possible the names in document [1]

TypeH_Header = [
    ('m_nextBlock','l'),
    ('m_version','h'),
    ('m_time_hour', 'B'),
    ('m_time_minute', 'B'),
    ('m_time_second', 'B'),
    ('m_time_hsecond', 'B'),
    ('m_date_day', 'B'),
    ('m_date_month', 'B'),
    ('m_date_year', 'H'),
    ('m_date_dayofweek', 'B'),
    ('blank', 'x'), # one byte blank because of the 2 bytes alignement
    ('m_MinimumTime','d'),
    ('m_MaximumTime','d')]

Type0_SetBoards = [
    ('m_nextBlock','l'),
    ('m_BoardCount','h'),
    ('m_GroupCount','h'),
    ('m_placeMainWindow','x')] # WARNING: unknown type ('x' is wrong)

Type1_Boards = [ # WARNING: needs to be checked
    ('m_nextBlock','l'),
    ('m_Number','h'),
    ('m_countChannel','h'),
    ('m_countAnIn','h'),
    ('m_countAnOut','h'),
    ('m_countDigIn','h'),
    ('m_countDigOut','h'),
    ('m_TrigCount', 'h'), # not defined in 5.3.3 but appears in 5.5.1 and 
                          # seems to really exist in files
    # WARNING: check why 'm_TrigCount is not in the C code [2]
    ('m_Amplitude','f'),
    ('m_cSampleRate','f'), # sample rate seems to be given in kHz
    ('m_Duration','f'),
    ('m_nPreTrigmSec','f'),
    ('m_nPostTrigmSec','f'),
    ('m_TrgMode','h'),
    ('m_LevelValue','h'), # after this line, 5.3.3 is wrong, 
                          # check example in 5.5.1 for the right fields
    # WARNING: check why the following part is not corrected in the C code [2]               
    ('m_nSamples','h'),
    ('m_fRMS','f'),
    ('m_ScaleFactor','f'),
    ('m_DapTime','f'),
    ('m_nameBoard', max_string_len)]
    #('m_DiscMaxValue','h'), # WARNING: should this exist?
    #('m_DiscMinValue','h') # WARNING: should this exist?

Type2_DefBlocksChannels = [
    # common parameters for all types of channels
    ('m_nextBlock','l'),
    ('m_isAnalog','h'),
    ('m_isInput','h'),
    ('m_numChannel','h'),
    ('m_numColor','h'),
    ('m_Mode','h')]
            
Type2_SubBlockContinuousChannels = [
    # continuous channels parameters
    ('blank', '2x'), # WARNING: this is not in the specs but it seems needed
    ('m_Amplitude','f'),
    ('m_SampleRate','f'),
    ('m_ContBlkSize','h'),
    ('m_ModeSpike','h'), # WARNING: the C code [2] uses usigned short here
    ('m_Duration','f'),
    ('m_bAutoScale','h'),
    ('m_Name', max_string_len)]

Type2_SubBlockLevelChannels = [ # WARNING: untested
    # level channels parameters
    ('m_Amplitude','f'),
    ('m_SampleRate','f'),
    ('m_nSpikeCount','h'),
    ('m_ModeSpike','h'),
    ('m_nPreTrigmSec','f'),
    ('m_nPostTrigmSec','f'),
    ('m_LevelValue','h'),
    ('m_TrgMode','h'),
    ('m_YesRms','h'),
    ('m_bAutoScale','h'),
    ('m_Name', max_string_len)]            
            
Type2_SubBlockExtTriggerChannels = [ # WARNING: untested
    # external trigger channels parameters
    ('m_Amplitude','f'),
    ('m_SampleRate','f'),
    ('m_nSpikeCount','h'),
    ('m_ModeSpike','h'),
    ('m_nPreTrigmSec','f'),
    ('m_nPostTrigmSec','f'),
    ('m_TriggerNumber','h'),
    ('m_Name', max_string_len)]

Type2_SubBlockDigitalChannels = [
    # digital channels parameters
    ('m_SampleRate','f'),
    ('m_SaveTrigger','h'),
    ('m_Duration','f'),
    ('m_PreviousStatus','h'), # WARNING: check difference with C code here
    ('m_Name', max_string_len)]
            
Type2_SubBlockUnknownChannels = [
    # WARNING: We have a mode that doesn't appear in our spec, so we don't 
    # know what are the fields.
    # It seems that for non-digital channels the beginning is 
    # similar to continuous channels. Let's hope we're right...
    ('blank', '2x'),
    ('m_Amplitude','f'),
    ('m_SampleRate','f')]
    # there are probably other fields after...
                
Type6_DefBlockTrigger = [ # WARNING: untested
    ('m_nextBlock','l'),
    ('m_Number','h'),
    ('m_countChannel','h'),
    ('m_StateChannels','i'),
    ('m_numChannel1','h'),
    ('m_numChannel2','h'),
    ('m_numChannel3','h'),
    ('m_numChannel4','h'),
    ('m_numChannel5','h'),
    ('m_numChannel6','h'),
    ('m_numChannel7','h'),
    ('m_numChannel8','h'),
    ('m_Name','c')]
            
Type3_DefBlockGroup = [ # WARNING: untested
    ('m_nextBlock','l'),
    ('m_Number','h'),
    ('m_Z_Order','h'),
    ('m_countSubGroups','h'),
    ('m_placeGroupWindow','x'), # WARNING: unknown type ('x' is wrong)
    ('m_NetLoc','h'),
    ('m_locatMax','x'), # WARNING: unknown type ('x' is wrong)
    ('m_nameGroup','c')]
            
Type4_DefBlockSubgroup = [ # WARNING: untested
    ('m_nextBlock','l'),
    ('m_Number','h'),
    ('m_TypeOverlap','h'),
    ('m_Z_Order','h'),
    ('m_countChannel','h'),
    ('m_NetLoc','h'),
    ('m_location','x'), # WARNING: unknown type ('x' is wrong)
    ('m_bIsMaximized','h'),
    ('m_numChannel1','h'),
    ('m_numChannel2','h'),
    ('m_numChannel3','h'),
    ('m_numChannel4','h'),
    ('m_numChannel5','h'),
    ('m_numChannel6','h'),
    ('m_numChannel7','h'),
    ('m_numChannel8','h'),
    ('m_Name','c')]
            
Type5_DataBlockOneChannel = [
    ('m_numChannel','h')] 
    # WARNING: 'm_numChannel' (called 'm_Number' in 5.4.1 of [1]) is supposed 
    # to be uint according to 5.4.1 but it seems to be a short in the files 
    # (or should it be ushort ?)
            
# WARNING: In 5.1.1 page 121 of [1], they say "Note: 5 is used for demo 
# purposes, 7 is used for real data", but looking at some real datafiles, 
# it seems that block of type 5 are also used for real data...
            
Type7_DataBlockMultipleChannels = [ # WARNING: unfinished
    ('m_lenHead', 'h'), # WARNING: unknown true type
    ('FINT','h')]
    # WARNING: there should be data after...
         
TypeP_DefBlockPeriStimHist = [ # WARNING: untested
    ('m_Number_Chan','h'),
    ('m_Position','x'), # WARNING: unknown type ('x' is wrong)
    ('m_isStatVisible','h'),
    ('m_DurationSec','f'),
    ('m_Rows','i'),
    ('m_DurationSecPre','f'),
    ('m_Bins','i'),
    ('m_NoTrigger','h')]
         
TypeF_DefBlockFRTachogram = [ # WARNING: untested
    ('m_Number_Chan','h'),
    ('m_Position','x'), # WARNING: unknown type ('x' is wrong)
    ('m_isStatVisible','h'),
    ('m_DurationSec','f'),
    ('m_AutoManualScale','i'),
    ('m_Max','i')]
            
TypeR_DefBlockRaster = [ # WARNING: untested
    ('m_Number_Chan','h'),
    ('m_Position','x'), # WARNING: unknown type ('x' is wrong)
    ('m_isStatVisible','h'),
    ('m_DurationSec','f'),
    ('m_Rows','i'),
    ('m_NoTrigger','h')]
            
TypeI_DefBlockISIHist = [ # WARNING: untested
    ('m_Number_Chan','h'),
    ('m_Position','x'), # WARNING: unknown type ('x' is wrong)
    ('m_isStatVisible','h'),
    ('m_DurationSec','f'),
    ('m_Bins','i'),
    ('m_TypeScale','i')]
            
Type8_MarkerBlock = [ # WARNING: untested
    ('m_Number_Channel','h'),
    ('m_Time','l')] # WARNING: check what's the right type here.
    # It seems that the size of time_t type depends on the system typedef, 
    # I put long here but I couldn't check if it is the right type
            
Type9_ScaleBlock = [ # WARNING: untested
    ('m_Number_Channel','h'),
    ('m_Scale','f')]

Type_Unknown = []

dict_header_type = {
                    'h' : TypeH_Header,
                    '0' : Type0_SetBoards,
                    '1' : Type1_Boards,
                    '2' : Type2_DefBlocksChannels,
                    '6' : Type6_DefBlockTrigger,
                    '3' : Type3_DefBlockGroup,
                    '4' : Type4_DefBlockSubgroup,
                    '5' : Type5_DataBlockOneChannel,
                    '7' : Type7_DataBlockMultipleChannels,
                    'P' : TypeP_DefBlockPeriStimHist,
                    'F' : TypeF_DefBlockFRTachogram,
                    'R' : TypeR_DefBlockRaster,
                    'I' : TypeI_DefBlockISIHist,
                    '8' : Type8_MarkerBlock,
                    '9' : Type9_ScaleBlock
                    }


class HeaderReader():
    def __init__(self,fid ,description ):
        self.fid = fid
        self.description = description
    def read_f(self, offset =None):
        if offset is not None :
            self.fid.seek(offset)
        d = { }
        for key, format in self.description :                            
            format = '<' + format # insures use of standard sizes
            buf = self.fid.read(struct.calcsize(format))
            if len(buf) != struct.calcsize(format) : return None
            val = struct.unpack(format , buf)
            if len(val) == 1:
                val = val[0]
            else :
                val = list(val)
            if 's' in format :
                val = val.split('\x00',1)[0]
            d[key] = val
        return d

    
