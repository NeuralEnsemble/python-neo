# encoding: utf-8
"""
NeuroshareIO is a wrap with ctypes of neuroshare DLLs.
Neuroshare is a C API for reading neural data.
Neuroshare also provides a Matlab and a Python API on top of that.

Neuroshare is an open source API but each dll is provided directly by the vendor.
The neo user have to download separtatly the dll on neurosharewebsite:
http://neuroshare.sourceforge.net/

For some vendors (Spike2/CED , Clampfit/Abf, ...), neo.io also provides pure Python
Neo users you should prefer them of course :)

Supported : Read

Author: sgarcia

"""

from .baseio import BaseIO
from ..core import *
from .tools import create_many_to_one_relationship

import numpy as np
import quantities as pq


import os
import ctypes
from ctypes import byref, c_char_p, c_uint32, c_char, c_double, c_int16, c_int32 , c_ulong



# file no longer exists in Python3
try:
    file
except NameError:
    import io
    file = io.BufferedReader


class NeuroshareIO(BaseIO):
    """
    Class for reading file trougth neuroshare API.
    The user need the DLLs in the path of the file format.
    
    Usage:
        >>> from neo import io
        >>> r = io.NeuroshareIO(filename='a_file', dllname=the_name_of_dll)
        >>> seg = r.read_segment(lazy=False, cascade=True, import_neuroshare_segment=True)
        >>> print seg.analogsignals        # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<AnalogSignal(array([ -1.77246094e+02,  -2.24707031e+02,  -2.66015625e+02,
        ...
        >>> print seg.spiketrains
        []
        >>> print seg.eventarrays
        [<EventArray: 1@1.12890625 s, 1@2.02734375 s, 1@3.82421875 s>]
        
    Note:
        neuroshare.ns_ENTITY_EVENT: are converted to neo.EventArray
        neuroshare.ns_ENTITY_ANALOG: are converted to neo.AnalogSignal
        neuroshare.ns_ENTITY_NEURALEVENT: are converted to neo.SpikeTrain
        
        neuroshare.ns_ENTITY_SEGMENT: is something between serie of small AnalogSignal 
                                        and Spiketrain with associated waveforms.
                                        It is arbitrarily converted as SpikeTrain.
    

    """
    
    is_readable        = True
    is_writable        = False

    supported_objects            = [Segment , AnalogSignal, EventArray, SpikeTrain ]
    readable_objects    = [Segment]
    writeable_objects    = [ ]
    
    has_header         = False
    is_streameable     = False
    
    read_params        = { Segment : [] }
    write_params       = None
    
    name               = 'neuroshare'
    extensions          = [  ]    
    mode = 'file'
    
    
    
    def __init__(self , filename = '', dllname = '') :
        """
        Arguments:
            filename: the file to read
            ddlname: the name of neuroshare dll to be used for this file
        """
        self.dllname = dllname
        self.filename = filename
        BaseIO.__init__(self)



    
    def read_segment(self, import_neuroshare_segment = True):
        """
        Arguments:
            import_neuroshare_segment: import neuroshare segment as SpikeTrain with associated waveforms or not imported at all.
            
        """
        
        seg = Segment( file_origin = os.path.basename(self.filename), )

        neuroshare = ctypes.windll.LoadLibrary(self.dllname)
        
        # API version
        info = ns_LIBRARYINFO()
        neuroshare.ns_GetLibraryInfo(byref(info) , ctypes.sizeof(info))
        seg.annotate(neuroshare_version = str(info.dwAPIVersionMaj)+'.'+str(info.dwAPIVersionMin))
        
        if not cascade:
            return seg
        
        
        # open file
        hFile = c_uint32(0)
        neuroshare.ns_OpenFile(c_char_p(self.filename) ,byref(hFile))
        fileinfo = ns_FILEINFO()
        neuroshare.ns_GetFileInfo(hFile, byref(fileinfo) , ctypes.sizeof(fileinfo))
        
        # read all entities
        for dwEntityID in range(fileinfo.dwEntityCount):
            entityInfo = ns_ENTITYINFO()
            neuroshare.ns_GetEntityInfo( hFile, dwEntityID, byref(entityInfo), ctypes.sizeof(entityInfo))
            #~ print 'type', entityInfo.dwEntityType,entity_types[entityInfo.dwEntityType], 'count', entityInfo.dwItemCount
            #~ print  entityInfo.szEntityLabel 

            # EVENT
            if entity_types[entityInfo.dwEntityType] == 'ns_ENTITY_EVENT': 
                pEventInfo = ns_EVENTINFO()
                neuroshare.ns_GetEventInfo ( hFile,  dwEntityID,  byref(pEventInfo), ctypes.sizeof(pEventInfo))
                #~ print pEventInfo.szCSVDesc, pEventInfo.dwEventType, pEventInfo.dwMinDataLength, pEventInfo.dwMaxDataLength
                
                if pEventInfo.dwEventType == 0: #TEXT
                    pData = ctypes.create_string_buffer(pEventInfo.dwMaxDataLength)
                elif pEventInfo.dwEventType == 1:#CVS
                    pData = ctypes.create_string_buffer(pEventInfo.dwMaxDataLength)
                elif pEventInfo.dwEventType == 2:# 8bit
                    pData = ctypes.c_byte(0)
                elif pEventInfo.dwEventType == 3:# 16bit
                    pData = ctypes.c_int16(0)
                elif pEventInfo.dwEventType == 4:# 32bit
                    pData = ctypes.c_int32(0)
                pdTimeStamp  = c_double(0.)
                pdwDataRetSize = c_uint32(0)
                
                ea = EventArray(name = str(entityInfo.szEntityLabel),)
                if not lazy:
                    times = [ ]
                    labels = [ ]
                    for dwIndex in range(entityInfo.dwItemCount ):
                        neuroshare.ns_GetEventData ( hFile, dwEntityID, dwIndex,
                                            byref(pdTimeStamp), byref(pData),
                                            ctypes.sizeof(pData), byref(pdwDataRetSize) )
                        times.append(pdTimeStamp.value)
                        labels.append(str(pData))
                    ea.times = times*pq.s
                    ea.labels = np.array(labels, dtype ='S')
                else :
                    ea.lazy_shape = entityInfo.dwItemCount
                seg.eventarrays.append(ea)
            
            # analog
            if entity_types[entityInfo.dwEntityType] == 'ns_ENTITY_ANALOG': 
                pAnalogInfo = ns_ANALOGINFO()
                
                neuroshare.ns_GetAnalogInfo( hFile, dwEntityID,byref(pAnalogInfo),ctypes.sizeof(pAnalogInfo) )
                #~ print 'dSampleRate' , pAnalogInfo.dSampleRate , pAnalogInfo.szUnits
                dwStartIndex = c_uint32(0)
                dwIndexCount = entityInfo.dwItemCount
                
                if lazy:
                    signal = [ ]*pq.Quantity(1, pAnalogInfo.szUnits) 
                else:
                    pdwContCount = c_uint32(0)
                    pData = zeros( (entityInfo.dwItemCount,), dtype = 'f8')
                    ns_RESULT = neuroshare.ns_GetAnalogData ( hFile,  dwEntityID,  dwStartIndex,
                                     dwIndexCount, byref( pdwContCount) , pData.ctypes.data_as(ctypes.POINTER(c_double)))
                    pszMsgBuffer  = ctypes.create_string_buffer(" "*256)
                    neuroshare.ns_GetLastErrorMsg(byref(pszMsgBuffer), 256)
                    #~ print 'pszMsgBuffer' , pszMsgBuffer.value
                    signal = pData[:pdwContCount.value]*pq.Quantity(1, pAnalogInfo.szUnits) 
                
                #t_start
                dwIndex = 0
                pdTime = c_double(0)
                neuroshare.ns_GetTimeByIndex( hFile,  dwEntityID,  dwIndex, byref(pdTime))
                
                anaSig = AnalogSignal(signal,
                                                    sampling_rate = pAnalogInfo.dSampleRate*pq.Hz,
                                                    t_start = pdTime.value * pq.s, 
                                                    name = str(entityInfo.szEntityLabel),
                                                    )
                if lazy:
                    anaSig.lazy_shape = entityInfo.dwItemCount
                seg.analogsignals.append( anaSig )
                
            
            #segment
            if entity_types[entityInfo.dwEntityType] == 'ns_ENTITY_SEGMENT' and import_neuroshare_segment: 
                
                pdwSegmentInfo = ns_SEGMENTINFO()
                
                ns_RESULT = neuroshare.ns_GetSegmentInfo( hFile,  dwEntityID,
                                                 byref(pdwSegmentInfo), ctypes.sizeof(pdwSegmentInfo) )
                nsource = pdwSegmentInfo.dwSourceCount
                
                pszMsgBuffer  = ctypes.create_string_buffer(" "*256)
                neuroshare.ns_GetLastErrorMsg(byref(pszMsgBuffer), 256)
                #~ print 'pszMsgBuffer' , pszMsgBuffer.value
                
                #~ print 'pdwSegmentInfo.dwSourceCount' , pdwSegmentInfo.dwSourceCount
                for dwSourceID in range(pdwSegmentInfo.dwSourceCount) :
                    pSourceInfo = ns_SEGSOURCEINFO()
                    neuroshare.ns_GetSegmentSourceInfo( hFile,  dwEntityID, dwSourceID,
                                    byref(pSourceInfo), ctypes.sizeof(pSourceInfo) )
                
                if lazy:
                    sptr = SpikeTrain(times, name = str(entityInfo.szEntityLabel))
                else:
                    pdTimeStamp  = c_double(0.)
                    dwDataBufferSize = pdwSegmentInfo.dwMaxSampleCount*pdwSegmentInfo.dwSourceCount
                    pData = zeros( (dwDataBufferSize), dtype = 'f8')
                    pdwSampleCount = c_uint32(0)
                    pdwUnitID= c_uint32(0)
                    
                    times = empty( (entityInfo.dwItemCount), drtype = 'f')
                    waveforms = empty( (entityInfo.dwItemCount, nsource, nsample), drtype = 'f')
                    for dwIndex in range(entityInfo.dwItemCount ):
                        ns_RESULT = neuroshare.ns_GetSegmentData ( hFile,  dwEntityID,  dwIndex,
                            byref(pdTimeStamp), pData.ctypes.data_as(ctypes.POINTER(c_double)),
                            dwDataBufferSize * 8, byref(pdwSampleCount),
                                byref(pdwUnitID ) )
                        nsample  = pdwSampleCount.value
                        #print 'dwDataBufferSize' , dwDataBufferSize,pdwSampleCount , pdwUnitID
                        
                        times[dwIndex] = pdTimeStamp.value
                        waveforms[dwIndex, :,:] = pData[:nsample*nsource].reshape(nsample ,nsource).transpose()
                    
                    sptr = SpikeTrain(times*pq.s,
                                        waveforms = waveforms*pq.Quantity(1., str(pdwSegmentInfo.szUnits) ),
                                        left_sweep = nsample/2./float(pdwSegmentInfo.dSampleRate)*pq.s,
                                        sampling_rate = float(pdwSegmentInfo.dSampleRate)*pq.Hz,
                                        name = str(entityInfo.szEntityLabel),
                                        )
                    if lazy:
                        sptr.lazy_shape = entityInfo.dwItemCount
                    seg.spiketrains.append(sptr)
            
            
            # neuralevent
            if entity_types[entityInfo.dwEntityType] == 'ns_ENTITY_NEURALEVENT': 
                
                pNeuralInfo = ns_NEURALINFO()
                neuroshare.ns_GetNeuralInfo ( hFile,  dwEntityID,
                                 byref(pNeuralInfo), ctypes.sizeof(pNeuralInfo))
                #print pNeuralInfo.dwSourceUnitID , pNeuralInfo.szProbeInfo
                if lazy:
                    times = [ ]*pq.s
                else:
                    pData = zeros( (entityInfo.dwItemCount,), dtype = 'f8')
                    dwStartIndex = 0
                    dwIndexCount = entityInfo.dwItemCount
                    neuroshare.ns_GetNeuralData( hFile,  dwEntityID,  dwStartIndex,
                        dwIndexCount,  pData.ctypes.data_as(ctypes.POINTER(c_double)))
                    times = pData*pq.s
                sptr = SpikeTrain(times, name = str(entityInfo.szEntityLabel),)
                if lazy:
                    sptr.lazy_shape = entityInfo.dwItemCount
                seg.spiketrains.append(sptr)
        
        # close
        neuroshare.ns_CloseFile(hFile)
        
        create_many_to_one_relationship(seg)
        return seg




# neuroshare structures
class ns_FILEDESC(ctypes.Structure):
    _fields_ = [('szDescription', c_char*32),
                ('szExtension', c_char*8),
                ('szMacCodes', c_char*8),
                ('szMagicCode', c_char*16),
                ]


class ns_LIBRARYINFO(ctypes.Structure):
    _fields_ = [('dwLibVersionMaj', c_uint32),
                ('dwLibVersionMin', c_uint32),
                ('dwAPIVersionMaj', c_uint32),
                ('dwAPIVersionMin', c_uint32),
                ('szDescription', c_char*64),
                ('szCreator',c_char*64),
                ('dwTime_Year',c_uint32),
                ('dwTime_Month',c_uint32),
                ('dwTime_Day',c_uint32),
                ('dwFlags',c_uint32),
                ('dwMaxFiles',c_uint32),
                ('dwFileDescCount',c_uint32),
                ('FileDesc',ns_FILEDESC*16),
                ]

class ns_FILEINFO(ctypes.Structure):
    _fields_ = [('szFileType', c_char*32),
                ('dwEntityCount', c_uint32),
                ('dTimeStampResolution', c_double),
                ('dTimeSpan', c_double),
                ('szAppName', c_char*64),
                ('dwTime_Year',c_uint32),
                ('dwTime_Month',c_uint32),
                ('dwReserved',c_uint32),
                ('dwTime_Day',c_uint32),
                ('dwTime_Hour',c_uint32),
                ('dwTime_Min',c_uint32),
                ('dwTime_Sec',c_uint32),
                ('dwTime_MilliSec',c_uint32),
                ('szFileComment',c_char*256),
                ]

class ns_ENTITYINFO(ctypes.Structure):
    _fields_ = [('szEntityLabel', c_char*32),
                ('dwEntityType',c_uint32),
                ('dwItemCount',c_uint32),
                ]

entity_types = { 0 : 'ns_ENTITY_UNKNOWN' ,
                    1 : 'ns_ENTITY_EVENT' ,
                    2 : 'ns_ENTITY_ANALOG' ,
                    3 : 'ns_ENTITY_SEGMENT' ,
                    4 : 'ns_ENTITY_NEURALEVENT' ,
                    }

class ns_EVENTINFO(ctypes.Structure):
    _fields_ = [
                ('dwEventType',c_uint32),
                ('dwMinDataLength',c_uint32),
                ('dwMaxDataLength',c_uint32),
                ('szCSVDesc', c_char*128),
                ]

class ns_ANALOGINFO(ctypes.Structure):
    _fields_ = [
                ('dSampleRate',c_double),
                ('dMinVal',c_double),
                ('dMaxVal',c_double),
                ('szUnits', c_char*16),
                ('dResolution',c_double),
                ('dLocationX',c_double),
                ('dLocationY',c_double),
                ('dLocationZ',c_double),
                ('dLocationUser',c_double),
                ('dHighFreqCorner',c_double),
                ('dwHighFreqOrder',c_uint32),
                ('szHighFilterType', c_char*16),
                ('dLowFreqCorner',c_double),
                ('dwLowFreqOrder',c_uint32),
                ('szLowFilterType', c_char*16),
                ('szProbeInfo', c_char*128),
            ]


class ns_SEGMENTINFO(ctypes.Structure):
    _fields_ = [
                ('dwSourceCount',c_uint32),
                ('dwMinSampleCount',c_uint32),
                ('dwMaxSampleCount',c_uint32),
                ('dSampleRate',c_double),
                ('szUnits', c_char*32),
                ]

class ns_SEGSOURCEINFO(ctypes.Structure):
    _fields_ = [
                ('dMinVal',c_double),
                ('dMaxVal',c_double),
                ('dResolution',c_double),
                ('dSubSampleShift',c_double),
                ('dLocationX',c_double),
                ('dLocationY',c_double),
                ('dLocationZ',c_double),
                ('dLocationUser',c_double),
                ('dHighFreqCorner',c_double),
                ('dwHighFreqOrder',c_uint32),
                ('szHighFilterType', c_char*16),
                ('dLowFreqCorner',c_double),
                ('dwLowFreqOrder',c_uint32),
                ('szLowFilterType', c_char*16),
                ('szProbeInfo', c_char*128),                
                ]

class ns_NEURALINFO(ctypes.Structure):
    _fields_ = [
                ('dwSourceEntityID',c_uint32),
                ('dwSourceUnitID',c_uint32),
                ('szProbeInfo',c_char*128),
                ]



