# -*- coding: utf-8 -*-
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

import ctypes
import os

# file no longer exists in Python3
try:
    file
except NameError:
    import io
    file = io.BufferedReader

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Segment, AnalogSignal, SpikeTrain, EventArray


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




    def read_segment(self, import_neuroshare_segment = True,
                     lazy=False, cascade=True):
        """
        Arguments:
            import_neuroshare_segment: import neuroshare segment as SpikeTrain with associated waveforms or not imported at all.

        """

        seg = Segment( file_origin = os.path.basename(self.filename), )

        neuroshare = ctypes.windll.LoadLibrary(self.dllname)

        # API version
        info = ns_LIBRARYINFO()
        neuroshare.ns_GetLibraryInfo(ctypes.byref(info) , ctypes.sizeof(info))
        seg.annotate(neuroshare_version = str(info.dwAPIVersionMaj)+'.'+str(info.dwAPIVersionMin))

        if not cascade:
            return seg


        # open file
        hFile = ctypes.c_uint32(0)
        neuroshare.ns_OpenFile(ctypes.c_char_p(self.filename) ,ctypes.byref(hFile))
        fileinfo = ns_FILEINFO()
        neuroshare.ns_GetFileInfo(hFile, ctypes.byref(fileinfo) , ctypes.sizeof(fileinfo))

        # read all entities
        for dwEntityID in range(fileinfo.dwEntityCount):
            entityInfo = ns_ENTITYINFO()
            neuroshare.ns_GetEntityInfo( hFile, dwEntityID, ctypes.byref(entityInfo), ctypes.sizeof(entityInfo))
            #~ print 'type', entityInfo.dwEntityType,entity_types[entityInfo.dwEntityType], 'count', entityInfo.dwItemCount
            #~ print  entityInfo.szEntityLabel

            # EVENT
            if entity_types[entityInfo.dwEntityType] == 'ns_ENTITY_EVENT':
                pEventInfo = ns_EVENTINFO()
                neuroshare.ns_GetEventInfo ( hFile,  dwEntityID,  ctypes.byref(pEventInfo), ctypes.sizeof(pEventInfo))
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
                pdTimeStamp  = ctypes.c_double(0.)
                pdwDataRetSize = ctypes.c_uint32(0)

                ea = EventArray(name = str(entityInfo.szEntityLabel),)
                if not lazy:
                    times = [ ]
                    labels = [ ]
                    for dwIndex in range(entityInfo.dwItemCount ):
                        neuroshare.ns_GetEventData ( hFile, dwEntityID, dwIndex,
                                            ctypes.byref(pdTimeStamp), ctypes.byref(pData),
                                            ctypes.sizeof(pData), ctypes.byref(pdwDataRetSize) )
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

                neuroshare.ns_GetAnalogInfo( hFile, dwEntityID,ctypes.byref(pAnalogInfo),ctypes.sizeof(pAnalogInfo) )
                #~ print 'dSampleRate' , pAnalogInfo.dSampleRate , pAnalogInfo.szUnits
                dwStartIndex = ctypes.c_uint32(0)
                dwIndexCount = entityInfo.dwItemCount

                if lazy:
                    signal = [ ]*pq.Quantity(1, pAnalogInfo.szUnits)
                else:
                    pdwContCount = ctypes.c_uint32(0)
                    pData = np.zeros( (entityInfo.dwItemCount,), dtype = 'f8')
                    neuroshare.ns_GetAnalogData ( hFile,  dwEntityID,  dwStartIndex,
                                     dwIndexCount, ctypes.byref( pdwContCount) , pData.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                    pszMsgBuffer  = ctypes.create_string_buffer(" "*256)
                    neuroshare.ns_GetLastErrorMsg(ctypes.byref(pszMsgBuffer), 256)
                    #~ print 'pszMsgBuffer' , pszMsgBuffer.value
                    signal = pData[:pdwContCount.value]*pq.Quantity(1, pAnalogInfo.szUnits)

                #t_start
                dwIndex = 0
                pdTime = ctypes.c_double(0)
                neuroshare.ns_GetTimeByIndex( hFile,  dwEntityID,  dwIndex, ctypes.byref(pdTime))

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

                neuroshare.ns_GetSegmentInfo( hFile,  dwEntityID,
                                             ctypes.byref(pdwSegmentInfo), ctypes.sizeof(pdwSegmentInfo) )
                nsource = pdwSegmentInfo.dwSourceCount

                pszMsgBuffer  = ctypes.create_string_buffer(" "*256)
                neuroshare.ns_GetLastErrorMsg(ctypes.byref(pszMsgBuffer), 256)
                #~ print 'pszMsgBuffer' , pszMsgBuffer.value

                #~ print 'pdwSegmentInfo.dwSourceCount' , pdwSegmentInfo.dwSourceCount
                for dwSourceID in range(pdwSegmentInfo.dwSourceCount) :
                    pSourceInfo = ns_SEGSOURCEINFO()
                    neuroshare.ns_GetSegmentSourceInfo( hFile,  dwEntityID, dwSourceID,
                                    ctypes.byref(pSourceInfo), ctypes.sizeof(pSourceInfo) )

                if lazy:
                    sptr = SpikeTrain(times, name = str(entityInfo.szEntityLabel))
                    sptr.lazy_shape = entityInfo.dwItemCount
                else:
                    pdTimeStamp  = ctypes.c_double(0.)
                    dwDataBufferSize = pdwSegmentInfo.dwMaxSampleCount*pdwSegmentInfo.dwSourceCount
                    pData = np.zeros( (dwDataBufferSize), dtype = 'f8')
                    pdwSampleCount = ctypes.c_uint32(0)
                    pdwUnitID= ctypes.c_uint32(0)

                    nsample  = pdwSampleCount.value
                    times = np.empty( (entityInfo.dwItemCount), drtype = 'f')
                    waveforms = np.empty( (entityInfo.dwItemCount, nsource, nsample), drtype = 'f')
                    for dwIndex in range(entityInfo.dwItemCount ):
                        neuroshare.ns_GetSegmentData ( hFile,  dwEntityID,  dwIndex,
                            ctypes.byref(pdTimeStamp), pData.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            dwDataBufferSize * 8, ctypes.byref(pdwSampleCount),
                                ctypes.byref(pdwUnitID ) )
                        #print 'dwDataBufferSize' , dwDataBufferSize,pdwSampleCount , pdwUnitID

                        times[dwIndex] = pdTimeStamp.value
                        waveforms[dwIndex, :,:] = pData[:nsample*nsource].reshape(nsample ,nsource).transpose()

                    sptr = SpikeTrain(times*pq.s,
                                        waveforms = waveforms*pq.Quantity(1., str(pdwSegmentInfo.szUnits) ),
                                        left_sweep = nsample/2./float(pdwSegmentInfo.dSampleRate)*pq.s,
                                        sampling_rate = float(pdwSegmentInfo.dSampleRate)*pq.Hz,
                                        name = str(entityInfo.szEntityLabel),
                                        )
                seg.spiketrains.append(sptr)


            # neuralevent
            if entity_types[entityInfo.dwEntityType] == 'ns_ENTITY_NEURALEVENT':

                pNeuralInfo = ns_NEURALINFO()
                neuroshare.ns_GetNeuralInfo ( hFile,  dwEntityID,
                                 ctypes.byref(pNeuralInfo), ctypes.sizeof(pNeuralInfo))
                #print pNeuralInfo.dwSourceUnitID , pNeuralInfo.szProbeInfo
                if lazy:
                    times = [ ]*pq.s
                else:
                    pData = np.zeros( (entityInfo.dwItemCount,), dtype = 'f8')
                    dwStartIndex = 0
                    dwIndexCount = entityInfo.dwItemCount
                    neuroshare.ns_GetNeuralData( hFile,  dwEntityID,  dwStartIndex,
                        dwIndexCount,  pData.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                    times = pData*pq.s
                sptr = SpikeTrain(times, name = str(entityInfo.szEntityLabel),)
                if lazy:
                    sptr.lazy_shape = entityInfo.dwItemCount
                seg.spiketrains.append(sptr)

        # close
        neuroshare.ns_CloseFile(hFile)

        seg.create_many_to_one_relationship()
        return seg




# neuroshare structures
class ns_FILEDESC(ctypes.Structure):
    _fields_ = [('szDescription', ctypes.c_char*32),
                ('szExtension', ctypes.c_char*8),
                ('szMacCodes', ctypes.c_char*8),
                ('szMagicCode', ctypes.c_char*16),
                ]


class ns_LIBRARYINFO(ctypes.Structure):
    _fields_ = [('dwLibVersionMaj', ctypes.c_uint32),
                ('dwLibVersionMin', ctypes.c_uint32),
                ('dwAPIVersionMaj', ctypes.c_uint32),
                ('dwAPIVersionMin', ctypes.c_uint32),
                ('szDescription', ctypes.c_char*64),
                ('szCreator',ctypes.c_char*64),
                ('dwTime_Year',ctypes.c_uint32),
                ('dwTime_Month',ctypes.c_uint32),
                ('dwTime_Day',ctypes.c_uint32),
                ('dwFlags',ctypes.c_uint32),
                ('dwMaxFiles',ctypes.c_uint32),
                ('dwFileDescCount',ctypes.c_uint32),
                ('FileDesc',ns_FILEDESC*16),
                ]

class ns_FILEINFO(ctypes.Structure):
    _fields_ = [('szFileType', ctypes.c_char*32),
                ('dwEntityCount', ctypes.c_uint32),
                ('dTimeStampResolution', ctypes.c_double),
                ('dTimeSpan', ctypes.c_double),
                ('szAppName', ctypes.c_char*64),
                ('dwTime_Year',ctypes.c_uint32),
                ('dwTime_Month',ctypes.c_uint32),
                ('dwReserved',ctypes.c_uint32),
                ('dwTime_Day',ctypes.c_uint32),
                ('dwTime_Hour',ctypes.c_uint32),
                ('dwTime_Min',ctypes.c_uint32),
                ('dwTime_Sec',ctypes.c_uint32),
                ('dwTime_MilliSec',ctypes.c_uint32),
                ('szFileComment',ctypes.c_char*256),
                ]

class ns_ENTITYINFO(ctypes.Structure):
    _fields_ = [('szEntityLabel', ctypes.c_char*32),
                ('dwEntityType',ctypes.c_uint32),
                ('dwItemCount',ctypes.c_uint32),
                ]

entity_types = { 0 : 'ns_ENTITY_UNKNOWN' ,
                    1 : 'ns_ENTITY_EVENT' ,
                    2 : 'ns_ENTITY_ANALOG' ,
                    3 : 'ns_ENTITY_SEGMENT' ,
                    4 : 'ns_ENTITY_NEURALEVENT' ,
                    }

class ns_EVENTINFO(ctypes.Structure):
    _fields_ = [
                ('dwEventType',ctypes.c_uint32),
                ('dwMinDataLength',ctypes.c_uint32),
                ('dwMaxDataLength',ctypes.c_uint32),
                ('szCSVDesc', ctypes.c_char*128),
                ]

class ns_ANALOGINFO(ctypes.Structure):
    _fields_ = [
                ('dSampleRate',ctypes.c_double),
                ('dMinVal',ctypes.c_double),
                ('dMaxVal',ctypes.c_double),
                ('szUnits', ctypes.c_char*16),
                ('dResolution',ctypes.c_double),
                ('dLocationX',ctypes.c_double),
                ('dLocationY',ctypes.c_double),
                ('dLocationZ',ctypes.c_double),
                ('dLocationUser',ctypes.c_double),
                ('dHighFreqCorner',ctypes.c_double),
                ('dwHighFreqOrder',ctypes.c_uint32),
                ('szHighFilterType', ctypes.c_char*16),
                ('dLowFreqCorner',ctypes.c_double),
                ('dwLowFreqOrder',ctypes.c_uint32),
                ('szLowFilterType', ctypes.c_char*16),
                ('szProbeInfo', ctypes.c_char*128),
            ]


class ns_SEGMENTINFO(ctypes.Structure):
    _fields_ = [
                ('dwSourceCount',ctypes.c_uint32),
                ('dwMinSampleCount',ctypes.c_uint32),
                ('dwMaxSampleCount',ctypes.c_uint32),
                ('dSampleRate',ctypes.c_double),
                ('szUnits', ctypes.c_char*32),
                ]

class ns_SEGSOURCEINFO(ctypes.Structure):
    _fields_ = [
                ('dMinVal',ctypes.c_double),
                ('dMaxVal',ctypes.c_double),
                ('dResolution',ctypes.c_double),
                ('dSubSampleShift',ctypes.c_double),
                ('dLocationX',ctypes.c_double),
                ('dLocationY',ctypes.c_double),
                ('dLocationZ',ctypes.c_double),
                ('dLocationUser',ctypes.c_double),
                ('dHighFreqCorner',ctypes.c_double),
                ('dwHighFreqOrder',ctypes.c_uint32),
                ('szHighFilterType', ctypes.c_char*16),
                ('dLowFreqCorner',ctypes.c_double),
                ('dwLowFreqOrder',ctypes.c_uint32),
                ('szLowFilterType', ctypes.c_char*16),
                ('szProbeInfo', ctypes.c_char*128),
                ]

class ns_NEURALINFO(ctypes.Structure):
    _fields_ = [
                ('dwSourceEntityID',ctypes.c_uint32),
                ('dwSourceUnitID',ctypes.c_uint32),
                ('szProbeInfo',ctypes.c_char*128),
                ]



