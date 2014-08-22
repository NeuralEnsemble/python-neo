#-*- coding: utf-8 -*-
"""
this code merges the neuroshareIO together with the multichannelIO for construction of
neo objetcs.
it should work in a way that it automatically selects the proper system to load the file 
and read in the data"""

#import things that are common to both methods
# needed for python 3 compatibility
from __future__ import absolute_import

# note neo.core needs only numpy and quantities
import numpy as np
import quantities as pq

#import BaseIO
from neo.io.baseio import BaseIO
#import objects from neo.core
from neo.core import Segment, AnalogSignal, SpikeTrain, EventArray



#check to see if the neuroshare bindings are properly imported    
flag = 0
try:
    import neuroshare as ns
except ImportError as err:
    flag = 1	
    print('\n neuroshare library not found, loading data will not work!' )
    print('\n be sure to install the library found at:')
    print('\n www.http://pythonhosted.org/neuroshare/')

else:
    print('neuroshare library successfully imported')


if flag == 1:

    """
    Class for "reading" data from *.mcd files (files generate with the MCRack
    software, distributed by Multichannelsystems company - Reutlingen, Germany).
    It runs through the whole file and searches for: analog signals, spike cutouts,
    and trigger events (without duration)
    Depends on: Neuroshare API 0.9.1, numpy 1.6.1, quantities 0.10.1
    
    Supported: Read
       
    Author: Andre Maia Chagas
    """

   

    # some tools to finalize the hierachy
    from neo.io.tools import create_many_to_one_relationship


    # create an object based on BaseIO
    class NeuroshareIO(BaseIO):
        """
        Class for "reading" data from *.mcd files (files generate with the MCRack
        software, distributed by Multichannelsystems company - Reutlingen, Germany).
        It runs through the whole file and searches for: analog signals, spike cutouts,
        and trigger events (without duration)
        Depends on: Neuroshare API 0.9.1, numpy 1.6.1, quantities 0.10.1

        Supported: Read

        Author: Andre Maia Chagas
        """
        #setting some class parameters
        is_readable = True # This class can only read data
        is_writable = False # write is not supported
        supported_objects  = [ Segment , AnalogSignal, SpikeTrain, EventArray ]
        
        has_header         = False
        is_streameable     = False

        readable_objects    = [ Segment , AnalogSignal, SpikeTrain, EventArray]
        # This class is not able to write objects
        writeable_objects   = [ ]

 
        # This is for GUI stuff : a definition for parameters when reading.
        # This dict should be keyed by object (`Block`). Each entry is a list
        # of tuple. The first entry in each tuple is the parameter name. The
        # second entry is a dict with keys 'value' (for default value),
        # and 'label' (for a descriptive name).
        # Note that if the highest-level object requires parameters,
        # common_io_test will be skipped.
        #read_params = {
        #Segment : [
        #('segment_duration',
         #{'value' : 15., 'label' : 'Segment size (s.)'}),
         #('num_analogsignal',
         #{'value' : 8, 'label' : 'Number of recording points'}),
         #('num_spiketrain_by_channel',
         #{'value' : 3, 'label' : 'Num of spiketrains'}),
         #],
         #}

        # do not supported write so no GUI stuff
        #write_params       = None
        #
        #name               = 'example'

        #extensions          = [ 'nof' ]
    
        # mode can be 'file' or 'dir' or 'fake' or 'database'
        # the main case is 'file' but some reader are base on a directory or a database
        # this info is for GUI stuff also
        mode = 'file'



        def __init__(self , filename = None) :
            """
            Arguments:
                filename : the filename
                The init function will run automatically upon calling of the class, as 
                in: test = MultichannelIO(filename = filetoberead.mcd), therefore the first
                operations with the file are set here, so that the user doesn't have to
                remember to use another method, than the ones defined in the NEO library
                
                """
            BaseIO.__init__(self)
            self.filename = filename
            #set the flags for each event type
            eventID = 1        
            analogID = 2
            epochID = 3
            #if a filename was given, create a dictionary with information that will 
            #be needed later on.
            if self.filename != None:
                self.fd = ns.File(self.filename)
                #get all the metadata from file
                self.metadata = self.fd.metadata_raw
                #get sampling rate
                self.metadata['sampRate'] = 1./self.metadata['TimeStampResolution']#hz
                #create lists and array for electrode, spike cutouts and trigger channels
                self.metadata['elecChannels'] = list()
                self.metadata['elecChanId']   = list()
                self.metadata['num_analogs']  = 0
                self.metadata['spkChannels']  = list()
                self.metadata['spkChanId']    = list()
                self.metadata['num_spkChans'] = 0
                self.metadata['triggers']     = list()
                self.metadata['triggersId']   = list()
                self.metadata['num_trigs']    = 0
                #loop through all entities in file to get the indexes for each entity
                #type, so that one can run through the indexes later, upon reading the 
                #segment
                for entity in self.fd.entities:
                    #if entity is analog and not the digital line recording 
                    #(stored as analog in the *.mcd files) 
                    if entity.entity_type == analogID and entity.label[0:4]!= 'digi':
                        #get the electrode number                    
                        self.metadata['elecChannels'].append(entity.label[-4:])
                        #get the electrode index
                        self.metadata['elecChanId'].append(entity.id)
                        #increase the number of electrodes found
                        self.metadata['num_analogs'] += 1
                        # if the entity is a event entitiy, but not the digital line,
                    if entity.entity_type == eventID and entity.label[-2:] != 'D1':
                        #get the digital bit/trigger number
                        self.metadata['triggers'].append(entity.label[0:4]+entity.label[-4:])
                        #get the digital bit index
                        self.metadata['triggersId'].append(entity.id)
                        #increase the number of triggers found                    
                        self.metadata['num_trigs'] += 1

                    if entity.entity_type == epochID and entity.label[0:4] == 'spks':
                        self.metadata['spkChannels'].append(entity.label[-4:])
                        self.metadata['spkChanId'].append(entity.id)
                        self.metadata['num_spkChans'] += 1
            
                    
        #create function to read segment
        def read_segment(self,
                     # the 2 first keyword arguments are imposed by neo.io API
                     lazy = False,
                     cascade = True,
                     # all following arguments are decied by this IO and are free
                     t_start = 0.,
                     segment_duration = 0.,
                     #num_spiketrain_by_channel = 3,
                    ):
            """
            Return a Segment containing all analog and spike channels, as well as
            all trigger events.
            
            Parameters:
                segment_duration :is the size in secend of the segment.
                num_analogsignal : number of AnalogSignal in this segment
                num_spiketrain : number of SpikeTrain in this segment
            
            """
            #if no segment duration is given, use the complete file
            if segment_duration ==0.:
                segment_duration=float(self.metadata['TimeSpan'])
                #time vector for generated signal
                #timevect = np.arange(t_start, t_start+ segment_duration , 1./self.metadata['sampRate'])
                
                # create an empty segment
                seg = Segment( name = 'segment from the MultichannelIO')

            if cascade:
                # read nested analosignal
            
                if self.metadata['num_analogs'] == 0:
                    print ('no analog signals in this file!')
                else:
                    #run through the number of analog channels found at the __init__ function
                    for i in range(self.metadata['num_analogs']):
                        #create an analog signal object for each channel found
                        ana = self.read_analogsignal( lazy = lazy , cascade = cascade ,
                                             channel_index = self.metadata['elecChanId'][i],
                                            segment_duration = segment_duration, t_start=t_start)
                        #add analog signal read to segment object
                        seg.analogsignals += [ ana ]
            
                # read triggers (in this case without any duration)
                for i in range(self.metadata['num_trigs']):
                    #create event object for each trigger/bit found
                    eva = self.read_eventarray(lazy = lazy , cascade = cascade,
                        channel_index = self.metadata['triggersId'][i])
                    #add event object to segment
                    seg.eventarrays +=  [eva]

                # read nested spiketrain
                #run through all spike channels found
                for i in range(self.metadata['num_spkChans']):
                    #create spike object
                    sptr = self.read_spiketrain(lazy = lazy, cascade = cascade,
                        channel_index = self.metadata['spkChanId'][i],
                        segment_duration = segment_duration,)
                    #add the spike object to segment
                    seg.spiketrains += [ sptr ]

            create_many_to_one_relationship(seg)
            return seg


        def read_analogsignal(self,
                          # the 2 first key arguments are imposed by neo.io API
                          lazy = False,
                          cascade = True,
                          channel_index = 0,
                          #time in seconds to be read
                          segment_duration = 0.,
                          #time in seconds to start reading from
                          t_start = 0.,
                          ):
            """
            With this IO AnalogSignal can e acces directly with its channel number

            """
        
        
            if lazy:
                anasig = AnalogSignal([], units='V', sampling_rate =  self.metadata['sampRate'] * pq.Hz,
                                  t_start=t_start * pq.s,
                                  channel_index=channel_index)
                #create a dummie time vector                     
                tvect = np.arange(t_start, t_start+ segment_duration , 1./self.metadata['sampRate'])                                  
                # we add the attribute lazy_shape with the size if loaded
                anasig.lazy_shape = tvect.shape
            else:
                #get the analog object
                sig =  self.fd.get_entity(channel_index)
                #get the units (V, mV etc)            
                sigUnits = sig.units
                #get the electrode number
                chanName = sig.label[-4:]
            
                #transform t_start into index (reading will start from this index)           
                startat = int(t_start*self.metadata['sampRate'])
                #get the number of bins to read in
                bins = int((segment_duration-t_start) * self.metadata['sampRate'])
            
                #if the number of bins to read is bigger than 
                #the total number of bins, read only till the end of analog object
                if startat+bins > sig.item_count:
                    bins = sig.item_count-startat
                    #read the data from the sig object
                    sig,_,_ = sig.get_data(index = startat, count = bins)
                    #store it to the 'AnalogSignal' object
                    anasig = AnalogSignal(sig, units = sigUnits, sampling_rate=self.metadata['sampRate'] * pq.Hz,
                                  t_start=t_start * pq.s,
                                  channel_index=channel_index)

            # annotate from which electrode the signal comes from
            anasig.annotate(info = 'signal from channel %s' %chanName )

            return anasig



        #function to read spike trains
        def read_spiketrain(self ,
                        # the 2 first key arguments are imposed by neo.io API
                        lazy = False,
                        cascade = True,
                        segment_duration = 15.,
                        t_start = 0.,
                        channel_index = 0):
            """
            Function to read in spike trains. This API still does not support read in of
            specific channels as they are recorded. rather the fuunction gets the entity set
            by 'channel_index' which is set in the __init__ function (all spike channels)
            """

                

            #sampling rate
            sr = self.metadata['sampRate']
        
            # create a list to store spiketrain times
            times = list() 
        
            if lazy:
                # we add the attribute lazy_shape with the size if lazy
                spiketr = SpikeTrain(times,units = pq.s, 
                       t_stop = t_start+segment_duration,
                       t_start = t_start*pq.s,lazy_shape = 40)
        
            else:
                #get the spike data from a specific channel index
                tempSpks =  self.fd.get_entity(channel_index)    
                #create a numpy empty array to store the waveforms
                waveforms=np.array(np.empty([tempSpks.item_count,tempSpks.max_sample_count]))
                #loop through the data from the specific channel index
                for i in range(tempSpks.item_count):
                    #get cutout, timestamp, cutout duration, and spike unit
                    tempCuts,timeStamp,duration,unit = tempSpks.get_data(i)
                    #save the cutout in the waveform matrix
                    waveforms[i]=tempCuts[0]
                    #append time stamp to list
                    times.append(timeStamp)
                
                #create a spike train object
                spiketr = SpikeTrain(times,units = pq.s, 
                         t_stop = t_start+segment_duration,
                         t_start = t_start*pq.s,
                         name ='spikes from electrode'+tempSpks.label[-3:],
                         waveforms = waveforms*pq.volt,
                         sampling_rate = sr * pq.Hz,
                         file_origin = self.filename,
                         annotate = ('channel_index:'+ str(channel_index)))
            
            return spiketr

        def read_eventarray(self,lazy = False, cascade = True,channel_index = 0):
            """function to read digital timestamps. this function only reads the event
            onset and disconsiders its duration. to get digital event durations, use 
            the epoch function (to be implemented)."""
            if lazy:
                eva = EventArray(file_origin = self.filename)        
            else:
                #create temporary empty lists to store data
                tempNames = list()
                tempTimeStamp = list()
                #get entity from file
                trigEntity = self.fd.get_entity(channel_index)
                #run through entity
                for i in range(trigEntity.item_count):
                    #get in which digital bit was the trigger detected
                    tempNames.append(trigEntity.label[-8:])
                    #get the time stamps
                    tempData, _ = trigEntity.get_data(i)
                    #append the time stamp to them empty list
                    tempTimeStamp.append(tempData)
                    #create an event array        
                    eva = EventArray(file_origin = self.filename,labels = tempNames,
                        times = tempTimeStamp,
                        description = 'here are stored all the trigger events'+
                            '(without their durations) as detected by '+
                            'the Trigger detector tool in MCRack' )       
            return eva

else:

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
