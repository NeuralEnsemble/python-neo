"""
Class for "reading" data from Neuroshare compatible files (check neuroshare.org)
It runs through the whole file and searches for: analog signals, spike cutouts,
and trigger events (without duration)
Depends on: Neuroshare API 0.9.1, numpy 1.6.1, quantities 0.10.1

Supported: Read

Author: Andre Maia Chagas
"""

# needed for python 3 compatibility
from __future__ import absolute_import

# note neo.core needs only numpy and quantities
import numpy as np
import quantities as pq

import os

#check to see if the neuroshare bindings are properly imported    
try:
    import neuroshare as ns
except ImportError as err:
    print (err)
    #print('\n neuroshare library not found, loading data will not work!' )
    #print('\n be sure to install the library found at:')
    #print('\n www.http://pythonhosted.org/neuroshare/')

else:
    pass
    #print('neuroshare library successfully imported')


#import BaseIO
from neo.io.baseio import BaseIO

#import objects from neo.core
from neo.core import Segment, AnalogSignal, SpikeTrain, EventArray, EpochArray


# create an object based on BaseIO
class NeuroshareapiIO(BaseIO):

    #setting some class parameters
    is_readable = True # This class can only read data
    is_writable = False # write is not supported
    supported_objects  = [ Segment , AnalogSignal, SpikeTrain, EventArray, EpochArray ]
    
    has_header         = False
    is_streameable     = False

    readable_objects    = [ Segment , AnalogSignal, SpikeTrain, EventArray, EpochArray]
    # This class is not able to write objects
    writeable_objects   = [ ]

 
#    # This is for GUI stuff : a definition for parameters when reading.
#    # This dict should be keyed by object (`Block`). Each entry is a list
#    # of tuple. The first entry in each tuple is the parameter name. The
#    # second entry is a dict with keys 'value' (for default value),
#    # and 'label' (for a descriptive name).
#    # Note that if the highest-level object requires parameters,
#    # common_io_test will be skipped.
    read_params = {
        Segment : [
            ("segment_duration",{"value" : 0., "label" : "Segment size (s.)"}),
            ("t_start",{"value" : 0.,"label" : "start reading (s.)"}),
            #("lazy",{"value" : False,"label" : "load in lazy mode?"}),
            #("cascade",{"value" : True,"label" : "Cascade?"})
#            ("num_analogsignal",
#                {'value" : 8, "label" : "Number of recording points"}),
#            ("num_spiketrain_by_channel',
#                {"value" : 3, "label" : "Num of spiketrains"}),
            ],
        }
#
    # do not supported write so no GUI stuff
    write_params       = None

    name               = "Neuroshare"

    extensions          = []

    # This object operates on neuroshare files
    mode = "file"



    def __init__(self , filename = None, dllpath = None) :
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
            if dllpath is not None:
                name = os.path.splitext(os.path.basename(dllpath))[0]
                library = ns.Library(name, dllpath)
            else:
                library = None
            self.fd = ns.File(self.filename, library = library)
            #get all the metadata from file
            self.metadata = self.fd.metadata_raw
            #get sampling rate
            self.metadata["sampRate"] = 1./self.metadata["TimeStampResolution"]#hz
            #create lists and array for electrode, spike cutouts and trigger channels
            self.metadata["elecChannels"] = list()
            self.metadata["elecChanId"]   = list()
            self.metadata["num_analogs"]  = 0
            self.metadata["spkChannels"]  = list()
            self.metadata["spkChanId"]    = list()
            self.metadata["num_spkChans"] = 0
            self.metadata["triggers"]     = list()
            self.metadata["triggersId"]   = list()
            self.metadata["num_trigs"]    = 0
            self.metadata["digital epochs"] = list()
            self.metadata["digiEpochId"]    = list()
            self.metadata["num_digiEpochs"] = 0

            #loop through all entities in file to get the indexes for each entity
            #type, so that one can run through the indexes later, upon reading the 
            #segment
            for entity in self.fd.entities:
                #if entity is analog and not the digital line recording 
                #(stored as analog in neuroshare files) 
                if entity.entity_type == analogID and entity.label[0:4]!= "digi":
                    #get the electrode number                    
                    self.metadata["elecChannels"].append(entity.label[-4:])
                    #get the electrode index
                    self.metadata["elecChanId"].append(entity.id)
                    #increase the number of electrodes found
                    self.metadata["num_analogs"] += 1
                # if the entity is a event entitiy and a trigger
                if entity.entity_type == eventID and entity.label[0:4] == "trig":
                    #get the digital bit/trigger number
                    self.metadata["triggers"].append(entity.label[0:4]+entity.label[-4:])
                    #get the digital bit index
                    self.metadata["triggersId"].append(entity.id)
                    #increase the number of triggers found                    
                    self.metadata["num_trigs"] += 1
                #if the entity is non triggered digital values with duration
                if entity.entity_type == eventID and entity.label[0:4] == "digi":
                    #get the digital bit number
                    self.metadata["digital epochs"].append(entity.label[-5:])
                    #get the digital bit index
                    self.metadata["digiEpochId"].append(entity.id)
                    #increase the number of triggers found                    
                    self.metadata["num_digiEpochs"] += 1
                #if the entity is spike cutouts
                if entity.entity_type == epochID and entity.label[0:4] == "spks":
                    self.metadata["spkChannels"].append(entity.label[-4:])
                    self.metadata["spkChanId"].append(entity.id)
                    self.metadata["num_spkChans"] += 1
            
    #function to create a block and read in a segment
#    def create_block(self,
#                     lazy = False,
#                     cascade = True,
#                     
#                     ):
#        
#        blk=Block(name = self.fileName+"_segment:",
#                  file_datetime = str(self.metadata_raw["Time_Day"])+"/"+
#                                  str(self.metadata_raw["Time_Month"])+"/"+
#                                  str(self.metadata_raw["Time_Year"])+"_"+
#                                  str(self.metadata_raw["Time_Hour"])+":"+
#                                  str(self.metadata_raw["Time_Min"]))
#        
#        blk.rec_datetime = blk.file_datetime
#        return blk
    
    #create function to read segment
    def read_segment(self,
                     # the 2 first keyword arguments are imposed by neo.io API
                     lazy = False,
                     cascade = True,
                     # all following arguments are decided by this IO and are free
                     t_start = 0.,
                     segment_duration = 0.,
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
        if segment_duration == 0. :
            segment_duration=float(self.metadata["TimeSpan"])
        #if the segment duration is bigger than file, use the complete file
        if segment_duration >=float(self.metadata["TimeSpan"]):
            segment_duration=float(self.metadata["TimeSpan"])
        #if the time sum of start point and segment duration is bigger than
        #the file time span, cap it at the end
        if segment_duration+t_start>float(self.metadata["TimeSpan"]):
            segment_duration = float(self.metadata["TimeSpan"])-t_start
        
        # create an empty segment
        seg = Segment( name = "segment from the NeuroshareapiIO")

        if cascade:
            # read nested analosignal
            
            if self.metadata["num_analogs"] == 0:
                print ("no analog signals in this file!")
            else:
                #run through the number of analog channels found at the __init__ function
                for i in range(self.metadata["num_analogs"]):
                    #create an analog signal object for each channel found
                    ana = self.read_analogsignal( lazy = lazy , cascade = cascade ,
                                             channel_index = self.metadata["elecChanId"][i],
                                            segment_duration = segment_duration, t_start=t_start)
                    #add analog signal read to segment object
                    seg.analogsignals += [ ana ]
            
            # read triggers (in this case without any duration)
            for i in range(self.metadata["num_trigs"]):
                #create event object for each trigger/bit found
                eva = self.read_eventarray(lazy = lazy , 
                                           cascade = cascade,
                                           channel_index = self.metadata["triggersId"][i],
                                           segment_duration = segment_duration,
                                           t_start = t_start,)
                #add event object to segment
                seg.eventarrays +=  [eva]
            #read epochs (digital events with duration)
            for i in range(self.metadata["num_digiEpochs"]):
                #create event object for each trigger/bit found
                epa = self.read_epocharray(lazy = lazy, 
                                           cascade = cascade,
                                           channel_index = self.metadata["digiEpochId"][i],
                                            segment_duration = segment_duration,
                                            t_start = t_start,)
                #add event object to segment
                seg.epocharrays +=  [epa]
            # read nested spiketrain
            #run through all spike channels found
            for i in range(self.metadata["num_spkChans"]):
                #create spike object
                sptr = self.read_spiketrain(lazy = lazy, cascade = cascade,
                        channel_index = self.metadata["spkChanId"][i],
                        segment_duration = segment_duration,
                        t_start = t_start)
                #add the spike object to segment
                seg.spiketrains += [sptr]

        seg.create_many_to_one_relationship()
        
        return seg

    """
        With this IO AnalogSignal can be accessed directly with its channel number
    """
    def read_analogsignal(self,
                          # the 2 first key arguments are imposed by neo.io
                          lazy = False,
                          cascade = True,
                          #channel index as given by the neuroshare API
                          channel_index = 0,
                          #time in seconds to be read
                          segment_duration = 0.,
                          #time in seconds to start reading from
                          t_start = 0.,
                          ):
        
        #some controls:        
        #if no segment duration is given, use the complete file
        if segment_duration ==0.:
            segment_duration=float(self.metadata["TimeSpan"])
        #if the segment duration is bigger than file, use the complete file
        if segment_duration >=float(self.metadata["TimeSpan"]):
            segment_duration=float(self.metadata["TimeSpan"])
            
        if lazy:
            anasig = AnalogSignal([], units="V", sampling_rate =  self.metadata["sampRate"] * pq.Hz,
                                  t_start=t_start * pq.s,
                                  )
            #create a dummie time vector                     
            tvect = np.arange(t_start, t_start+ segment_duration , 1./self.metadata["sampRate"])                                  
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
            startat = int(t_start*self.metadata["sampRate"])
            #get the number of bins to read in
            bins = int(segment_duration * self.metadata["sampRate"])
            
            #if the number of bins to read is bigger than 
            #the total number of bins, read only till the end of analog object
            if startat+bins > sig.item_count:
                bins = sig.item_count-startat
            #read the data from the sig object
            sig,_,_ = sig.get_data(index = startat, count = bins)
            #store it to the 'AnalogSignal' object
            anasig = AnalogSignal(sig, units = sigUnits, sampling_rate=self.metadata["sampRate"] * pq.Hz,
                                  t_start=t_start * pq.s,
                                  t_stop = (t_start+segment_duration)*pq.s,
                                  channel_index=channel_index)

            # annotate from which electrode the signal comes from
            anasig.annotate(info = "signal from channel %s" %chanName )

        return anasig



    #function to read spike trains
    def read_spiketrain(self ,
                        # the 2 first key arguments are imposed by neo.io API
                        lazy = False,
                        cascade = True,
                        channel_index = 0,
                        segment_duration = 0.,
                        t_start = 0.):
        """
        Function to read in spike trains. This API still does not support read in of
        specific channels as they are recorded. rather the fuunction gets the entity set
        by 'channel_index' which is set in the __init__ function (all spike channels)
        """
        
        #sampling rate
        sr = self.metadata["sampRate"]
        
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
            #transform t_start into index (reading will start from this index) 
            startat = tempSpks.get_index_by_time(t_start,0)#zero means closest index to value
            #get the last index to read, using segment duration and t_start
            endat = tempSpks.get_index_by_time(float(segment_duration+t_start),-1)#-1 means last index before time
            numIndx = endat-startat
            #get the end point using segment duration
            #create a numpy empty array to store the waveforms
            waveforms=np.array(np.zeros([numIndx,tempSpks.max_sample_count]))
            #loop through the data from the specific channel index
            for i in range(startat,endat,1):
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
                         name ="spikes from electrode"+tempSpks.label[-3:],
                         waveforms = waveforms*pq.volt,
                         sampling_rate = sr * pq.Hz,
                         file_origin = self.filename,
                         annotate = ("channel_index:"+ str(channel_index)))
            
        return spiketr

    def read_eventarray(self,lazy = False, cascade = True,
                        channel_index = 0,
                        t_start = 0.,
                        segment_duration = 0.):
        """function to read digital timestamps. this function only reads the event
        onset. to get digital event durations, use the epoch function (to be implemented)."""
        if lazy:
            eva = EventArray(file_origin = self.filename)        
        else:
            #create temporary empty lists to store data
            tempNames = list()
            tempTimeStamp = list()
            #get entity from file
            trigEntity = self.fd.get_entity(channel_index)
            #transform t_start into index (reading will start from this index) 
            startat = trigEntity.get_index_by_time(t_start,0)#zero means closest index to value
            #get the last index to read, using segment duration and t_start
            endat = trigEntity.get_index_by_time(float(segment_duration+t_start),-1)#-1 means last index before time
            #numIndx = endat-startat
            #run through specified intervals in entity
            for i in range(startat,endat+1,1):#trigEntity.item_count):
                #get in which digital bit was the trigger detected
                tempNames.append(trigEntity.label[-8:])
                #get the time stamps of onset events
                tempData, onOrOff = trigEntity.get_data(i)
                #if this was an onset event, save it to the list
                #on triggered recordings it seems that only onset events are
                #recorded. On continuous recordings both onset(==1) 
                #and offset(==255) seem to be recorded
                if onOrOff == 1:               
                    #append the time stamp to them empty list
                    tempTimeStamp.append(tempData)
                #create an event array        
            eva = EventArray(labels = np.array(tempNames,dtype = "S"),
    			     times = np.array(tempTimeStamp)*pq.s,
			     file_origin = self.filename,                            
                             description = "the trigger events (without durations)")       
        return eva
        
       
    def read_epocharray(self,lazy = False, cascade = True, 
                        channel_index = 0,
                        t_start = 0.,
                        segment_duration = 0.):
        """function to read digital timestamps. this function reads the event
        onset and offset and outputs onset and duration. to get only onsets use
        the event array function"""
        if lazy:
            epa = EpochArray(file_origin = self.filename,
                             times=None, durations=None, labels=None)
        else:
            #create temporary empty lists to store data
            tempNames = list()
            tempTimeStamp = list()
            durations = list()
            #get entity from file
            digEntity = self.fd.get_entity(channel_index)
            #transform t_start into index (reading will start from this index) 
            startat = digEntity.get_index_by_time(t_start,0)#zero means closest index to value
            #get the last index to read, using segment duration and t_start
            endat = digEntity.get_index_by_time(float(segment_duration+t_start),-1)#-1 means last index before time       
            
            #run through entity using only odd "i"s 
            for i in range(startat,endat+1,1):
                if i % 2 == 1:
                    #get in which digital bit was the trigger detected
                    tempNames.append(digEntity.label[-8:])
                    #get the time stamps of even events
                    tempData, onOrOff = digEntity.get_data(i-1)
                    #if this was an onset event, save it to the list
                    #on triggered recordings it seems that only onset events are
                    #recorded. On continuous recordings both onset(==1) 
                    #and offset(==255) seem to be recorded
                    #if onOrOff == 1:
                    #append the time stamp to them empty list
                    tempTimeStamp.append(tempData)
                
                    #get time stamps of odd events
                    tempData1, onOrOff = digEntity.get_data(i)
                    #if onOrOff == 255:
                    #pass
                    durations.append(tempData1-tempData)
            epa = EpochArray(file_origin = self.filename,
                                 times = np.array(tempTimeStamp)*pq.s, 
                                 durations = np.array(durations)*pq.s, 
                                 labels = np.array(tempNames,dtype = "S"),
                                 description = "digital events with duration")
            return epa
        
        
