# -*- coding: utf-8 -*-
"""
Class for "reading" data from *.mcd files (files generate with the MCRack
software, distributed by Multichannelsystems company - Reutlingen, Germany).
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

#check to see if the neuroshare bindings are properly imported    
try:
    import neuroshare as ns
except ImportError as err:
    print('\n neuroshare library not found, loading data will not work!' )
    print('\n be sure to install the library found at:')
    print('\n www.http://pythonhosted.org/neuroshare/')

else:
    print('neuroshare library successfully imported')


#import BaseIO
from neo.io.baseio import BaseIO

#import objects from neo.core
from neo.core import Segment, AnalogSignal, SpikeTrain, EventArray

# some tools to finalize the hierachy
from neo.io.tools import create_many_to_one_relationship


# create an object based on BaseIO
class MultichannelIO(BaseIO):
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

 
#    # This is for GUI stuff : a definition for parameters when reading.
#    # This dict should be keyed by object (`Block`). Each entry is a list
#    # of tuple. The first entry in each tuple is the parameter name. The
#    # second entry is a dict with keys 'value' (for default value),
#    # and 'label' (for a descriptive name).
#    # Note that if the highest-level object requires parameters,
#    # common_io_test will be skipped.
#    read_params = {
#        Segment : [
#            ('segment_duration',
#                {'value' : 15., 'label' : 'Segment size (s.)'}),
#            ('num_analogsignal',
#                {'value' : 8, 'label' : 'Number of recording points'}),
#            ('num_spiketrain_by_channel',
#                {'value' : 3, 'label' : 'Num of spiketrains'}),
#            ],
#        }
#
#    # do not supported write so no GUI stuff
#    write_params       = None
#
#    name               = 'example'
#
#    extensions          = [ 'nof' ]
#
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
                #
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
                     segment_duration = 15.,
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
                        channel_index = self.metadata['spkChanId'][i])
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
                        t_start = -1,
                        channel_index = 0,):
        """
        Function to read in spike trains. This API still does not support read in of
        specific channels as they are recorded. rather the fuunction gets the entity set
        by 'channel_index' which is set in the __init__ function (all spike channels)
        """

                

        #sampling rate
        sr = self.metadata['sampRate']
        
        # create a list to store spiketrain times
        times = list() 
        #create a spike train object
        spiketr = SpikeTrain(times, t_start = t_start*pq.s, 
                             t_stop = (t_start+segment_duration)*pq.s ,
                             units = pq.s,)
        
        if lazy:
            # we add the attribute lazy_shape with the size if lazy
            spiketr.lazy_shape = (40)
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
            
            #set data into spike train object
            #name
            spiketr.name = 'spikes from electrode'+tempSpks.label[-3:],
            #cutouts
            spiketr.waveforms = waveforms*pq.volt
            #samp rate
            spiketr.sampling_rate = sr * pq.Hz
            #file origin
            spiketr.file_origin = self.filename
            # the channel index used
            spiketr.annotate(channel_index = channel_index)
            
        return spiketr

    def read_eventarray(self,lazy = False, cascade = True,channel_index = 0):
        """function to read digital timestamps. this function only reads the event
        onset and disconsiders its duration. to get digital event durations, use 
        the epoch function (to be implemented)."""
        #create an event array        
        eva = EventArray()    
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
            tempData,_ = trigEntity.get_data(i)
            #append the time stamp to them empty list
            tempTimeStamp.append(tempData)
        #set event object variables
        #filename
        eva.file_origin = self.filename
        #which digital bits were detected
        eva.labels = tempNames
        #the time stamps
        eva.times = tempTimeStamp*pq.sec
#        eva.description('here are stored all the trigger events'+
#                        '(without their durations) as detected by '+
#                        'the Trigger detector tool in MCRack')
        return eva
            