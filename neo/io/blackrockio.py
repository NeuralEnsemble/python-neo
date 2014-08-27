# -*- coding: utf-8 -*-
"""
Module for reading binary file from Blackrock format.

This work is based on:
  * Chris rodger first version
  * Michael Denker, Lyuba Zehl second version.

It support reading only.
his IO is able to read:
  * the nev file with contain spikes
  * ns1, ns2, .., ns6 files that contains signals at differents sampling rates



The neural data channels
            are 1 - 128. The analog inputs are 129 - 144.
            
spike- and event-data; 30000 Hz
"ns1": "analog data: 500 Hz",
                   "ns2": "analog data: 1000 Hz",
                   "ns3": "analog data: 2000 Hz",
                   "ns4": "analog data: 10000 Hz",
                   "ns5": "analog data: 30000 Hz",
                   "ns6": "analog data: 30000 Hz (no digital filter)"
"""

import logging
import struct
import datetime
import os

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import (Block, Segment, SpikeTrain, Unit,
                      RecordingChannel, RecordingChannelGroup, AnalogSignal)
from neo.io import tools


class BlackrockIO(BaseIO):
    """
    
    """
    is_readable        = True
    is_writable        = False

    supported_objects  = [Block, Segment, AnalogSignal, RecordingChannelGroup, 
                                                    RecordingChannel, SpikeTrain, Unit]
    readable_objects    = [Block, Segment]

    name               = 'Blackrock'
    extensions = ['ns' + str(i) for i in range(1, 10)] + ['nev']
    
    mode = 'file'

    read_params = {
        Block: [ ],
        Segment : [],
        }


    def __init__(self, filename) :
        """
        """
        BaseIO.__init__(self)
        
        for ext in self.extensions:
            if filename.endswith('.'+ext):
                filename = filename.strip('.'+ext)
        self.filename = filename
        
        
        #assiciate all files ( nev nsX )
        for nsx_i in range(0, 10):
            filename_nsx = self.filename+ '.ns' + str(nsx_i)
            if os.path.exists(filename_nsx):
                pass
        
        filename_nev = self.filename + '.nev'
        if os.path.exists(filename_nev):
            pass
    
    def read_block(self, lazy=False, cascade=True, load_waveform = False):
        """
        """


        # Create block
        block = Block(file_origin=self.filename)

        if not cascade:
            return block
        
        seg = self.read_segment(self.filename, load_waveform = load_waveform)
        block.segments.append(seg)
        
        return block

    def read_segment(self, n_start=None, n_stop=None,
                load_waveform = False, nsx_num = None,
                lazy=False, cascade=True):
        """Reads one Segment.

        The Segment will contain one AnalogSignal for each channel
        and will go from n_start to n_stop (in samples).

        Arguments:
            n_start : time in samples that the Segment begins
            n_stop : time in samples that the Segment ends

        Python indexing is used, so n_stop is not inclusive.

        Returns a Segment object containing the data.
        """
        
        seg = Segment(file_origin=self.filename)
        if not cascade:
            return seg
        
        
        #~ filename_nev = self.filename + '.nev'
        #~ if os.path.exists(filename_nev):
            #~ self.read_nev(filename_nev, seg, load_waveform = load_waveform)
        
        #~ filename_sif = self.filename + '.sif'
        #~ if os.path.exists(filename_sif):
            #~ sif_header = self.read_sif(filename_sif)
        
        for i in range(1,10):
            if nsx_num is not None and nsx_num!=i: continue
            filename_nsx = self.filename + '.ns'+str(i)
            if os.path.exists(filename_nsx):
                self.read_nsx(filename_nsx, seg)
        
        

        return seg

    def read_nev(self, filename_nev, seg, load_waveform = False):
        
        # basic hedaer
        dt = [('header_id','S8'),
                    ('ver_major','uint8'),
                    ('ver_minor','uint8'),
                    ('additionnal_flag', 'uint16'), # Read flags, currently basically unused
                    ('header_size', 'uint32'), #i.e. index of first data
                    ('packet_size', 'uint32'),# Read number of packet bytes, i.e. byte per timestamp
                    ('sampling_rate', 'uint32'),# Read time resolution in Hz of time stamps, i.e. data packets
                    ('waveform_res', 'uint32'),# Read sampling frequency of waveforms in Hz
                    ('datetime_year', 'uint16'),
                    ('datetime_mouth', 'uint16'),
                    ('datetime_weekday', 'uint16'),
                    ('datetime_day', 'uint16'),
                    ('datetime_hour', 'uint16'),
                    ('datetime_min', 'uint16'),
                    ('datetime_sec', 'uint16'),
                    ('datetime_usec', 'uint16'),
                    ('application', 'S16'), # 
                    ('comments', 'S256'), # comments
                    ('num_ext_header', 'uint32') #Read number of extended headers
                    
                ]
        nev_header = h = np.fromfile(filename_nev, count = 1, dtype = dt)[0]
        version = '{}.{}'.format(h['ver_major'], h['ver_minor'])
        assert  h['header_id'] == 'NEURALEV' or version=='2.1', 'Unsupported version {}'.format(version)
        
        print h['sampling_rate']
        
        # extented header
        # this consist in N block with code 8bytes + 24 data bytes
        # the data bytes depend on the code and need to be converted cafilename_nsx, segse by case
        offset_ext_header =  np.dtype(dt).itemsize
        n = h['num_ext_header']
        print n
        ext_header = np.memmap(filename_nev, offset = offset_ext_header,
                                                dtype = [('code', 'S8'), ('data', 'S24')],  shape = n)
        
        ext_header_codes = [
                ('ARRAYNME', [('ArrayName', 'S24')] ),
                ('ECOMMENT', [('ExtendedComment', 'S24')] ),
                ('CCOMMENT', [('ExtendedCommentComplement', 'S24')] ),
                ('MAPFILE', [('MapFile', 'S24')] ),
                ('NEUEVWAV', [('electrode_id', 'uint16'),# Read electrode number
                                            ('connector_id', 'uint8'),# Read physical connector (bank A-D)
                                            ('connector_pin', 'uint8'),# Read connector sampling_ratepin (pin number on connector, 1-37)
                                            ('digitization_factor', 'uint16'),# Read digitization factor in nV per LSB step
                                            ('energy_threshold', 'uint16'),# Read energy threshold in nV per LSB step
                                            ('amp_threshold_high', 'int16'),# Read high threshold in µV
                                            ('amp_threshold_low', 'int16'),# Read low threshold in µV
                                            ('num_sorted_units', 'uint8'),# Read number of sorted units
                                            ('num_bytes_per_waveform', 'uint8'),# Read number of bytes per waveform sample
                                                                                                                # 0 or 1 both imply 1 byte, so convert a 0 to 1 for
                                                                                                                # simplification
                                            ('unused', 'S10'),
                                        ]),
                ('NEUEVLBL', [('electrode_id', 'uint16'),
                                            ('electrode_label',  'S16'),
                                            ('unused', 'S6'),
                                        ]),
                ('NEUEVFLT', [('electrode_id', 'uint16'),
                                            ('hi_freq_corner',  'uint32'),
                                            ('hi_freq_order',  'uint32'),
                                            ('hi_freq_type',  'uint16'), #0=None 1=Butterworth
                                            ('lo_freq_corner',  'uint32'),
                                            ('lo_freq_order',  'uint32'),
                                            ('lo_freq_type',  'uint16'), #0=None 1=Butterworth
                                            ('unused', 'S2'),
                                        ]),
                ('DIGLABEL', [('digital_channel_label', 'S16'),# Read name of digital
                                            ('digital_channel_type',  'uint8'), # 0=serial, 1=parallel
                                            ('unused', 'S7'),
                                        ]),
                ('NSASEXEV', [('PeriodicPacketGenerator', 'uint16'),# Read frequency of periodic packet generation
                                            ('DigitialChannelEnable',  'uint8'), # Read if digital input triggers events
                                            ('AnalogChannel',  [ ('enable', 'uint8') , ('edge_detec', 'uint16')], (5,) ), # Read if analog input triggers events
                                            ('unused', 'S6'),
                                        ]),                
            ]
        
        #TODO            
        #for code, dt in ext_header_codes:
        #    print code, np.dtype(dt).itemsize
        #    sel = ext_header['code']==code
        #    ext_header['code'][sel].astype(dt)
        
        # read data packet and markers
        
        
        dt0 =  [('timestamp', 'uint32'),
                    ('id', 'uint16'), 
                    ('value', 'S{}'.format(h['packet_size']-6)),
            ]
        data = np.memmap( filename_nev, offset = h['header_size'], dtype = dt0)
        all_ids = np.unique(data['id'])
        print data.shape
        
        #Triggers PaquetID 0
        mask = data['id']==0
        dt1 =  [('timestamp', 'uint32'),
                    ('id', 'uint16'), 
                    ('reason', 'uint8'), 
                    ('reserved0', 'uint8'), 
                    ('digital_port', 'uint16'), 
                    ('reserved1', 'S{}'.format(h['packet_size']-10)),
                ]
        data_trigger = data.view(dt1)[mask]
        print data_trigger.shape
        #~ is_digital = (data_trigger ['reason']&1)>0
        #~ for i in range(16):
            #~ #digital channel
            #~ select = ((data_trigger['digital_port'] & (2**i))>0) & is_digital
            #~ times = data_trigger['timestamp'][select]
            #~ #TODO create EventArray
        #~ is_serial = (data_trigger ['reason']&128)>0
        
        
         # TODO deal with:additionnal_flag
        
        # Spike channel
        waveform_size = h['packet_size']-8
        
        dt2 =   [('timestamp', 'uint32'),
                    ('id', 'uint16'), 
                    ('cluster', 'uint8'), 
                    ('reserved0', 'uint8'), 
                    ('waveform', 'S{}'.format(waveform_size)),
                ]
        data_spike = data.view(dt2)
        channel_ids = all_ids[all_ids>0] & all_ids[all_ids<=2048]
        for channel_id in channel_ids:
            print 'channel_id', channel_id
            data_spike_chan = data_spike[data['id']==channel_id]
            cluster_ids = np.unique(data_spike_chan['cluster'])
            for cluster_id in cluster_ids:
                print 'cluster_id', cluster_id
                if cluster_id==0: 
                    name =  'unclassified'
                elif cluster_id==255:
                    name =  'noise'
                else:
                    name = 'Cluster {}'.format(cluster_id)
                name = 'Channel {} '.format(channel_id)+name
                
                data_spike_chan_clus = data_spike_chan[data_spike_chan['cluster']==cluster_id]
                times = data_spike_chan_clus['timestamp']
                n_spike = data_spike_chan_clus.size
                if load_waveform:
                    waveforms = data_spike_chan_clus['waveform'].view('int8').reshape(n_spike, waveform_size)
                    waveforms = waveforms.transpose()
                    waveforms = waveforms[:,None,:] #ad  one axis because mono electrode
                    waveforms =waveforms*pq.mV #FIXME: unit of waveform
                    sampling_rate = 30000*pq.Hz
                    left_sweep = int(waveform_size/2)/30000.*pq.s
                else:
                    waveforms = None
                    sampling_rate = None
                    left_sweep=  None
                st = SpikeTrain(times =  times*pq.s, name = name,
                                t_start = 0*pq.s, t_stop = max(times)*pq.s,
                                waveform = waveforms, sampling_rate = sampling_rate, left_sweep = left_sweep)
                print st
                seg.spiketrains.append(st)
                print len(seg.spiketrains)
    
    
    def read_nsx(self, filename_nsx, seg):
        
        # basic header
        dt0 = [('header_id','S8'),
                    ('ver_major','uint8'),
                    ('ver_minor','uint8'),
                    ('header_size', 'uint32'), #i.e. index of first data
                    ('group_label', 'S16'),# Read number of packet bytes, i.e. byte per timestamp
                    ('comments', 'S256'),
                    ('period_ratio', 'uint32'),
                    ('sampling_rate', 'uint32'),
                    ('datetime_year', 'uint16'),
                    ('datetime_mouth', 'uint16'),
                    ('datetime_weekday', 'uint16'),
                    ('datetime_day', 'uint16'),
                    ('datetime_hour', 'uint16'),
                    ('datetime_min', 'uint16'),
                    ('datetime_sec', 'uint16'),
                    ('datetime_usec', 'uint16'),
                    ('nb_channel', 'uint32'),
                ]
        nsx_header = h = np.fromfile(filename_nsx, count = 1, dtype = dt0)[0]
        version = '{}.{}'.format(h['ver_major'], h['ver_minor'])
        print h
        
        n = h['nb_channel']
        #extented header
        dt1 = [('header_id','S2'),
                    ('electrode_id', 'uint16'),
                    ('label', 'S16'),
                    ('connector_id', 'uint8'),
                    ('connector_pin', 'uint8'),
                    ('min_digital_val', 'int16'),
                    ('max_digital_val', 'int16'),
                    ('min_analog_val', 'int16'),
                    ('max_analog_val', 'int16'),
                    ('units', 'S16'),
                    ('hi_freq_corner',  'uint32'),
                    ('hi_freq_order',  'uint32'),
                    ('hi_freq_type',  'uint16'), #0=None 1=Butterworth
                    ('lo_freq_corner',  'uint32'),
                    ('lo_freq_order',  'uint32'),
                    ('lo_freq_type',  'uint16'), #0=None 1=Butterworth
            ]
        channels_header = np.memmap(filename_nsx, shape = n,
                    offset = np.dtype(dt1).itemsize,   dtype = dt1)
        
        #data
        dt2 = [('header_id','uint8'),
                    ('timestamp','uint32'),
                    ('nb_sample','uint32'),
                    ('data_channels','uint16', n),
                    ]
        print nsx_header['header_size']
        print np.dtype(dt2).itemsize
        data = np.memmap(filename_nsx, dtype = dt2,
                        offset = nsx_header['header_size'])
        print data.shape
        print np.unique(data['header_id'])
        



    def read_sif(self, filename_sif, seg):
        pass
        #TODO
    
        
    
        




