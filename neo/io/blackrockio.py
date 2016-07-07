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
                   


TODO:
  * synchro video
  * tracking
  * Units for spiketrain

"""

import logging
import struct
import datetime
import os

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import (Block, Segment, SpikeTrain, Unit, EventArray,
                      RecordingChannel, RecordingChannelGroup, AnalogSignal)
from neo.io import tools as iotools
import neo.io.tools



class BlackrockIO(BaseIO):
    """
    
    """
    is_readable        = True
    is_writable        = False

    supported_objects  = [Block, Segment, AnalogSignal, RecordingChannelGroup, 
                                                    RecordingChannel, SpikeTrain, Unit]
    readable_objects    = [Block, Segment]

    name               = 'Blackrock'
    extensions = ['ns' + str(i) for i in range(1, 10)] + ['nev', 'sif']
    
    mode = 'file'

    read_params = {
        Block: [('load_waveforms' , { 'value' : True } ) , ],
        Segment : [('load_waveforms' , { 'value' : False } ) ,],
        }


    def __init__(self, filename) :
        """
        """
        BaseIO.__init__(self)
        
        # remove extension because there is bunch of files : nev, ns1, ..., ns5, nsf
        for ext in self.extensions:
            if filename.endswith('.'+ext):
                filename = filename.strip('.'+ext)
        self.filename = filename
        
    def read_block(self, lazy=False, cascade=True, load_waveforms = False):
        """
        """
        # Create block
        bl = Block(file_origin=self.filename)
        if not cascade:
            return bl
        
        seg = self.read_segment(self.filename,lazy=lazy, cascade=cascade,
                            load_waveforms = load_waveforms)
        bl.segments.append(seg)
        neo.io.tools.populate_RecordingChannel(bl, remove_from_annotation=False)
        
        # This create rc and RCG for attaching Units
        rcg0 = bl.recordingchannelgroups[0]
        def find_rc(chan):
            for rc in rcg0.recordingchannels:
                if rc.index==chan:
                    return rc
        for st in seg.spiketrains:
            chan = st.annotations['channel_index']
            rc = find_rc(chan)
            if rc is None:
                rc = RecordingChannel(index = chan)
                rcg0.recordingchannels.append(rc)
                rc.recordingchannelgroups.append(rcg0)
            if len(rc.recordingchannelgroups) == 1:
                rcg = RecordingChannelGroup(name = 'Group {0}'.format(chan))
                rcg.recordingchannels.append(rc)
                rc.recordingchannelgroups.append(rcg)
                bl.recordingchannelgroups.append(rcg)
            else:
                rcg = rc.recordingchannelgroups[1]
            unit = Unit(name = st.name)
            rcg.units.append(unit)
            unit.spiketrains.append(st)
        bl.create_many_to_one_relationship()
        
        return bl

    def read_segment(self, n_start=None, n_stop=None,
                load_waveforms = False, nsx_num = None,
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
        
        filename_nev = self.filename + '.nev'
        if os.path.exists(filename_nev):
            self.read_nev(filename_nev, seg, lazy, cascade, load_waveforms = load_waveforms)
        
        filename_sif = self.filename + '.sif'
        if os.path.exists(filename_sif):
            sif_header = self.read_sif(filename_sif)
        
        for i in range(1,10):
            if nsx_num is not None and nsx_num!=i: continue
            filename_nsx = self.filename + '.ns'+str(i)
            if os.path.exists(filename_nsx):
                self.read_nsx(filename_nsx, seg, lazy, cascade)
        
        return seg

    def read_nev(self, filename_nev, seg, lazy, cascade, load_waveforms = False):
        # basic header
        dt = [('header_id','S8'),
                    ('ver_major','uint8'),
                    ('ver_minor','uint8'),
                    ('additionnal_flag', 'uint16'), # Read flags, currently basically unused
                    ('header_size', 'uint32'), #i.e. index of first data
                    ('packet_size', 'uint32'),# Read number of packet bytes, i.e. byte per sample
                    ('sampling_rate', 'uint32'),# Read time resolution in Hz of time stamps, i.e. data packets
                    ('waveform_sampling_rate', 'uint32'),# Read sampling frequency of waveforms in Hz
                    ('window_datetime', 'S16'),
                    ('application', 'S32'), # 
                    ('comments', 'S256'), # comments
                    ('num_ext_header', 'uint32') #Read number of extended headers
                    
                ]
        nev_header = h = np.fromfile(filename_nev, count = 1, dtype = dt)[0]
        version = '{0}.{1}'.format(h['ver_major'], h['ver_minor'])
        assert h['header_id'].decode('ascii') == 'NEURALEV' or version == '2.1', 'Unsupported version {0}'.format(version)
        version = '{0}.{1}'.format(h['ver_major'], h['ver_minor'])
        seg.annotate(blackrock_version = version)
        seg.rec_datetime = get_window_datetime(nev_header['window_datetime'])
        sr = float(h['sampling_rate'])
        wsr = float(h['waveform_sampling_rate'])
        
        if not cascade:
            return
        
        # extented header
        # this consist in N block with code 8bytes + 24 data bytes
        # the data bytes depend on the code and need to be converted cafilename_nsx, segse by case
        raw_ext_header = np.memmap(filename_nev, offset = np.dtype(dt).itemsize,
                                                dtype = [('code', 'S8'), ('data', 'S24')],  shape = h['num_ext_header'])
        # this is for debuging
        ext_header = { }
        for code, dt_ext in ext_nev_header_codes.items():
            sel = raw_ext_header['code']==code
            ext_header[code] = raw_ext_header[sel].view(dt_ext)
        
        
        # channel label
        neuelbl_header = ext_header['NEUEVLBL']
        # Sometimes when making the channel labels we have only one channel and so must address it differently.
        try:
            channel_labels = dict(zip(neuelbl_header['channel_id'], neuelbl_header['channel_label']))
        except TypeError:
            channel_labels = dict([(neuelbl_header['channel_id'], neuelbl_header['channel_label'])])

        # TODO ext_header['DIGLABEL'] is there only one label ???? because no id in that case
        # TODO ECOMMENT + CCOMMENT for annotations
        # TODO NEUEVFLT for annotations
        
        
        # read data packet and markers
        dt0 =  [('samplepos', 'uint32'),
                    ('id', 'uint16'), 
                    ('value', 'S{0}'.format(h['packet_size']-6)),
            ]
        data = np.memmap( filename_nev, offset = h['header_size'], dtype = dt0)
        all_ids = np.unique(data['id'])
        
        t_start = 0*pq.s
        t_stop = data['samplepos'][-1]/sr*pq.s
        
        
        
        # read event (digital 9+ analog+comment)
        def create_event_array_trig_or_analog(selection, name, labelmode = None):
            if lazy:
                times = [ ]
                labels = np.array([ ], dtype = 'S')
            else:
                times = data_trigger['samplepos'][selection].astype(float)/sr
                if labelmode == 'digital_port':
                    labels = data_trigger['digital_port'][selection].astype('S2')
                elif labelmode is None:
                    label = None
            ev = EventArray(times= times*pq.s,
                            labels= labels,
                            name=name)
            if lazy:
                ev.lazy_shape = np.sum(is_digital)
            seg.eventarrays.append(ev)

        mask = (data['id']==0) 
        dt_trig =  [('samplepos', 'uint32'),
                    ('id', 'uint16'), 
                    ('reason', 'uint8'), 
                    ('reserved0', 'uint8'), 
                    ('digital_port', 'uint16'), 
                    ('reserved1', 'S{0}'.format(h['packet_size']-10)),
                ]
        data_trigger = data.view(dt_trig)[mask]
        # Digital Triggers (PaquetID 0)
        is_digital = (data_trigger ['reason']&1)>0
        create_event_array_trig_or_analog(is_digital, 'Digital trigger', labelmode =  'digital_port' )
        
        # Analog Triggers (PaquetID 0)
        if version in ['2.1', '2.2' ]:
            for i in range(5):
                is_analog = (data_trigger ['reason']&(2**(i+1)))>0
                create_event_array_trig_or_analog(is_analog, 'Analog trigger {0}'.format(i), labelmode = None)
        
        # Comments
        mask = (data['id']==0xFFF) 
        dt_comments = [('samplepos', 'uint32'),
                    ('id', 'uint16'), 
                    ('charset', 'uint8'), 
                    ('reserved0', 'uint8'), 
                    ('color', 'uint32'), 
                    ('comment', 'S{0}'.format(h['packet_size']-12)),
                ]
        data_comments = data.view(dt_comments)[mask]
        if data_comments.size>0:
            if lazy:
                times = [ ]
                labels = [ ]
            else:
                times = data_comments['samplepos'].astype(float)/sr
                labels = data_comments['comment'].astype('S')
            ev = EventArray(times= times*pq.s,
                            labels= labels,
                            name='Comments')
            if lazy:
                ev.lazy_shape = np.sum(is_digital)
            seg.eventarrays.append(ev)
        
        
        # READ Spike channel
        channel_ids = all_ids[(all_ids>0) & (all_ids<=2048)]

        # get the dtype of waveform (this is stupidly complicated)
        if nev_header['additionnal_flag']&0x1:
            #dtype_waveforms = { k:'int16' for k in channel_ids }
            dtype_waveforms = dict( (k,'int16') for k in channel_ids)
        else:
            # there is a code electrodes by electrodes given the approiate dtype
            neuewav_header = ext_header['NEUEVWAV']
            dtype_waveform = dict(zip(neuewav_header['channel_id'], neuewav_header['num_bytes_per_waveform']))
            dtypes_conv = { 0: 'int8', 1 : 'int8', 2: 'int16', 4 : 'int32' }
            #dtype_waveforms = { k:dtypes_conv[v] for k,v in dtype_waveform.items() }
            dtype_waveforms = dict( (k,dtypes_conv[v]) for k,v in dtype_waveform.items() )
        
        dt2 =   [('samplepos', 'uint32'),
                    ('id', 'uint16'), 
                    ('cluster', 'uint8'), 
                    ('reserved0', 'uint8'), 
                    ('waveform','uint8',(h['packet_size']-8, )),
                ]
        data_spike = data.view(dt2)
        
        for channel_id in channel_ids:
            data_spike_chan = data_spike[data['id']==channel_id]
            cluster_ids = np.unique(data_spike_chan['cluster'])
            for cluster_id in cluster_ids:
                if cluster_id==0: 
                    name =  'unclassified'
                elif cluster_id==255:
                    name =  'noise'
                else:
                    name = 'Cluster {0}'.format(cluster_id)
                name = 'Channel {0} '.format(channel_id)+name
                
                data_spike_chan_clus = data_spike_chan[data_spike_chan['cluster']==cluster_id]
                n_spike = data_spike_chan_clus.size
                waveforms, w_sampling_rate, left_sweep = None, None, None
                if lazy:
                    times = [ ]
                else:
                    times = data_spike_chan_clus['samplepos'].astype(float)/sr
                    if load_waveforms:
                        dtype_waveform = dtype_waveforms[channel_id]
                        waveform_size = (h['packet_size']-8)/np.dtype(dtype_waveform).itemsize
                        waveforms = data_spike_chan_clus['waveform'].flatten().view(dtype_waveform)
                        waveforms = waveforms.reshape(n_spike,1, waveform_size)
                        waveforms =waveforms*pq.uV
                        w_sampling_rate = wsr*pq.Hz
                        left_sweep = waveform_size//2/sr*pq.s
                st = SpikeTrain(times =  times*pq.s, name = name,
                                t_start = t_start, t_stop =t_stop,
                                waveforms = waveforms, sampling_rate = w_sampling_rate, left_sweep = left_sweep)
                st.annotate(channel_index = int(channel_id))
                if lazy:
                    st.lazy_shape = n_spike
                seg.spiketrains.append(st)
    
    
    def read_nsx(self, filename_nsx, seg, lazy, cascade):
        # basic header
        dt0 = [('header_id','S8'),
                    ('ver_major','uint8'),
                    ('ver_minor','uint8'),
                    ('header_size', 'uint32'), #i.e. index of first data
                    ('group_label', 'S16'),# Read number of packet bytes, i.e. byte per samplepos
                    ('comments', 'S256'),
                    ('period_ratio', 'uint32'),
                    ('sampling_rate', 'uint32'),
                    ('window_datetime', 'S16'),
                    ('nb_channel', 'uint32'),
                ]
        nsx_header = h = np.fromfile(filename_nsx, count = 1, dtype = dt0)[0]
        version = '{0}.{1}'.format(h['ver_major'], h['ver_minor'])
        seg.annotate(blackrock_version = version)
        seg.rec_datetime = get_window_datetime(nsx_header['window_datetime'])
        nb_channel = h['nb_channel']
        sr = float(h['sampling_rate'])/h['period_ratio']

        if not cascade:
            return
        
        # extended header = channel information
        dt1 = [('header_id','S2'),
                    ('channel_id', 'uint16'),
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
        channels_header = ch= np.memmap(filename_nsx, shape = nb_channel,
                    offset = np.dtype(dt0).itemsize,   dtype = dt1)
        
        # read data
        dt2 = [('header_id','uint8'),
                    ('n_start','uint32'),
                    ('nb_sample','uint32'),
                    ]
        sample_header = sh =  np.memmap(filename_nsx, dtype = dt2, shape = 1,
                        offset = nsx_header['header_size'])[0]
        nb_sample = sample_header['nb_sample']
        data = np.memmap(filename_nsx, dtype = 'int16', shape = (nb_sample, nb_channel),
                        offset = nsx_header['header_size'] +np.dtype(dt2).itemsize )
        
        # create new objects
        for i in range(nb_channel):
            unit = channels_header['units'][i].decode()
            if lazy:
                sig = [ ]
            else:
                sig = data[:,i].astype(float)
                # dig value to physical value
                if ch['max_analog_val'][i] == -ch['min_analog_val'][i] and\
                     ch['max_digital_val'][i] == -ch['min_digital_val'][i]:
                    # when symmetric it is simple
                    sig *= float(ch['max_analog_val'][i])/float(ch['max_digital_val'][i])
                else:
                    # general case
                    sig -= ch['min_digital_val'][i]
                    sig *= float(ch['max_analog_val'][i] - ch['min_analog_val'])/\
                                    float(ch['max_digital_val'][i] - ch['min_digital_val'])
                    sig += float(ch['min_analog_val'][i])
            anasig = AnalogSignal(signal = pq.Quantity(sig,unit, copy = False),
                                                        sampling_rate = sr*pq.Hz,
                                                        t_start = sample_header['n_start']/sr*pq.s,
                                                        name = str(ch['label'][i]),
                                                        channel_index = int(ch['channel_id'][i]))
            if lazy:
                anasig.lazy_shape = nb_sample
            seg.analogsignals.append(anasig)


    def read_sif(self, filename_sif, seg, ):
        pass
        #TODO


def get_window_datetime(buf):
    """This transform a buffer of 16 bytes window datetime type
     n python datetime object
    """

    dt = [('year', 'uint16'), ('mouth', 'uint16'), ('weekday', 'uint16'),
                ('day', 'uint16'), ('hour', 'uint16'), ('min', 'uint16'),
                ('sec', 'uint16'), ('usec', 'uint16'),]
    assert len(buf) == np.dtype(dt).itemsize, 'This buffer do not have the good size for window date'
    d = np.fromstring(buf, dtype = dt)[0]
    thedatetime = datetime.datetime(d['year'], d['mouth'], d['day'],
                d['hour'], d['min'], d['sec'], d['usec'])
    return thedatetime


# the extented header of nev file is composed of 8+24 bytes
# the comments of the 24 bytes depend of the code of 8 bytes
# this is conversion
ext_nev_header_codes = [
        ('ARRAYNME', [('ArrayName', 'S24')] ),
        ('ECOMMENT', [('ExtendedComment', 'S24')] ),
        ('CCOMMENT', [('ExtendedCommentComplement', 'S24')] ),
        ('MAPFILE', [('MapFile', 'S24')] ),
        ('NEUEVWAV', [('channel_id', 'uint16'),# Read electrode number
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
                                    ('spike_width','uint16'), #only for version>=2.3
                                    ('unused', 'S8'),
                                ]),
        ('NEUEVLBL', [('channel_id', 'uint16'),
                                    ('channel_label',  'S16'),
                                    ('unused', 'S6'),
                                ]),
        ('NEUEVFLT', [('channel_id', 'uint16'),
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
ext_nev_header_codes = [ (code, [ ('code', 'S8'),  ]+dt) for code, dt in ext_nev_header_codes ]
ext_nev_header_codes = dict(ext_nev_header_codes)

