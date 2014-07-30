# -*- coding: utf-8 -*-
"""
Class for reading data from from Tucker Davis TTank format.
Terminology:
TDT hold data with tanks (actually a directory). And tanks hold sub block (sub directories).
Tanks correspond to neo.Block and tdt block correspond to neo.Segment.

Note the name Block is ambiguous because it does not refer to same thing in TDT terminilogy and neo.


Depend on:

Supported : Read

Author: sgarcia

"""

import os
import struct
import sys

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal, SpikeTrain, EventArray
from neo.io.tools import iteritems

PY3K = (sys.version_info[0] == 3)


def get_chunks(sizes, offsets, dt, fid):
    sizes = (sizes -10)  * dt.itemsize
    f = np.memmap(fid, mode = 'r', dtype = 'uint8')
    all = [ ]
    for s, o in zip(sizes, offsets):
        all.append(f[o:o+s])
    all = np.concatenate(all)
    return all.view(dt)

class TdtIO(BaseIO):
    """
    Class for reading data from from Tucker Davis TTank format.

    Usage:
        >>> from neo import io
        >>> r = io.TdtIO(dirname='aep_05')
        >>> bl = r.read_block(lazy=False, cascade=True)
        >>> print bl.segments
        [<neo.core.segment.Segment object at 0x1060a4d10>]
        >>> print bl.segments[0].analogsignals
        [<AnalogSignal(array([ 2.18811035,  2.19726562,  2.21252441, ...,  1.33056641,
                1.3458252 ,  1.3671875 ], dtype=float32) * pA, [0.0 s, 191.2832 s], sampling rate: 10000.0 Hz)>]
        >>> print bl.segments[0].eventarrays
        []
    """



    is_readable        = True
    is_writable        = False

    supported_objects  = [Block, Segment , AnalogSignal, EventArray ]
    readable_objects   = [Block]
    writeable_objects  = []

    has_header         = False
    is_streameable     = False

    read_params        = {
                        Block : [
                                ],
                        }

    write_params       = None

    name               = 'TDT'
    extensions          = [ ]

    mode = 'dir'

    def __init__(self , dirname = None) :
        """
        This class read a WinEDR wcp file.

        **Arguments**
        Arguments:
            dirname: path of the TDT tank (a directory)

        """
        BaseIO.__init__(self)
        self.dirname = dirname
        if self.dirname.endswith('/'):
            self.dirname = self.dirname[:-1]

    def read_block(self,
                                        lazy = False,
                                        cascade = True,
                                ):
        bl = Block()
        tankname = os.path.basename(self.dirname)
        bl.file_origin = tankname
        if not cascade : return bl
        for blockname in os.listdir(self.dirname):
            if blockname == 'TempBlk': continue
            subdir = os.path.join(self.dirname,blockname)
            if not os.path.isdir(subdir): continue

            seg = Segment(name = blockname)
            bl.segments.append( seg)


            #TSQ is the global index
            tsq_filename = os.path.join(subdir, tankname+'_'+blockname+'.tsq')
            dt = [('size','int32'),
                        ('evtype','int32'),
                        ('code','S4'),
                        ('channel','uint16'),
                        ('sortcode','uint16'),
                        ('timestamp','float64'),
                        ('eventoffset','int64'),
                        ('dataformat','int32'),
                        ('frequency','float32'),
                    ]
            tsq = np.memmap(tsq_filename, mode = 'r', dtype = dt)
            
            #0x8801: 'EVTYPE_MARK' give the global_start
            global_t_start = tsq[tsq['evtype']==0x8801]['timestamp'][0]
            #print global_t_start, type(global_t_start)

            
            #TEV is the old data file
            if os.path.exists(os.path.join(subdir, tankname+'_'+blockname+'.tev')):
                tev_filename = os.path.join(subdir, tankname+'_'+blockname+'.tev')
            else:
                tev_filename = None


            for type_code, type_label in tdt_event_type:
                mask1 = tsq['evtype']==type_code
                codes = np.unique(tsq[mask1]['code'])
                
                for code in codes:
                    mask2 = mask1 & (tsq['code']==code)
                    channels = np.unique(tsq[mask2]['channel'])
                    
                    for channel in channels:
                        mask3 = mask2 & (tsq['channel']==channel)
                        
                        if type_label in ['EVTYPE_STRON', 'EVTYPE_STROFF']:
                            if lazy:
                                times = [ ]*pq.s
                                labels = np.array([ ], dtype = str)
                            else:
                                times = (tsq[mask3]['timestamp'] - global_t_start) * pq.s
                                labels = tsq[mask3]['eventoffset'].view('float64').astype('S')
                            ea = EventArray(times = times, name = code , channel_index = int(channel), labels = labels)
                            if lazy:
                                ea.lazy_shape = np.sum(mask3)
                            seg.eventarrays.append(ea)
                        
                        elif type_label == 'EVTYPE_SNIP':
                            sortcodes = np.unique(tsq[mask3]['sortcode'])
                            for sortcode in sortcodes:
                                mask4 = mask3 & (tsq['sortcode']==sortcode)
                                nb_spike = np.sum(mask4)
                                sr = tsq[mask4]['frequency'][0]
                                waveformsize = tsq[mask4]['size'][0]-10
                                if lazy:
                                    times = [ ]*pq.s
                                    waveforms = None
                                else:
                                    times = (tsq[mask4]['timestamp'] - global_t_start) * pq.s
                                    dt = np.dtype(data_formats[ tsq[mask3]['dataformat'][0]])                                    
                                    waveforms = get_chunks(tsq[mask4]['size'],tsq[mask4]['eventoffset'],dt,  tev_filename)
                                    waveforms = waveforms.reshape(nb_spike, -1, waveformsize)
                                    waveforms = waveforms * pq.mV
                                if nb_spike>0:
                                 #   t_start = (tsq['timestamp'][0] - global_t_start) * pq.s # this hould work but not
                                    t_start = 0 *pq.s
                                    t_stop = (tsq['timestamp'][-1] - global_t_start) * pq.s
                                    
                                else:
                                    t_start = 0 *pq.s
                                    t_stop = 0 *pq.s
                                st = SpikeTrain(times = times, name = str(sortcode),
                                                                t_start = t_start,
                                                                t_stop = t_stop,
                                                                waveforms = waveforms,
                                                                left_sweep = waveformsize/2./sr * pq.s,
                                                                sampling_rate = sr * pq.Hz,
                                                                )
                                st.annotate(channel_index = channel)
                                if lazy:
                                    st.lazy_shape = nb_spike
                                seg.spiketrains.append(st)
                        
                        elif type_label == 'EVTYPE_STREAM':
                            dt = np.dtype(data_formats[ tsq[mask3]['dataformat'][0]])
                            shape = np.sum(tsq[mask3]['size']-10)
                            sr = tsq[mask3]['frequency'][0]
                            if lazy:
                                signal = [ ]
                            else:
                                if PY3K:
                                    signame = code.decode('ascii')
                                else:
                                    signame = code
                                filename = os.path.join(subdir, tankname+'_'+blockname+'_'+signame+'_ch'+str(channel)+'.sev')
                                if not os.path.exists(filename):
                                    filename = tev_filename
                                signal = get_chunks(tsq[mask3]['size'],tsq[mask3]['eventoffset'],dt,  filename)
                            
                            anasig = AnalogSignal(signal = signal* pq.V,
                                                                    name = code,
                                                                    sampling_rate= sr * pq.Hz,
                                                                    t_start = (tsq[mask3]['timestamp'][0] - global_t_start) * pq.s,
                                                                    channel_index = int(channel))
                            if lazy:
                                anasig.lazy_shape = shape
                            seg.analogsignals.append(anasig)
            bl.create_many_to_one_relationship()
            return bl
            


tdt_event_type = [
   #(0x0,'EVTYPE_UNKNOWN'),
    (0x101, 'EVTYPE_STRON'),
    (0x102,'EVTYPE_STROFF'),
    #(0x201,'EVTYPE_SCALER'),
    (0x8101, 'EVTYPE_STREAM'),
    (0x8201, 'EVTYPE_SNIP'),
    #(0x8801, 'EVTYPE_MARK'),
    ]
 

            

data_formats = {
        0 : np.float32,
        1 : np.int32,
        2 : np.int16,
        3 : np.int8,
        4 : np.float64,
        #~ 5 : ''
        }

