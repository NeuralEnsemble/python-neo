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
import itertools

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal, SpikeTrain, EventArray
from neo.io.tools import iteritems

PY3K = (sys.version_info[0] == 3)


def get_chunks(sizes, offsets, big_array):
    # offsets are octect count
    # sizes are not!!
    # so need this (I really do not knwo why...):
    sizes = (sizes -10)  * 4 #
    all = np.concatenate([ big_array[o:o+s] for s, o in itertools.izip(sizes, offsets) ])
    return all

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

    supported_objects  = [Block, Segment , AnalogSignal, EventArray]
    readable_objects   = [Block, Segment]
    writeable_objects  = []

    has_header         = False
    is_streameable     = False

    read_params        = {
                         Block : [],
                         Segment : []
                         }

    write_params       = None

    name               = 'TDT'
    extensions         = [ ]

    mode = 'dir'

    def __init__(self , dirname=None) :
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

    def read_segment(self, blockname=None, lazy=False, cascade=True):
        """
        Read a single segment from the tank. Note that TDT blocks are Neo
        segments, and TDT tanks are Neo blocks, so here the 'blockname' argument
        refers to the TDT block's name, which will be the Neo segment name.
        """
        if not blockname:
            blockname = os.listdir(self.dirname)[0]

        if blockname == 'TempBlk': return None

        if not self.is_tdtblock(blockname): return None    # if not a tdt block

        subdir = os.path.join(self.dirname, blockname)
        if not os.path.isdir(subdir): return None

        seg = Segment(name=blockname)

        tankname = os.path.basename(self.dirname)

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
        tsq = np.fromfile(tsq_filename, dtype=dt)

        #0x8801: 'EVTYPE_MARK' give the global_start
        global_t_start = tsq[tsq['evtype']==0x8801]['timestamp'][0]

        #TEV is the old data file
        try:
            tev_filename = os.path.join(subdir, tankname+'_'+blockname+'.tev')
            #tev_array = np.memmap(tev_filename, mode = 'r', dtype = 'uint8') # if memory problem use this instead
            tev_array = np.fromfile(tev_filename, dtype='uint8')
        except IOError:
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
                            labels = np.array([ ], dtype=str)
                        else:
                            times = (tsq[mask3]['timestamp'] - global_t_start) * pq.s
                            labels = tsq[mask3]['eventoffset'].view('float64').astype('S')
                        ea = EventArray(times           = times,
                                        name            = code ,
                                        channel_index   = int(channel),
                                        labels          = labels
                                        )
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
                                waveforms = get_chunks(tsq[mask4]['size'],tsq[mask4]['eventoffset'], tev_array).view(dt)
                                waveforms = waveforms.reshape(nb_spike, -1, waveformsize)
                                waveforms = waveforms * pq.mV
                            if nb_spike > 0:
                             #   t_start = (tsq['timestamp'][0] - global_t_start) * pq.s # this hould work but not
                                t_start = 0 *pq.s
                                t_stop = (tsq['timestamp'][-1] - global_t_start) * pq.s

                            else:
                                t_start = 0 *pq.s
                                t_stop = 0 *pq.s
                            st = SpikeTrain(times           = times,
                                            name            = 'Chan{0} Code{1}'.format(channel,sortcode),
                                            t_start         = t_start,
                                            t_stop          = t_stop,
                                            waveforms       = waveforms,
                                            left_sweep      = waveformsize/2./sr * pq.s,
                                            sampling_rate   = sr * pq.Hz,
                                            )
                            st.annotate(channel_index=channel)
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
                            sev_filename = os.path.join(subdir, tankname+'_'+blockname+'_'+signame+'_ch'+str(channel)+'.sev')
                            try:
                                #sig_array = np.memmap(sev_filename, mode = 'r', dtype = 'uint8') # if memory problem use this instead
                                sig_array = np.fromfile(sev_filename, dtype='uint8')
                            except IOError:
                                sig_array = tev_array
                            signal = get_chunks(tsq[mask3]['size'],tsq[mask3]['eventoffset'],  sig_array).view(dt)

                        anasig = AnalogSignal(signal        = signal* pq.V,
                                              name          = '{0} {1}'.format(code, channel),
                                              sampling_rate = sr * pq.Hz,
                                              t_start       = (tsq[mask3]['timestamp'][0] - global_t_start) * pq.s,
                                              channel_index = int(channel)
                                              )
                        if lazy:
                            anasig.lazy_shape = shape
                        seg.analogsignals.append(anasig)
        return seg

    def read_block(self, lazy=False, cascade=True):
        bl = Block()
        tankname = os.path.basename(self.dirname)
        bl.file_origin = tankname

        if not cascade : return bl

        for blockname in os.listdir(self.dirname):
            if self.is_tdtblock(blockname):    # if the folder is a tdt block
                seg = self.read_segment(blockname, lazy, cascade)
                bl.segments.append(seg)

        bl.create_many_to_one_relationship()
        return bl


    # to determine if this folder is a TDT block, based on the extension of the files inside it
    # to deal with unexpected files in the tank, e.g. .DS_Store on Mac machines
    def is_tdtblock(self, blockname):

        file_ext = list()
        blockpath = os.path.join(self.dirname, blockname)  # get block path
        if os.path.isdir(blockpath):
            for file in os.listdir( blockpath ):   # for every file, get extension, convert to lowercase and append
                file_ext.append( os.path.splitext( file )[1].lower() )

        file_ext = set(file_ext)
        tdt_ext  = set(['.tbk', '.tdx', '.tev', '.tsq'])
        if file_ext >= tdt_ext:    # if containing all the necessary files
            return True
        else:
            return False

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
        }
