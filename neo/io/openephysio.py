# -*- coding: utf-8 -*-
"""
Class for reading OpenEphys continuous data from a folder.

Generates a :class:`Segment` or a :class:`Block` with a
sinusoidal :class:`AnalogSignal`

Depends on: analysis-tools/OpenEphys.py

Supported: Read

Acknowledgements: :ref:`neo_io_API` sgarcia, open-ephys

Author: Cristian Tatarau, Charite Berlin, Experimental Psychiatry Group
"""

# needed for python 3 compatibility
from __future__ import absolute_import
from __future__ import division

# note neo.core needs only numpy and quantities
import numpy as np
import quantities as pq
import datetime as dt

# I need to subclass BaseIO
from neo.io.baseio import BaseIO

# to import from core
from neo.core import Block, Segment, AnalogSignal, SpikeTrain #, EventArray

# need to link to open-ephys/analysis-tools
import re, os, sys
from neo.io import OpenEphys as OEIO

# import Tkinter as tk

# I need to subclass BaseIO
class Open_Ephys_IO(BaseIO):
    """
    Class for reading OpenEphys data from a folder
    """

    is_readable = True # This class can only read data
    is_writable = False # write is not supported

    # This class is able to directly or indirectly handle the following objects
    supported_objects  = [ Block, Segment , AnalogSignal ]

    # This class can return either a Block or a Segment
    # The first one is the default ( self.read )
    # These lists should go from highest object to lowest object because
    # common_io_test assumes it.
    readable_objects    = [ Block, Segment , AnalogSignal ]
    # This class is not able to write objects
    writeable_objects   = [ ]

    has_header         = False
    is_streameable     = False


    # # This is for GUI stuff : a definition for parameters when reading.
    # # This dict should be keyed by object (`Block`). Each entry is a list
    # # of tuple. The first entry in each tuple is the parameter name. The
    # # second entry is a dict with keys 'value' (for default value),
    # # and 'label' (for a descriptive name).
    # # Note that if the highest-level object requires parameters,
    # # common_io_test will be skipped.
    # read_params = {
    #     Segment : [
    #         ('segment_duration',
    #             {'value' : 0., 'label' : 'Segment size (s.)'}),
    #         ('num_analogsignal',
    #             {'value' : 0, 'label' : 'Number of recording points'}),
    #         ],
    #     }

    # do not supported write so no GUI stuff
    write_params       = None
    name               = 'Open Ephys IO'
    extensions          = [ 'nof' ]
    mode = 'dir'

    def __init__(self , dirname) :
        """
        Arguments:
            pathname : the file or dir pathname
        """
        BaseIO.__init__(self)
        self.dirname = dirname


    def get_number_of_recording_sessions(self, dirname):
        return OEIO.get_number_of_recording_sessions(dirname)

    # loads one recording session from the current file
    def _load_recording_session(self, i):
        data, filelist=OEIO.loadFolderToArray(self.dirname, channels='all', recording= i, dtype=float)
        if data.size == 0:
            print("Folder ", self.dirname, 'recording session', i,  "is empty or can't be read")
            return None, None, 0
        if len(data.shape) ==1:
            # just one single channel
            n = 1
        else:
            # more than 1 channel
            n = data.shape[1]
        return data, filelist, n

    def read_block(self,
                     # the 2 first keyword arguments are imposed by neo.io API
                     lazy = False,
                     cascade = True,
                    recording_session = None
                    ):
        """
        In this IO read by default a Block with one or many Segments.

        If recording_session is None, all sessions contained in the folder will be loaded,
        if not None, it loads the specified session. Counting starts at 1, NOT 0 based!
        The OpenEphys format saves continuous signals as separate files with increasing suffixes
        if measurement stops and restarts. We define as recording session one contiguous data set.
        The time between sessions will be zero-padded

        The input variable cascade has no effect in this implementation
        """

        # read number or recording session in the current folder
        N_sessions = OEIO.get_number_of_recording_sessions(self.dirname)

        # ask which session to load
        # tkroot = tk._Tk()
        # usr_rbutton = tk.IntVar()
        # usr_load_all = tk.IntVar()
        # if N_sessions > 1:
        #     # ask which recording session to load
        #     tk.Label(tkroot, text="Select the recording sessions you want to load:").pack()
        #     tk.Label(tkroot, text=self.dirname).pack()
        #     tk.Checkbutton(tkroot, text='load all recording sessions', variable=usr_load_all).pack()
        #     tk.Label(tkroot, text="load only one recording session:").pack()
        #     for i in range(1, N_sessions+1):
        #         tk.Radiobutton(tkroot, text=i, variable=usr_rbutton, value=i).pack()
        #     tk.Button(tkroot, text='Load', command=tkroot.quit).pack()
        #     tkroot.mainloop()
        # # tkroot.destroy()
        # del tkroot

        headers = []
        # load header of first recording session
        print('reading first header...')
        # header=OEIO.get_header_from_folder(self.dirname, recording= 1)
        # if header != None:
        #     headers.append(header)
        # load headers of next recording sessions
        for i in range(1, N_sessions+1):
            print('reading header...', i)
            header = OEIO.get_header_from_folder(self.dirname, recording=i)
            if header != None:
                headers.append(header)

        header = headers[0]         # use first header for basic info and time
        sr = headers[i - 2]['sampleRate']   # get sr and use it for all recording sessions

        # create an empty block
        block = Block( name = header['date_created'],
                       description=header['format'],
                       file_datetime= dt.datetime.strptime(header['date_created'], "'%d-%b-%Y %H%M%S'"),
                       rec_datetime=dt.datetime.strptime(header['date_created'], "'%d-%b-%Y %H%M%S'"),
                       file_origin=self.dirname)
        seg = Segment(name = header['date_created'],
                       description=header['format'],
                       file_datetime= dt.datetime.strptime(header['date_created'], "'%d-%b-%Y %H%M%S'"),
                       rec_datetime=dt.datetime.strptime(header['date_created'], "'%d-%b-%Y %H%M%S'"),
                       file_origin=self.dirname)

        # if usr_load_all.get() == 0:
        if recording_session is not None:
            if N_sessions == 1:
                i = None
            else:
                i = recording_session
            # else:
            #     i = usr_rbutton.get()
            print('--- reading recording session', i)
            data, filelist, n = self._load_recording_session(i)
            if n == 0:
                return
            data_final = data
        else:
            duration = 0
            # if usr_load_all.get() == 1:
            # read all selected recording sessions and concatenate with zero paddings
            data_list = []
            for i in range(1, N_sessions+1):    # counting starts at 1!
                # read nested analosignal
                print('--- reading recording session', i)
                data, filelist, n = self._load_recording_session(i)
                if n == 0:
                    return
                if i >= 2:
                    # stack rec sessions with zero padding
                    # get time between rec sessions and sr, index -2 and -1 because of different counting! recording starts at 1, python array at 0!
                    rec_datetime_prev=dt.datetime.strptime(headers[i-2]['date_created'], "'%d-%b-%Y %H%M%S'")
                    rec_datetime_curr=dt.datetime.strptime(headers[i-1]  ['date_created'], "'%d-%b-%Y %H%M%S'")
                    t = rec_datetime_curr - rec_datetime_prev
                    # length of zero padding between recording sessions
                    nz = np.floor(t.total_seconds()*sr) - duration
                    # np array for filling the gap
                    z = np.zeros((int(nz), n))
                    data_list.append(z)
                duration = len(data) # length of recording session
                data_list.append(data)

            # stack all together
            data_final = np.vstack(data_list)
        for i in range(n):
            name = re.findall(r'CH\d*', filelist[i])[0]
            ana = AnalogSignal(signal           = data_final[:,i],
                               units            = pq.microvolt,
                               sampling_rate    = headers[0]['sampleRate']*pq.Hz,
                               name             = name,
                               channel_index    = i,
                               description      = headers[0]['format'],
                               file_origin      = os.path.join(self.dirname,
                                                         filelist[i]))
            seg.analogsignals += [ ana ]
        block.segments.append(seg)
        block.create_many_to_one_relationship()
        return block

