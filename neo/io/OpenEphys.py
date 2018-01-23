# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 15:18:38 2014

@author: Dan Denman and Josh Siegle

Loads .continuous, .events, and .spikes files saved from the Open Ephys GUI

Usage:
    import OpenEphys
    data = OpenEphys.load(pathToFile) # returns a dict with data, timestamps, etc.

"""

import os
import numpy as np
import scipy.signal
import scipy.io
import time
import struct
import json
from copy import deepcopy
import re
import math

# constants for pre-allocating matrices:
MAX_NUMBER_OF_SPIKES = 1e6
MAX_NUMBER_OF_EVENTS = 1e6

def load(filepath):
    
    # redirects to code for individual file types
    if 'continuous' in filepath:
        data = loadContinuous(filepath)
    elif 'spikes' in filepath:
        data = loadSpikes(filepath)
    elif 'events' in filepath:
        data = loadEvents(filepath)
    else:
        raise Exception("Not a recognized file type. Please input a .continuous, .spikes, or .events file")
        
    return data

def loadFolder(folderpath,**kwargs):

    # load all continuous files in a folder

    data = { }

    # load all continuous files in a folder  
    if 'channels' in kwargs.keys():
        filelist = ['100_CH'+x+'.continuous' for x in map(str,kwargs['channels'])]
    else:
        filelist = os.listdir(folderpath)   

    t0 = time.time()
    numFiles = 0
    
    for i, f in enumerate(filelist):
        if '.continuous' in f:
            data[f.replace('.continuous','')] = loadContinuous(os.path.join(folderpath, f))
            numFiles += 1

    print(''.join(('Avg. Load Time: ', str((time.time() - t0)/numFiles),' sec')))
    print(''.join(('Total Load Time: ', str((time.time() - t0)),' sec')))
            
    return data

def loadFolderToArray(folderpath, channels='all', dtype=float, 
    source='100', recording=None, start_record=None, stop_record=None,
    verbose=True):
    """Load the neural data files in a folder to a single array.
    
    By default, all channels in the folder are loaded in numerical order.
    
    Args:
        folderpath : string, path to folder containing OpenEphys files
        channels : list of channel numbers to read
            If 'all', then all channels are loaded in numerical order
        dtype : float or np.int16
            If float, then the data will be multiplied by bitVolts to convert
            to microvolts. This increases the memory required by 4 times.
        source :
        recording : int, or None
            Multiple recordings in the same folder are suffixed with an
            incrementing label. For the first or only recording, leave this as
            None. Otherwise, specify an integer.
        start_record, stop_record : the first and last record to read from
            each file. This is converted into an appropriate number of samples
            and passed to loadContinuous. Python indexing is used, so 
            `stop_record` is not inclusive. If `start_record` is None, 
            start at the beginning; if `stop_record` is None, read to the end.
        verbose : print status updateds
    
    Returns: numpy array of shape (n_samples, n_channels)
    """
    # Get list of files
    # filelist = get_filelist(folderpath, source, channels, recording=None)
    filelist = get_filelist(folderpath, source, channels, recording)

    # Keep track of the time taken
    t0 = time.time()

    # Get the header info and use this to set start_record and stop_record
    header = get_header_from_folder(folderpath, filelist)
    if start_record is None:
        start_record = 0
    if stop_record is None:
        stop_record = header['n_records']

    # Extract each channel in order
    arr_l = []
    for filename in filelist:
        arr = loadContinuous(os.path.join(folderpath, filename), dtype,
            start_record=start_record, stop_record=stop_record,
            verbose=verbose)['data']
        arr_l.append(arr)
    
    # Concatenate into an array of shape (n_samples, n_channels)
    data_array = np.transpose(arr_l)
    
    if verbose:
        time_taken = time.time() - t0
        print('Avg. Load Time: %0.3f sec' % (time_taken / len(filelist)))
        print('Total Load Time: %0.3f sec' % time_taken)

    return data_array, filelist


def loadContinuous(filepath, dtype=float, verbose=True, 
    start_record=None, stop_record=None, ignore_last_record=True):
    """Load continuous data from a single channel in the file `filepath`.
    
    This is intended to be mostly compatible with the previous version.
    The differences are:
    - Ability to specify start and stop records
    - Converts numeric data in the header from string to numeric data types
    - Does not rely on a predefined maximum data size
    - Does not necessarily drop the last record, which is usually incomplete
    - Uses the block length that is specified in the header, instead of
        hardcoding it.
    - Returns timestamps and recordNumbers as int instead of float
    - Tests the record metadata (N and record marker) for internal consistency

    The OpenEphys file format breaks the data stream into "records", 
    typically of length 1024 samples. There is only one timestamp per record.

    Args:
        filepath : string, path to file to load
        dtype : float or np.int16
            If float, then the data will be multiplied by bitVolts to convert
            to microvolts. This increases the memory required by 4 times.
        verbose : whether to print debugging messages
        start_record, stop_record : indices that control how much data
            is read and returned. Pythonic indexing is used,
            so `stop_record` is not inclusive. If `start` is None, reading
            begins at the beginning; if `stop` is None, reading continues
            until the end.
        ignore_last_record : The last record in the file is almost always
            incomplete (padded with zeros). By default it is ignored, for
            compatibility with the old version of this function.

    Returns: dict, with following keys
        data : array of samples of data
        header : the header info, as returned by readHeader
        timestamps : the timestamps of each record of data that was read
        recordingNumber : the recording number of each record of data that
            was read. The length is the same as `timestamps`.
    """
    if dtype not in [float, np.int16]:
        raise ValueError("Invalid data type. Must be float or np.int16")

    if verbose:
        print("Loading continuous data from " + filepath)

    """Here is the OpenEphys file format:
    'each record contains one 64-bit timestamp, one 16-bit sample 
    count (N), 1 uint16 recordingNumber, N 16-bit samples, and 
    one 10-byte record marker (0 1 2 3 4 5 6 7 8 255)'
    Thus each record has size 2*N + 22 bytes.
    """
    # This is what the record marker should look like
    spec_record_marker = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255])

    # Lists for data that's read
    timestamps = []
    recordingNumbers = []
    samples = []
    samples_read = 0
    records_read = 0
    
    # Open the file
    with open(filepath, 'rb') as f:
        # Read header info, file length, and number of records
        header = readHeader(f)
        record_length_bytes = 2 * header['blockLength'] + 22
        fileLength = os.fstat(f.fileno()).st_size
        n_records = get_number_of_records(filepath)
        
        # Use this to set start and stop records if not specified
        if start_record is None:
            start_record = 0
        if stop_record is None:
            stop_record = n_records
        
        # We'll stop reading after this many records are read
        n_records_to_read = stop_record - start_record
        
        # Seek to the start location, relative to the current position
        # right after the header.
        f.seek(record_length_bytes * start_record, 1)
        
        # Keep reading till the file is finished
        while f.tell() < fileLength and records_read < n_records_to_read:
            # Skip the last record if requested, which usually contains
            # incomplete data
            if ignore_last_record and f.tell() == (
                fileLength - record_length_bytes):
                break
            
            # Read the timestamp for this record
            # litte-endian 64-bit signed integer
            timestamps.append(np.fromfile(f, np.dtype('<i8'), 1))
        
            # Read the number of samples in this record
            # little-endian 16-bit unsigned integer
            N = np.fromfile(f, np.dtype('<u2'), 1).item() 
            if N != header['blockLength']:
                raise IOError('Found corrupted record in block ' + 
                    str(recordNumber))
            
            # Read and store the recording numbers
            # big-endian 16-bit unsigned integer
            recordingNumbers.append(np.fromfile(f, np.dtype('>u2'), 1))
            
            # Read the data
            # big-endian 16-bit signed integer
            data = np.fromfile(f, np.dtype('>i2'), N)
            if len(data) != N:
                raise IOError("could not load the right number of samples")
            
            # Optionally convert dtype
            if dtype == float: 
                data = data * header['bitVolts']
                        
            # Store the data
            samples.append(data)

            # Extract and test the record marker
            record_marker = np.fromfile(f, np.dtype('<u1'), 10)
            if np.any(record_marker != spec_record_marker):
                raise IOError("corrupted record marker at record %d" %
                    records_read)
            
            # Update the count
            samples_read += len(samples)            
            records_read += 1

    # Concatenate results, or empty arrays if no data read (which happens
    # if start_sample is after the end of the data stream)
    res = {'header': header}
    if samples_read > 0:
        res['timestamps'] = np.concatenate(timestamps)
        res['data'] = np.concatenate(samples)
        res['recordingNumber'] = np.concatenate(recordingNumbers)
    else:
        res['timestamps'] = np.array([], dtype=np.int)
        res['data'] = np.array([], dtype=dtype)
        res['recordingNumber'] = np.array([], dtype=np.int)
    return res
    
def loadSpikes(filepath):
    
    data = { }
    
    print('loading spikes...')
    
    f = open(filepath,'rb')
    header = readHeader(f)
    
    if float(header[' version']) < 0.4:
        raise Exception('Loader is only compatible with .spikes files with version 0.4 or higher')
     
    data['header'] = header 
    numChannels = int(header['num_channels'])
    numSamples = 40 # **NOT CURRENTLY WRITTEN TO HEADER**
    
    spikes = np.zeros((MAX_NUMBER_OF_SPIKES, numSamples, numChannels))
    timestamps = np.zeros(MAX_NUMBER_OF_SPIKES)
    source = np.zeros(MAX_NUMBER_OF_SPIKES)
    gain = np.zeros((MAX_NUMBER_OF_SPIKES, numChannels))
    thresh = np.zeros((MAX_NUMBER_OF_SPIKES, numChannels))
    sortedId = np.zeros((MAX_NUMBER_OF_SPIKES, numChannels))
    recNum = np.zeros(MAX_NUMBER_OF_SPIKES)
    
    currentSpike = 0
    
    while f.tell() < os.fstat(f.fileno()).st_size:
        
        eventType = np.fromfile(f, np.dtype('<u1'),1) #always equal to 4, discard
        timestamps[currentSpike] = np.fromfile(f, np.dtype('<i8'), 1)
        software_timestamp = np.fromfile(f, np.dtype('<i8'), 1)
        source[currentSpike] = np.fromfile(f, np.dtype('<u2'), 1)
        numChannels = np.fromfile(f, np.dtype('<u2'), 1)
        numSamples = np.fromfile(f, np.dtype('<u2'), 1)
        sortedId[currentSpike] = np.fromfile(f, np.dtype('<u2'),1)
        electrodeId = np.fromfile(f, np.dtype('<u2'),1)
        channel = np.fromfile(f, np.dtype('<u2'),1)
        color = np.fromfile(f, np.dtype('<u1'), 3)
        pcProj = np.fromfile(f, np.float32, 2)
        sampleFreq = np.fromfile(f, np.dtype('<u2'),1)
        
        waveforms = np.fromfile(f, np.dtype('<u2'), numChannels*numSamples)
        wv = np.reshape(waveforms, (numChannels, numSamples))

        gain[currentSpike,:] = np.fromfile(f, np.float32, numChannels)
        thresh[currentSpike,:] = np.fromfile(f, np.dtype('<u2'), numChannels)
        
        recNum[currentSpike] = np.fromfile(f, np.dtype('<u2'), 1)       
        
        for ch in range(numChannels):
            spikes[currentSpike,:,ch] = (np.float64(wv[ch])-32768)/(gain[currentSpike,ch]/1000)
        
        currentSpike += 1
        
    data['spikes'] = spikes[:currentSpike,:,:]
    data['timestamps'] = timestamps[:currentSpike]
    data['source'] = source[:currentSpike]
    data['gain'] = gain[:currentSpike,:]
    data['thresh'] = thresh[:currentSpike,:]
    data['recordingNumber'] = recNum[:currentSpike]
    data['sortedId'] = sortedId[:currentSpike]

    return data
    
def loadEvents(filepath):

    data = { }
    
    print('loading events...')
    
    f = open(filepath,'rb')
    header = readHeader(f)
    
    if float(header['version']) < 0.4:
        raise Exception('Loader is only compatible with .events files with version 0.4 or higher')
     
    data['header'] = header 
    
    index = -1

    channel = np.zeros(MAX_NUMBER_OF_EVENTS)
    timestamps = np.zeros(MAX_NUMBER_OF_EVENTS)
    sampleNum = np.zeros(MAX_NUMBER_OF_EVENTS)
    nodeId = np.zeros(MAX_NUMBER_OF_EVENTS)
    eventType = np.zeros(MAX_NUMBER_OF_EVENTS)
    eventId = np.zeros(MAX_NUMBER_OF_EVENTS)
    recordingNumber = np.zeros(MAX_NUMBER_OF_EVENTS)

    while f.tell() < os.fstat(f.fileno()).st_size:
        
        index += 1
        
        timestamps[index] = np.fromfile(f, np.dtype('<i8'), 1)
        sampleNum[index] = np.fromfile(f, np.dtype('<i2'), 1)
        eventType[index] = np.fromfile(f, np.dtype('<u1'), 1)
        nodeId[index] = np.fromfile(f, np.dtype('<u1'), 1)
        eventId[index] = np.fromfile(f, np.dtype('<u1'), 1)
        channel[index] = np.fromfile(f, np.dtype('<u1'), 1)
        recordingNumber[index] = np.fromfile(f, np.dtype('<u2'), 1)
        
    data['channel'] = channel[:index]
    data['timestamps'] = timestamps[:index]
    data['eventType'] = eventType[:index]
    data['nodeId'] = nodeId[:index]
    data['eventId'] = eventId[:index]
    data['recordingNumber'] = recordingNumber[:index]
    data['sampleNum'] = sampleNum[:index]
    
    return data


def readHeader(f):
    """Read header information from the first 1024 bytes of an OpenEphys file.

    Args:
        f: An open file handle to an OpenEphys file

    Returns: dict with the following keys.
        - bitVolts : float, scaling factor, microvolts per bit
        - blockLength : int, e.g. 1024, length of each record (see
            loadContinuous)
        - bufferSize : int, e.g. 1024
        - channel : the channel, eg "'CH1'"
        - channelType : eg "'Continuous'"
        - date_created : eg "'15-Jun-2016 21212'" (What are these numbers?)
        - description : description of the file format
        - format : "'Open Ephys Data Format'"
        - header_bytes : int, e.g. 1024
        - sampleRate : float, e.g. 30000.
        - version: eg '0.4'
        Note that every value is a string, even numeric data like bitVolts.
        Some strings have extra, redundant single apostrophes.
    """
    header = {}

    # Read the data as a string
    # Remove newlines and redundant "header." prefixes
    # The result should be a series of "key = value" strings, separated
    # by semicolons.
    # header_string = f.read(1024).replace('\n','').replace('header.','')
    header_string = f.read(1024).decode().replace('\n', '').replace('header.', '')

    # Parse each key = value string separately
    for pair in header_string.split(';'):
        if '=' in pair:
            key, value = pair.split(' = ')
            key = key.strip()
            value = value.strip()

            # Convert some values to numeric
            if key in ['bitVolts', 'sampleRate']:
                header[key] = float(value)
            elif key in ['blockLength', 'bufferSize', 'header_bytes']:
                header[key] = int(value)
            else:
                # Keep as string
                header[key] = value
    # print(header)
    return header
    
def downsample(trace,down):
    downsampled = scipy.signal.resample(trace,np.shape(trace)[0]/down)
    return downsampled
    
def writeChannelMapFile(mapping, filename='mapping.prb'):
    
    with open(filename, 'w') as outfile:
        json.dump( \
                      {'0': {  \
                            'mapping' : mapping.tolist(), \
                            'reference' : [-1] * mapping.size, \
                            'enabled' : [True] * mapping.size \
                            }, \
                        'refs' : {\
                            'channels' : [-1] * mapping.size \
                            }, \
                        'recording' : { \
                           'channels': [False] * mapping.size \
                           }, \
                     }, \
                      outfile, \
                      indent = 4, separators = (',', ': ') \
                 ) 

def pack(folderpath, filename='openephys.dat', dref=None,
    chunk_size=4000, start_record=None, stop_record=None, verbose=True,
    **kwargs):
    """Read OpenEphys formatted data in chunks and write to a flat binary file.
    
    The data will be written in a fairly standard binary format:
        ch0_sample0, ch1_sample0, ..., chN_sample0,
        ch0_sample1, ch1_sample1, ..., chN_sample1,
    and so on. Each sample is a 2-byte signed integer.
    
    Because the data are read from the OpenEphys files in chunks, it
    is not necessary to hold the entire dataset in memory at once. It is
    also possible to specify starting and stopping locations to write out
    a subset of the data.
    
    Args:
        folderpath : string, path to folder containing all channels
        filename : name of file to store packed binary data
            If this file exists, it will be overwritten
        dref:  Digital referencing - either supply a channel number or 
            'ave' to reference to the average of packed channels.
        chunk_size : the number of records (not bytes or samples!) to read at
            once. 4000 records of 64-channel data requires ~500 MB of memory.
            The record size is usually 1024 samples.
        start_record, stop_record : the first record to process and the
            last record to process. If start_record is None, start at the
            beginning; if stop_record is None, go until the end.
        verbose : print out status info
        **kwargs : This is passed to loadFolderToArray for each chunk.
            See documentation there for the keywords `source`, `channels`,
            `recording`, and `ignore_last_record`.
    """
    # Get header info to determine how many records we have to pack
    header = get_header_from_folder(folderpath, **kwargs)
    if start_record is None:
        start_record = 0
    if stop_record is None:
        stop_record = header['n_records']
    
    # Manually remove the output file if it exists (later we append)
    if os.path.exists(filename):
        if verbose:
            print("overwriting %s" % filename)
        os.remove(filename)
    
    # Iterate over chunks
    for chunk_start in range(start_record, stop_record, chunk_size):
        # Determine where the chunk stops
        chunk_stop = np.min([stop_record, chunk_start + chunk_size])
        if verbose:
            print("loading chunk from %d to %d" % (chunk_start, chunk_stop))
        
        # Load the chunk
        data_array = loadFolderToArray(folderpath, dtype=np.int16,
            start_record=chunk_start, stop_record=chunk_stop,
            verbose=False, **kwargs)

        # This only happens if we happen to be loading a chunk consisting
        # of only the last record, and also ignore_last_record is True
        if len(data_array) == 0:
            break

        # Digital referencing
        if dref: 
            # Choose a reference
            if dref == 'ave':
                reference = np.mean(data_array, 1)
            else:
                # Figure out which channels are included
                if 'channels' in kwargs and kwargs['channels'] != 'all':
                    channels = kwargs['channels']
                else:
                    channels = _get_sorted_channels(folderpath)
                
                # Find the reference channel
                dref_idx = channels.index(dref)
                reference = data_array[:, dref_idx].copy()
            
            # Subtract the reference
            for i in range(data_array.shape[1]):
                data_array[:,i] = data_array[:,i] - reference
        
        # Explicity open in append mode so we don't just overwrite
        with file(os.path.join(folderpath, filename), 'ab') as fi:
            data_array.tofile(fi)

def regex_capture(pattern, list_of_strings, take_index=0):
    """Apply regex `pattern` to each string and return a captured group.
    
    pattern : string, regex pattern
    list_of_strings : list of strings to apply the pattern to
        Strings that do not match the pattern are ignored.
    take_index : The index of the captured group to return
    
    Returns: a list of strings. Each element is the captured group from
        one of the input strings.
    """
    res_l = []
    for s in list_of_strings:
        m = re.match(pattern, s)
        
        # Append the capture, if any
        if m is not None:
            res_l.append(m.groups()[take_index])
    
    return res_l

def get_number_of_recording_sessions(folderpath):
    """Return the number of recordings in folderpath.

    folderpath : string, path to location of continuous files on disk
    """
    recording_s = sorted([(f.split('_CH')[0], (f.split('_CH')[1].split('.')[0])) for f in os.listdir(folderpath)
                          if
                          '.continuous' in f
                          and '_CH' in f],
                         key=lambda x: (x[0], (x[1])))
    # and source in f.split('_CH')[0]])

    full_list = [a + '_1' if len(a.split('_')) == 1 else a for a in list(zip(*recording_s))[1]]
    N = max([a.split('_')[1] for a in full_list])
    return int(N)


def _get_sorted_channels(folderpath, recording=None):
    """Return a sorted list of the continuous channels in folderpath.
    
    folderpath : string, path to location of continuous files on disk
    recording : None, or int
        If there is only one recording in the folder, leave as None.
        Otherwise, specify the number of the recording as an integer.
    """
    if recording is None:
        return sorted([int(f.split('_CH')[1].split('.')[0]) for f in os.listdir(folderpath)
                    if '.continuous' in f and '_CH' in f])
    else:
        # Form a string from the recording number
        if recording == 1:
            # The first recording has no suffix
            recording_s = ''
        else:
            recording_s = '_%d' % recording
        
        # Form a regex pattern to be applied to each filename
        # We will capture the channel number: (\d+)
        regex_pattern = '%s_CH(\d+)%s.continuous' % ('100', recording_s)
        
        # Apply the pattern to each filename and return the captured channels
        channel_numbers_s = regex_capture(regex_pattern, os.listdir(folderpath))
        channel_numbers_int = map(int, channel_numbers_s)
        return sorted(channel_numbers_int)

def get_number_of_records(filepath):
    # Open the file
    with open(filepath, 'rb') as f:
        # Read header info
        header = readHeader(f)
        
        # Get file length
        fileLength = os.fstat(f.fileno()).st_size
        
        # Determine the number of records
        record_length_bytes = 2 * header['blockLength'] + 22
        # some files have broken records, remove check...
        n_records = int(math.floor((fileLength - 1024) / record_length_bytes))
        # if (n_records * record_length_bytes + 1024) != fileLength:
        #     n_records -= 1
#            raise IOError("file does not divide evenly into full records")
    
    return n_records

def get_filelist(folderpath, source='100', channels='all', recording=None):
    """Given a folder of data files and a list of channels, get filenames.
    
    folderpath : string, folder containing OpenEphys data files
    source : string, typically '100'
    channels : list of numeric channel numbers to acquire
        If 'all', then _get_sorted_channels is used to get all channels
        from that folder in sorted order
    recording : the recording number, or None if there is only one recording
    
    Returns: a list of filenames corresponding one-to-one to the channels
        in `channels`. The filenames must be joined with `folderpath` to
        construct a full filename.
    """

        # Get all channels if requested
    if channels == 'all':
        channels = _get_sorted_channels(folderpath, recording=recording)

    # Get the list of continuous filenames
    if recording is None or recording == 1:
        # The first recording has no suffix
        filelist = ['%s_CH%d.continuous' % (source, chan)
            for chan in channels]
    else:
        filelist = ['%s_CH%d_%d.continuous' % (source, chan, recording)
            for chan in channels]

    return filelist


def get_header_from_folder(folderpath, filelist=None, **kwargs):
    """Return the header info for all files in `folderpath`.
    
    The header for each file is loaded individually. The following keys
    are supposed to be the same for every file:
        ['bitVolts', 'blockLength', 'bufferSize', 'date_created',
        'description', 'format', 'header_bytes', 'sampleRate', 'version']
    They are checked for consistency and returned in a single dict.    
    
    Finally the number of records is also checked for each file, checked
    for consistency, and returned as the key 'n_records'.
    
    folderpath : folder containing OpenEphys data files
    filelist : list of filenames within `folderpath` to load
        If None, then provide optional keyword arguments `source`, 
        `channels`, and/or `recording`. They are passed to `get_filelist`
        to get the filenames in this folder.
    
    Returns: dict
    """
    included_keys = ['blockLength', 'bufferSize', 'date_created',
        'description', 'format', 'header_bytes', 'version', 'n_records']
    included_float_keys = ['bitVolts', 'sampleRate']
    
    # Get filelist if it was not provided
    if filelist is None:
        filelist = get_filelist(folderpath, **kwargs)
    
    # Get header for each file, as well as number of records
    header_l = []
    for filename in filelist:
        full_filename = os.path.join(folderpath, filename)
        with open(full_filename, 'rb') as fi:
            header = readHeader(fi)
        header['n_records'] = get_number_of_records(full_filename)
        header_l.append(header)
    if len(header_l) == 0:
        raise IOError("no headers could be loaded")
    
    # Form a single header based on all of them, starting with the first one
    unique_header = {}
    for key in included_keys + included_float_keys:
        unique_header[key] = header_l[0][key]
    
    # Check every header
    for header in header_l:
        # Check the regular keys
        for key in (k for k in included_keys if k != 'n_records'):
            if unique_header[key] != header[key]:
                raise ValueError("inconsistent header info in key %s" % key)
        # due to recording interruption, there might be a diff of 1 in rec length, which needs to be ignored
        key = 'n_records'
        if unique_header[key] != header[key]:
            if abs(int(unique_header[key]) - int(header[key])) > 2:
                raise ValueError("inconsistent header info in key %s" % key)


        # Check the floating point keys
        for key in included_float_keys:
            if not np.isclose(unique_header[key], header[key]):
                raise ValueError("inconsistent header info in key %s" % key)
            
    return unique_header
