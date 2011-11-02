"""Reading and writing from KlustaKwik-format files.
Ref: http://klusters.sourceforge.net/UserManual/data-files.html

Supported : Read, Write

Author : Chris Rodgers

TODO:
* When reading, put the Unit into the RCG, RC hierarchy
* When writing, figure out how to get group and cluster if those annotations
weren't set. Consider removing those annotations if they are redundant.
* Load features in addition to spiketimes.
"""

# I need to subclass BaseIO
from .baseio import BaseIO

# to import : Block, Segment, AnalogSignal, SpikeTrain, SpikeTrainList
from ..core import *
from .tools import create_many_to_one_relationship

# note neo.core need only numpy and quantitie
import numpy as np
import quantities as pq

# Pasted version of feature file format spec
"""
The Feature File

Generic file name: base.fet.n

Format: ASCII, integer values

The feature file lists for each spike the PCA coefficients for each
electrode, followed by the timestamp of the spike (more features can
be inserted between the PCA coefficients and the timestamp). 
The first line contains the number of dimensions. 
Assuming N1 spikes (spike1...spikeN1), N2 electrodes (e1...eN2) and
N3 coefficients (c1...cN3), this file looks like:

nbDimensions
c1_e1_spike1   c2_e1_spike1  ... cN3_e1_spike1   c1_e2_spike1  ... cN3_eN2_spike1   timestamp_spike1
c1_e1_spike2   c2_e1_spike2  ... cN3_e1_spike2   c1_e2_spike2  ... cN3_eN2_spike2   timestamp_spike2
...
c1_e1_spikeN1  c2_e1_spikeN1 ... cN3_e1_spikeN1  c1_e2_spikeN1 ... cN3_eN2_spikeN1  timestamp_spikeN1

The timestamp is expressed in multiples of the sampling interval. For
instance, for a 20kHz recording (50 microsecond sampling interval), a
timestamp of 200 corresponds to 200x0.000050s=0.01s from the beginning
of the recording session.

Notice that the last line must end with a newline or carriage return. 
"""

import numpy as np
import glob
import matplotlib.mlab as mlab
import os.path
import shutil
import logging


class KlustaKwikIO(BaseIO):
    """Reading and writing from KlustaKwik-format files."""    
    # Class variables demonstrating capabilities of this IO
    is_readable        = True
    is_writable        = True
    
    # This IO can only manipulate objects relating to spike times
    supported_objects  = [Block, SpikeTrain, Unit]
    
    # Keep things simple by always returning a block
    readable_objects    = [Block]
    
    # And write a block
    writeable_objects   = [Block]

    # Not sure what these do, if anything
    has_header         = False
    is_streameable     = False
    
    # GUI params
    read_params = {}
    
    # GUI params
    write_params = {}
    
    # The IO name and the file extensions it uses
    name               = 'KlustaKwik'    
    extensions          = ['fet', 'clu', 'res', 'spk']
    
    # Operates on directories
    mode = 'file'     
    
    def __init__(self, filename, sampling_rate=30000.):
        """Create a new IO to operate on a directory        
        
        filename : the directory to contain the files
        basename : string, basename of KlustaKwik format, or None
        sampling_rate : in Hz, necessary because the KlustaKwik files
            stores data in samples.
        """
        BaseIO.__init__(self)
        #self.filename = os.path.normpath(filename)
        self.filename, self.basename = os.path.split(os.path.abspath(filename))
        self.sampling_rate = float(sampling_rate)
        
        # error check
        if not os.path.isdir(self.filename):
            raise ValueError("filename must be a directory")
        
        # initialize a helper object to parse filenames
        self._fp = FilenameParser(dirname=self.filename, basename=self.basename)

    # The reading methods. The `lazy` and `cascade` parameters are imposed
    # by neo.io API
    def read_block(self, lazy=False, cascade=True):
        """Returns a Block containing spike information.
        
        There is no obvious way to infer the segment boundaries from
        raw spike times, so for now all spike times are returned in one
        big segment. The way around this would be to specify the segment
        boundaries, and then change this code to put the spikes in the right
        segments.
        """                
        # Create block and segment to hold all the data
        block = Block()            
        # Search data directory for KlustaKwik files.
        # If nothing found, return empty block
        self._fetfiles = self._fp.read_filenames('fet')
        self._clufiles = self._fp.read_filenames('clu')        
        if len(self._fetfiles) == 0 or not cascade:
            return block

        # Create a single segment to hold all of the data
        seg = Segment(name='seg0', index=0, file_origin=self.filename)
        block.segments.append(seg)
        
        # Load spike times from each group and store in a dict, keyed
        # by group number
        self.spiketrains = dict()
        for group in sorted(self._fetfiles.keys()):
            # Load spike times 
            fetfile = self._fetfiles[group]
            spks, features = self._load_spike_times(fetfile)
            
            # Load cluster ids or generate
            if group in self._clufiles:
                clufile = self._clufiles[group]
                uids = self._load_unit_id(clufile)
            else:
                # unclustered data, assume all zeros
                uids = np.zeros(spks.shape, dtype=np.int32)

            # error check
            if len(spks) != len(uids):
                raise ValueError("lengths of fet and clu files are different")
            
            # Create Unit for each cluster
            unique_unit_ids = np.unique(uids)            
            for unit_id in sorted(unique_unit_ids):
                # Initialize the unit
                u = Unit(name=('unit %d from group %d' % (unit_id, group)), 
                    index=unit_id, group=group)                
                
                # Initialize a new SpikeTrain for the spikes from this unit
                if lazy:
                    st = SpikeTrain(
                        times=[], 
                        units='sec', t_start=0.0, 
                        t_stop=spks.max() / self.sampling_rate,
                        name=('unit %d from group %d' % (unit_id, group)))
                    st.lazy_shape = len(spks[uids==unit_id])
                else:
                    st = SpikeTrain(
                        times=spks[uids==unit_id] / self.sampling_rate, 
                        units='sec', t_start=0.0, 
                        t_stop=spks.max() / self.sampling_rate,
                        name=('unit %d from group %d' % (unit_id, group)))
                st.annotations['cluster'] = unit_id
                st.annotations['group'] = group
                
                # put features in
                if not lazy and len(features) != 0:
                    st.annotations['waveform_features'] = features
                
                # Link
                u.spiketrains.append(st)
                seg.spiketrains.append(st)
        
        create_many_to_one_relationship(block)
        return block

    # Helper hidden functions for reading
    def _load_spike_times(self, fetfilename):
        """Reads and returns the spike times and features"""
        f = file(fetfilename, 'r')
        
        # Number of clustering features is integer on first line
        nbFeatures = int(f.readline().strip())
        
        # Each subsequent line consists of nbFeatures values, followed by
        # the spike time in samples.
        names = ['fet%d' % n for n in xrange(nbFeatures)]
        names.append('spike_time')
        
        # Load into recarray
        data = mlab.csv2rec(f, names=names, skiprows=1, delimiter=' ')
        f.close()
        
        # get features
        features = np.array([data['fet%d' % n] for n in xrange(nbFeatures)])
        
        # Return the spike_time column
        return data['spike_time'], features.transpose()
    
    def _load_unit_id(self, clufilename):
        """Reads and return the cluster ids as int32"""
        f = file(clufilename, 'r')
        
        # Number of clusters on this tetrode is integer on first line
        nbClusters = int(f.readline().strip())
        
        # Read each cluster name as a string
        cluster_names = f.readlines()
        f.close()
        
        # Convert names to integers
        # I think the spec requires cluster names to be integers, but
        # this code could be modified to support string names which are
        # auto-numbered.
        try:
            cluster_ids = [int(name) for name in cluster_names]
        except ValueError:
            raise ValueError(
                "Could not convert cluster name to integer in %s" % clufilename)
        
        # convert to numpy array and error check
        cluster_ids = np.array(cluster_ids, dtype=np.int32)
        if len(np.unique(cluster_ids)) != nbClusters:
            logging.warning("warning: I got %d clusters instead of %d in %s" % (
                len(np.unique(cluster_ids)), nbClusters, clufilename))
        
        return cluster_ids
    
    
    # writing functions
    def write_block(self, block):     
        """Write spike times and unit ids to disk.
        
        Currently descends hierarchy from block to segment to spiketrain.
        Then gets group and cluster information from spiketrain.
        Then writes the time and cluster info to the file associated with
        that group. 
        
        The group and cluster information are extracted from annotations,
        eg `sptr.annotations['group']`. If no cluster information exists,
        it is assigned to cluster 0.
        
        Note that all segments are essentially combined in
        this process, since the KlustaKwik format does not allow for
        segment boundaries.
        
        As implemented currently, does not use the `Unit` object at all.
        
        We first try to use the sampling rate of each SpikeTrain, or if this
        is not set, we use `self.sampling_rate`.
        
        If the files already exist, backup copies are created by appending
        the filenames with a "~".
        """
        # set basename
        if self.basename is None:
            logging.warning("warning: no basename provided, using `basename`")
            self.basename = 'basename'
        
        # First create file handles for each group which will be stored
        self._make_all_file_handles(block)
        
        # We'll detect how many features belong in each group
        self._group2features = {}
        
        # Iterate through segments in this block
        for seg in block.segments:
            # Write each spiketrain of the segment
            for st in seg.spiketrains:
                # Get file handles for this spiketrain using its group
                group = self.st2group(st)
                fetfilehandle = self._fetfilehandles[group]
                clufilehandle = self._clufilehandles[group]
                
                # Get the id to write to clu file for this spike train
                cluster = self.st2cluster(st)
                
                # Choose sampling rate to convert to samples
                try:
                    sr = st.annotations['sampling_rate']
                except KeyError:
                    sr = self.sampling_rate
                
                # Convert to samples
                spike_times_in_samples = np.rint(
                    np.array(st) * sr).astype(np.int)
                
                # Try to get features from spiketrain
                try:
                    all_features = st.annotations['waveform_features']
                except KeyError:
                    # Use empty
                    all_features = [
                        [] for n in range(len(spike_times_in_samples))]
                all_features = np.asarray(all_features)
                if all_features.ndim != 2:
                    raise ValueError("waveform features should be 2d array")
                
                # Check number of features we're supposed to have
                try:
                    n_features = self._group2features[group]
                except KeyError:
                    # First time through .. set number of features
                    n_features = all_features.shape[1]
                    self._group2features[group] = n_features
                    
                    # and write to first line of file
                    fetfilehandle.write("%d\n" % n_features)                    
                if n_features != all_features.shape[1]:
                    raise ValueError("inconsistent number of features: " +
                        "supposed to be %d but I got %d" %\
                        (n_features, all_features.shape[1]))
                
                # Write features and time for each spike
                for stt, features in zip(spike_times_in_samples, all_features):
                    # first features
                    for val in features:
                        fetfilehandle.write(str(val))
                        fetfilehandle.write(" ")
                    
                    # now time
                    fetfilehandle.write("%d\n" % stt)
                    
                    # and cluster id
                    clufilehandle.write("%d\n" % cluster)

        # We're done, so close the files
        self._close_all_files()   

    # Helper functions for writing
    def st2group(self, st):
        # Not sure this is right so make it a method in case we change it
        try:
            return st.annotations['group']
        except KeyError:
            return 0
    
    def st2cluster(self, st):
        # Not sure this is right so make it a method in case we change it
        try:
            return st.annotations['cluster']
        except KeyError:
            return 0
    
    def _make_all_file_handles(self, block):
        """Get the tetrode (group) of each neuron (cluster) by descending
        the hierarchy through segment and block.
        Store in a dict {group_id: list_of_clusters_in_that_group}
        """
        group2clusters = {}
        for seg in block.segments:
            for st in seg.spiketrains:
                group = self.st2group(st)
                cluster = self.st2cluster(st)
            
                if group in group2clusters:
                    if cluster not in group2clusters[group]:
                        group2clusters[group].append(cluster)
                else:
                    group2clusters[group] = [cluster]
        
        # Make new file handles for each group
        self._fetfilehandles, self._clufilehandles = {}, {}
        for group, clusters in group2clusters.items():
            self._new_group(group, nbClusters=len(clusters))

    def _new_group(self, id_group, nbClusters):
        # generate filenames
        fetfilename = os.path.join(self.filename, 
            self.basename + ('.fet.%d' % id_group))
        clufilename = os.path.join(self.filename,
            self.basename + ('.clu.%d' % id_group))
        
        # back up before overwriting
        if os.path.exists(fetfilename):
            shutil.copyfile(fetfilename, fetfilename + '~')
        if os.path.exists(clufilename):
            shutil.copyfile(clufilename, clufilename + '~')
        
        # create file handles
        self._fetfilehandles[id_group] = file(fetfilename, 'w')
        self._clufilehandles[id_group] = file(clufilename, 'w')
        
        # write out first line        
        #self._fetfilehandles[id_group].write("0\n") # Number of features
        self._clufilehandles[id_group].write("%d\n" % nbClusters)
    
    def _close_all_files(self):
        for val in self._fetfilehandles.values(): val.close()        
        for val in self._clufilehandles.values(): val.close()


class FilenameParser:
    """Simple class to interpret user's requests into KlustaKwik filenames"""
    def __init__(self, dirname, basename=None):
        """Initialize a new parser for a directory containing files
        
        dirname: directory containing files
        basename: basename in KlustaKwik format spec
        
        If basename is left None, then files with any basename in the directory
        will be used. An error is raised if files with multiple basenames
        exist in the directory.
        """
        self.dirname = os.path.normpath(dirname)
        self.basename = basename
        
        # error check
        if not os.path.isdir(self.dirname):
            raise ValueError("filename must be a directory")
    
    def read_filenames(self, typestring='fet'):
        """Returns filenames in the data directory matching the type.
        
        Generally, `typestring` is one of the following:
            'fet', 'clu', 'spk', 'res'
        
        Returns a dict {group_number: filename}, e.g.:
            {   0: 'basename.fet.0',
                1: 'basename.fet.1',
                2: 'basename.fet.2'}

        'basename' can be any string not containing whitespace.
        
        Only filenames that begin with "basename.typestring." and end with
        a sequence of digits are valid. The digits are converted to an integer
        and used as the group number.
        """
        all_filenames = glob.glob(os.path.join(self.dirname, '*'))

        
        # Fill the dict with valid filenames
        d = {}
        for v in all_filenames:
            # Test whether matches format, ie ends with digits
            split_fn = os.path.split(v)[1]
            m = glob.re.search(('^(\w+)\.%s\.(\d+)$' % typestring), split_fn)             
            if m is not None:
                # get basename from first hit if not specified
                if self.basename is None:
                    self.basename = m.group(1)                
                
                # return files with correct basename
                if self.basename == m.group(1):
                    # Key the group number to the filename
                    # This conversion to int should always work since only
                    # strings of digits will match the regex                
                    tetn = int(m.group(2))
                    d[tetn] = v
        
        return d        
        
    

