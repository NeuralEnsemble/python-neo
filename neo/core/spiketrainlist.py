# -*- coding: utf-8 -*-
import numpy
from spiketrain import SpikeTrain
from neuron import Neuron

class SpikeTrainList(object):
     
    """
    Object to represent a list of SpikeTrains

    **Definition**
    An :class:`SpikeTrainList` is a list of SpikeTrains. Those :class:`SpikeTrain` can be 
    from different :class:`Neuron` within the same :class:`Segment`, or from the same 
    :class:`Neuron` across several :class:`Segment`

    with arguments:
    
    ``spiketrains`` 
        The list of :class:`SpikeTrain` contained in the :class:`SpikeTrainList`. The :class:`Neuron`
        objects belonging to all those :class:`SpikeTrain` will be used to create a dictionnary of
        :class:`SpikeTrain`.

    ``t_start``
        The absolute beginning (in second) of the :class:`SpikeTrain`
    
    ``t_stop``
        The absolute end (in second) of the :class:`SpikeTrain`

    **Usage**

    **Example**

    """
    
    _t_start     = None
    _t_stop      = None    
    
    def __init__(self, *arg, **karg):
        
        self.spiketrains = {}
        self.size        = len(karg['spiketrains'])        
        
        if karg.has_key('t_start'):
            self._t_start = karg['t_start']
        if karg.has_key('t_stop'):
            self._t_stop = karg['t_stop']
        if karg.has_key('interval'):
            self._t_start, self._t_stop = karg['interval']        
        
        for train in karg['spiketrains']:
            id = train.neuron.id             
            self.spiketrains[id] = train
            if self._t_start or self._t_stop:
                self.spiketrains[id].set_times(self._t_start, self._t_stop) 
                
    @property
    def id_list(self):
        """ 
        Return the list of all the cells ids contained in the
        SpikeTrainList object
        
        Examples
            >> spklist.id_list()
                [0,1,2,3,....,9999]
        """
        return numpy.array(self.spiketrains.keys(), int)
    
    @property
    def neuron_list(self):
        """ 
        Return the list of all the Neurons objects contained in the
        SpikeTrainList object
        
        Examples
            >> spklist.id_list()
                [0,1,2,3,....,9999]
        """
        return [spk.neuron for spk in self]
    
    def __getitem__(self, id):
        if id in self.id_list:
            return self.spiketrains[id]
        else:
            raise Exception("Neuron with id %d is not present in this SpikeTrainList" %id)
    
    def __getslice__(self, i, j):
        """
        Return a new SpikeList object with all the ids between i and j
        """        
        ids = numpy.where((self.id_list >= i) & (self.id_list < j))[0]
        return self.id_slice(ids)
    
    def __iter__(self):
        return self.spiketrains.itervalues()

    def __len__(self):
        return len(self.spiketrains)

    def __get_sublist__(self, sublist=None):
        """
        Internal function used to get a sublist for the Spikelist id list
        
        Inputs:
            sublist - can be an int (and then N random cells are selected). Otherwise
                      sub_list is a list of cell in self.id_list(). If None, id_list is returned
        
        Examples:
            >> self.__get_sublist__(50)
        """
        if sublist is None:
            return self.id_list
        elif type(sublist) == int:
            return numpy.random.permutation(self.id_list)[0:sublist]
        else:
            return sublist

    def duration(self):
        """
        Return the duration of the SpikeTrain
        """
        return self._t_stop - self._t_start
    
    def append(self, spktrain):
        """
        Add a SpikeTrain object to the SpikeTrainList
        
        Inputs:
            spktrain - the SpikeTrain object emitted by a particular cell
        
        The SpikeTrain object is sliced according to the t_start and t_stop times
        of the SpikeTrainList object
        
        Examples
            >> st=SpikeTrain(range(0,100,5),0.1,0,100)
            >> spklist.append(st)
        
        See also
            concatenate, __setitem__
        """
        assert isinstance(spktrain, SpikeTrain), "A SpikeList object can only contain SpikeTrain objects"
        id = spktrain.neuron.id
        if id in self.id_list:
            raise Exception("id %d already present in SpikeList. Use __setitem__ (spk[id]=...) instead()" %id)
        else:
            self.spiketrains[id] = spktrain.time_slice(self._t_start, self._t_stop)
     
    def complete(self, id_list):
        """
        Complete the SpikeTrainList by adding empty SpikeTrain for all the ids present in
        id_list that will not already be in the SpikeTrainList
        
         Inputs:
            id_list - The id_list that should be at the end in the SpikeTrainList
        
        Examples:
            >> spklist.id_list
                [0,2,5]
            >> spklist.complete(arange(5))
            >> spklist.id_list
                [0,1,2,3,4]
        """
        id_list     = set(id_list)
        missing_ids = id_list.difference(set(self.id_list))
        for id in missing_ids:
            data = SpikeTrain(spike_times=[], interval=(self._t_start, self._t_stop), neuron=Neuron(id=id))
            self.append(data)
    
    def id_slice(self, id_list):
        """
        Return a new SpikeTrainList obtained by selecting particular ids
        
        Inputs:
            id_list - Can be an integer (and then N random cells will be selected)
                      or a sublist of the current ids
        
        The new SpikeTrainList inherits the time parameters (t_start, t_stop)
        
        Examples:
            >> spklist.id_list()
                [830, 1959, 1005, 416, 1011, 1240, 729, 59, 1138, 259]
            >> new_spklist = spklist.id_slice(5)
            >> new_spklist.id_list()
                [1011, 729, 1138, 416, 59]

        See also
            time_slice, interval_slice
        """
        
        data    = []
        id_list = self.__get_sublist__(id_list)
        for id in id_list:
            data.append(self[id])
        return SpikeTrainList(spiketrains=data, interval=(self._t_start, self._t_stop))

    def time_offset(self, offset):
        """
        Add an offset to the whole SpikeTrainList object. t_start and t_stop are
        shifted from offset, so does all the SpikeTrainList.
         
        Inputs:
            offset - the time offset, in ms
        
        Examples:
            >> spklist.t_start
                1000
            >> spklist.time_offset(50)
            >> spklist.t_start
                1050
        """
        if self._t_start:
            self._t_start += offset
        if self._t_stop:
            self._t_stop  += offset
        for spk in self:
            spk.time_offset(offset)
    
    def first_spike(self):
        """
        Get the (time, id) of the first spike emitted by a cell in the SpikeTrainList
        """
        spike = self._t_stop
        id    = None
        for spk in self:
            if len(spk) > 0:
                if spk.spike_times[0] < spike:
                    spike = spk.spike_times[0]
                    id    = spk.neuron.id  
        if id is None:
            raise Exception("No spikes can be found in the SpikeList object !")
        else:
            return spike, id
    
    def last_spike(self):
        """
        Get the (time, id) of the last spike emitted by a cell in the SpikeTrainList
        """
        spike = self._t_start
        id    = None
        for spk in self:
            if len(spk) > 0:
                if spk.spike_times[-1] > spike:
                    spike = spk.spike_times[-1]
        if id is None:
            raise Exception("No spikes can be found in the SpikeList object !")
        else:
            return spike, id
    
    def select_ids(self, criteria):
        """
        Return the list of all the cells in the SpikeList that will match the criteria
        expressed with the following syntax. 
        
        Inputs : 
            criteria - a string that can be evaluated on a SpikeTrain object, where the 
                       SpikeTrain should be named ``cell''.
        
        Exemples:
            >> spklist.select_ids("cell.mean_rate() > 0") (all the active cells)
            >> spklist.select_ids("cell.mean_rate() == 0") (all the silent cells)
            >> spklist.select_ids("len(cell.spike_times) > 10")
            >> spklist.select_ids("mean(cell.isi()) < 1")
        """
        selected_ids = []
        for cell in self:
            if eval(criteria):
                selected_ids.append(id)
        return selected_ids

    def sort_by(self, criteria, descending=False):
        """
        Return an array with all the ids of the cells in the SpikeList, 
        sorted according to a particular criteria.
        
        Inputs:
            criteria   - the criteria used to sort the cells. It should be a string
                         that can be evaluated on a SpikeTrain object, where the 
                         SpikeTrain should be named ``cell''.
            descending - if True, then the cells are sorted from max to min. 
            
        Examples:
            >> spk.sort_by('cell.mean_rate()')
            >> spk.sort_by('cell.cv_isi()', descending=True)
            >> spk.sort_by('cell.distance_victorpurpura(target, 0.05)')
        """
        criterias = numpy.empty(len(self), float)
        for count, cell in enumerate(self):
            criterias[count] = eval(criteria)
        result = self.id_list[numpy.argsort(criterias)]
        if descending:
            return result[numpy.arange(len(result)-1, -1, -1)]
        else:
            return result

    def isi(self):
        """
        Return the list of all the isi vectors for all the SpikeTrains objects
        within the SpikeList.
        
        See also:
            isi_hist
        """
        isis = []
        for spk in self:
            isis.append(spk.isi())
        return numpy.array(isis)

    def cv_isi(self, no_nan=False):
        """
        Return the list of all the CV coefficients for each SpikeTrains object
        within the SpikeList. Return NaN when not enough spikes are present
        
        Inputs:
            no_nan - False by default. If true, NaN values are automatically
                     removed
        
        Examples:
            >> spklist.cv_isi()
                [0.2,0.3,Nan,2.5,Nan,1.,2.5]
            >> spklist.cv_isi(True)
                [0.2,0.3,2.5,1.,2.5]

        See also:
            cv_isi_hist, cv_local, cv_kl, SpikeTrain.cv_isi
            
        """
        cvs_isi = numpy.empty(len(self))
        for idx, spk in enumerate(self):
            cvs_isi[idx] = spk.cv_isi()
        if no_nan:
            cvs_isi = numpy.extract(numpy.logical_not(numpy.isnan(cvs_isi)),cvs_isi)
        return cvs_isi

    
    def mean_rate(self, t_start=None, t_stop=None):
        """
        Return the mean firing rate averaged accross all SpikeTrains between t_start and t_stop.
        
        Inputs:
            t_start - begining of the selected area to compute mean_rate, in ms
            t_stop  - end of the selected area to compute mean_rate, in ms
        
        If t_start or t_stop are not defined, those of the SpikeList are used
        
        Examples:
            >> spklist.mean_rate()
            >> 12.63
        
        See also
            mean_rates, mean_rate_std
        """
        return numpy.mean(self.mean_rates(t_start, t_stop))


    def mean_rates(self, t_start=None, t_stop=None):
        """ 
        Returns a vector of the size of id_list giving the mean firing rate for each neuron

        Inputs:
            t_start - begining of the selected area to compute std(mean_rate), in ms
            t_stop  - end of the selected area to compute std(mean_rate), in ms
        
        If t_start or t_stop are not defined, those of the SpikeList are used
        
        See also
            mean_rate, mean_rate_std
        """
        rates = numpy.empty(len(self))
        for count, spk in enumerate(self):
            rates[count] = spk.mean_rate(t_start, t_stop)
        return rates


    def spike_histogram(self, t_start=None, t_stop=None, bin_size=0.01, normalized=True):
        """
        Generate an array with all the spike_histograms of all the SpikeTrains
        objects within the SpikeTrainList.
        
        Inputs:
            bin_size   - the time bin used to gather the data
            normalized - if True, the histogram are in Hz (spikes/second), otherwise they are
                         in spikes/bin        
        See also
            firing_rate, time_axis
        """
        if t_start is None:
            t_start = self._t_start
        if t_stop is None:
            t_stop = self.t_stop
        bins       = arange(t_start, t_stop, bin_size)
        N          = len(self)
        M          = len(nbins)
        spike_hist = numpy.zeros((N, M), numpy.float32)
        for idx, spk in enumerate(self):
            spike_hist[idx,:], edges = spk.time_histogram(t_start, t_stop, time_bin, normalized)
        return spike_hist, edges


    def firing_rate(self, bin_size, average=False):
        """
        Generate an array with all the instantaneous firing rates along time (in Hz) 
        of all the SpikeTrains objects within the SpikeTrainList. If average is True, it gives the
        average firing rate over the whole SpikeTrainList
        
        Inputs:
            bin_size   - the time bin (in seconds) used to gather the data
            average    - If True, return a single vector of the average firing rate over the whole SpikeList
        
        See also
            spike_histogram, time_axis
        """
        result = self.spike_histogram(bin_size, normalized=True, display=display, kwargs=kwargs)
        if average:
            return numpy.mean(result, axis=0)
        else:
            return result

    def rates_histogram(self, nbins=25, normalize=True):
        """
        Return an histogram of the mean firing rates within the SpikeTrainList.
        
        Inputs:
            bins    - the number of bins (between the min and max of the data) 
                      or a list/array containing the lower edges of the bins.
        
        See also
            mean_rate, mean_rates
        """
        rates   = self.mean_rates()
        values, xaxis = numpy.histogram(rates, nbins, normed=True)
        return values, edges
    
    def fano_factor(self, bin_size=0.005):
        """
        Compute the Fano Factor of the population activity.
        
        Inputs:
            bin_size   - the number of bins (between the min and max of the data) 
                         or a list/array containing the lower edges of the bins.
        
        The Fano Factor is computed as the variance of the averaged activity divided by its
        mean
        
        See also
            spike_histogram, firing_rate
        """
        firing_rate = self.spike_histogram(bin_size, average=True)
        fano        = numpy.var(firing_rate)/numpy.mean(firing_rate)
        return fano
    
    def pairwise_cc(self, nb_pairs, pairs_generator=None, bin_size=0.001, average=True):
        """
        Function to generate an array of cross correlations computed
        between pairs of cells within the SpikeTrains.
        
        Inputs:
            nb_pairs        - int specifying the number of pairs
            pairs_generator - The generator that will be used to draw the pairs. If None, a default one is
                              created as RandomPairs(spk, spk, no_silent=False, no_auto=True)
            bin_size        - The time bin used to gather the spikes
            average         - If true, only the averaged CC among all the pairs is returned (less memory needed)
        
        Examples
            >> a.pairwise_cc(500, time_bin=1, averaged=True)
            >> a.pairwise_cc(500, time_bin=1, averaged=True, display=subplot(221), kwargs={'color':'r'})
            >> a.pairwise_cc(100, CustomPairs(a,a,[(i,i+1) for i in xrange(100)]), time_bin=5)
        
        See also
            pairwise_pearson_corrcoeff, pairwise_cc_zero, RandomPairs, AutoPairs, CustomPairs
        """
        
        ## We have to extract only the non silent cells, to avoid problems
        if pairs_generator is None:
            pairs_generator = RandomPairs(self, self, False, True)

        # Then we select the pairs of cells
        pairs_generator.get_pairs(nb_pairs)
        N      = len(pairs)
        bins   = arange(t_start, t_stop, bin_size)
        M      = len(nbins)
        length = 2*(M-1)
        if not average:
            results = numpy.empty((N,length), float)
        else:
            results = numpy.empty(length, float)
        for idx, pair in enumerate(pairs_generator):
            # We need to avoid empty spike histogram, otherwise the ccf function
            # will give a nan vector
            hist_1 = pairs_generator.spk1[pairs[0]].time_histogram(bin_size)
            hist_2 = pairs_generator.spk2[pairs[1]].time_histogram(bin_size)
            if not average:
                results[idx,:] = analysis.ccf(hist_1, hist_2)
            else:
                results += analysis.ccf(hist_1,hist_2)
        if not average:
            return results
        else:
            return results/N