# -*- coding: utf-8 -*-
import numpy
from spiketrain import SpikeTrain
from neuron import Neuron
from pairgenerator import *

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
        
        if len(self) > 0 and (self._t_start is None or self._t_stop is None):
            self.__calc_startstop()    
    
    def __calc_startstop(self):
        """
        t_start and t_stop are shared for all neurons, so we take min and max values respectively.
        TO DO : check the t_start and t_stop parameters for a SpikeList. Is it commun to
        all the spikeTrains within the spikelist or each spikelistes do need its own.
        """
        if len(self) > 0:
            if self._t_start is None:
                start_times   = numpy.array([spk._t_start for spk in self], numpy.float32)
                self._t_start = start_times.min()
                for spk in self:
                    spk._t_start = self._t_start
            if self._t_stop is None:
                stop_times   = numpy.array([spk._t_stop for spk in self], numpy.float32)
                self._t_stop = stop_times.max()
                for spk in self:
                    spk._t_stop = self._t_stop
        else:
            raise Exception("No SpikeTrains")
                
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
    
    @property
    def t_stop(self):
        return self._t_stop
    
    @property
    def t_start(self):
        return self._t_start
    
    def __getitem__(self, id):
        if id in self.id_list:
            return self.spiketrains[id]
        else:
            raise Exception("Neuron with id %d is not present in this SpikeTrainList" %id)
    
    #def __setitem__(self, spktrain):
        #assert isinstance(spktrain, SpikeTrain), "A SpikeList object can only contain SpikeTrain objects"
        #self.spiketrains[spktrain.neuron.id] = spktrain
        #self.__calc_startstop()
    
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

    def __select_with_pairs__(self, nb_pairs, pairs_generator):
        """
        Internal function used to slice two SpikeList according to a list
        of pairs.  Return a list of pairs
        
        Inputs:
            nb_pairs        - an int specifying the number of cells desired
            pairs_generator - a pairs generator
        
        Examples:
            >> self.__select_with_pairs__(50, RandomPairs(spk1, spk2))
        
        See also
            RandomPairs, AutoPairs, CustomPairs
        """
        pairs  = pairs_generator.get_pairs(nb_pairs)
        spk1   = pairs_generator.spk1.id_slice(pairs[:,0])
        spk2   = pairs_generator.spk2.id_slice(pairs[:,1])
        return spk1, spk2, pairs
    
    def copy(self):
        """
        Return a copy of the SpikeList object
        """
        res = []
        for spk in self:
            res.append(spk)
        return SpikeTrainList(spiketrains=res, t_start=self._t_start, t_stop=self._t_stop)


    def duration(self):
        """
        Return the duration of the SpikeTrain
        """
        return self._t_stop - self._t_start
    
    def time_parameters(self):
        """
        Return the time parameters of the SpikeList (t_start, t_stop)
        """
        return (self.t_start, self.t_stop)
    
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

    def time_slice(self, t_start, t_stop):
        """
        Return a new SpikeList obtained by slicing between t_start and t_stop
        
        Inputs:
            t_start - begining of the new SpikeTrain, in ms.
            t_stop  - end of the new SpikeTrain, in ms.
        
        See also
            id_slice, interval_slice
        """
        data = []
        for spk in self:
            data.append(spk)
        return SpikeTrainList(spiketrains=data, interval=(t_start, t_stop))

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
                    id    = spk.neuron.id  
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
        for id in self.id_list:
            cell = self[id]
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


    def spike_histogram(self, t_start=None, t_stop=None, bin_size=0.01, normalized=True, average=False):
        """
        Generate an array with all the spike_histograms of all the SpikeTrains
        objects within the SpikeTrainList.
        
        Inputs:
            bin_size   - the time bin used to gather the data
            normalized - if True, the histogram are in Hz (spikes/second), otherwise they are
                         in spikes/bin        
            average    - If True, return a single vector of the average spike histogram over the whole SpikeList
        
        See also
            firing_rate, time_axis
        """
        if t_start is None:
            t_start = self._t_start
        if t_stop is None:
            t_stop = self._t_stop
        bins       = numpy.arange(t_start, t_stop, bin_size)
        N          = len(self)
        M          = len(bins)
        if not average:
            spike_hist = numpy.zeros((N, M-1), numpy.float32)
            for idx, spk in enumerate(self):
                spike_hist[idx,:], edges = spk.time_histogram(t_start, t_stop, bin_size, normalized)        
        else:
            spike_hist = numpy.zeros(M-1, numpy.float32)
            for idx, spk in enumerate(self):
                tmp, edges  = spk.time_histogram(t_start, t_stop, bin_size, normalized)
                spike_hist += tmp
            spike_hist /= N 
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
        fr, edges = self.spike_histogram(bin_size, normalized=True, display=display, kwargs=kwargs, average=average)
        return fr

    def rates_histogram(self, nbins=25, normalize=True):
        """
        Return an histogram of the mean firing rates within the SpikeTrainList.
        
        Inputs:
            bins    - the number of bins (between the min and max of the data) 
                      or a list/array containing the lower edges of the bins.
        
        See also
            mean_rate, mean_rates
        """
        rates         = self.mean_rates()
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
        firing_rate, edges = self.spike_histogram(bin_size, average=True)
        fano               = numpy.var(firing_rate)/numpy.mean(firing_rate)
        return fano
    
    def distance_victorpurpura(self, nb_pairs, pairs_generator=None, cost=0.5):
        """
        Function to calculate the Victor-Purpura distance averaged over N pairs in the SpikeList
        See J. D. Victor and K. P. Purpura,
            Nature and precision of temporal coding in visual cortex: a metric-space
            analysis.,
            J Neurophysiol,76(2):1310-1326, 1996
        
        Inputs:
            nb_pairs        - int specifying the number of pairs
            pairs_generator - The generator that will be used to draw the pairs. If None, a default one is
                              created as RandomPairs(spk, spk, no_silent=False, no_auto=True)
            cost            - The cost parameter. See the paper for more informations. BY default, set to 0.5
        
        See also
            RandomPairs, AutoPairs, CustomPairs
        """
        if pairs_generator is None:
            pairs_generator = RandomPairs(self, self, False, True)

        pairs = pairs_generator.get_pairs(nb_pairs)
        N     = len(pairs)
        distance   = 0.
        for idx in xrange(N):
            idx_1 = pairs[idx,0]
            idx_2 = pairs[idx,1]
            distance += pairs_generator.spk1[idx_1].distance_victorpurpura(pairs_generator.spk2[idx_2], cost)
        return distance/N
    
    
    def distance_kreuz(self, nb_pairs, pairs_generator=None, dt=0.1e-3):
        """
        Function to calculate the Kreuz/Politi distance between two spike trains
        See Kreuz, T.; Haas, J.S.; Morelli, A.; Abarbanel, H.D.I. & Politi, 
        A. Measuring spike train synchrony. 
        J Neurosci Methods, 2007, 165, 151-161

        Inputs:
            nb_pairs        - int specifying the number of pairs
            pairs_generator - The generator that will be used to draw the pairs. If None, a default one is
                              created as RandomPairs(spk, spk, no_silent=False, no_auto=True)
            dt              - The time bin used to discretized the spike times
        
        See also
            RandomPairs, AutoPairs, CustomPairs
        """
        if pairs_generator is None:
            pairs_generator = RandomPairs(self, self, False, True)

        pairs = pairs_generator.get_pairs(nb_pairs)
        N     = len(pairs)
        
        distance   = 0.
        for idx in xrange(N):
            idx_1 = pairs[idx,0]
            idx_2 = pairs[idx,1]
            distance += pairs_generator.spk1[idx_1].distance_kreuz(pairs_generator.spk2[idx_2], dt)
        return distance/N
    
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
            >> a.pairwise_cc(500, bin_size=1, averaged=True)
            >> a.pairwise_cc(500, bin_size=1, averaged=True, display=subplot(221), kwargs={'color':'r'})
            >> a.pairwise_cc(100, CustomPairs(a,a,[(i,i+1) for i in xrange(100)]), bin_size=5)
        
        See also
            pairwise_pearson_corrcoeff, pairwise_cc_zero, RandomPairs, AutoPairs, CustomPairs
        """
        
        ## We have to extract only the non silent cells, to avoid problems
        if pairs_generator is None:
            pairs_generator = RandomPairs(self, self, False, True)

        # Then we select the pairs of cells
        pairs_generator.get_pairs(nb_pairs)
        N      = len(pairs)
        bins   = numpy.arange(t_start, t_stop, bin_size)
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
            
    def pairwise_cc_zero(self, nb_pairs, pairs_generator=None, bin_size=5e-3, time_window=None):
        """
        Function to return the normalized cross correlation coefficient at zero time
        lag according to the method given in:
        See A. Aertsen et al, 
            Dynamics of neuronal firing correlation: modulation of effective connectivity
            J Neurophysiol, 61:900-917, 1989
        
        The coefficient is averaged over N pairs of cells. If time window is specified, compute
        the corr coeff on a sliding window, and therefore returns not a value but a vector.
        
        Inputs:
            nb_pairs        - int specifying the number of pairs
            pairs_generator - The generator that will be used to draw the pairs. If None, a default one is
                              created as RandomPairs(spk, spk, no_silent=False, no_auto=True)
            bin_size        - The time bin used to gather the spikes
            time_window     - None by default, and then a single number, the normalized CC is returned.
                              If this is a float, then size (in ms) of the sliding window used to 
                              compute the normalized cc. A Vector is then returned
            display         - if True, a new figure is created. Could also be a subplot. The averaged
                              spike_histogram over the whole population is then plotted
            kwargs          - dictionary contening extra parameters that will be sent to the plot 
                              function
        
        Examples:
            >> a.pairwise_cc_zero(100, bin_size=1)
                1.0
            >> a.pairwise_cc_zero(100, CustomPairs(a, a, [(i,i+1) for i in xrange(100)]), bin_size=1)
                0.45
            >> a.pairwise_cc_zero(100, RandomPairs(a, a, no_silent=True), bin_size=5, time_window=10, display=True)
        
        See also:
            pairwise_cc, pairwise_pearson_corrcoeff, RandomPairs, AutoPairs, CustomPairs
        """
        
        if pairs_generator is None:
            pairs_generator = RandomPairs(self, self, False, True)
        
        spk1, spk2, pairs = self.__select_with_pairs__(nb_pairs, pairs_generator)
        N = len(pairs)
        
        if spk1.time_parameters() != spk2.time_parameters():
            raise Exception("The two SpikeList must have common time axis !")
        
        num_bins     = int(numpy.round((self.t_stop-self.t_start)/bin_size)+1)
        mat_neur1    = numpy.zeros((num_bins,N),int)
        mat_neur2    = numpy.zeros((num_bins,N),int)
        times1, ids1 = spk1.convert("times, ids")
        times2, ids2 = spk2.convert("times, ids")
        
        cells_id     = spk1.id_list()
        for idx in xrange(len(cells_id)):
            ids1[numpy.where(ids1 == cells_id[idx])[0]] = idx
        cells_id     = spk2.id_list()
        for idx in xrange(len(cells_id)):
            ids2[numpy.where(ids2 == cells_id[idx])[0]] = idx
        times1  = numpy.array(((times1 - self.t_start)/bin_size),int)
        times2  = numpy.array(((times2 - self.t_start)/bin_size),int)
        mat_neur1[times1,ids1] = 1
        mat_neur2[times2,ids2] = 1
        if time_window:
            nb_pts   = int(time_window/bin_size)
            mat_prod = mat_neur1*mat_neur2
            cc_time  = numpy.zeros((num_bins-nb_pts),float)
            xaxis    = numpy.zeros((num_bins-nb_pts))
            M        = float(nb_pts*N)
            bound    = int(numpy.ceil(nb_pts/2))
            for idx in xrange(bound,num_bins-bound):
                s_min = idx-bound
                s_max = idx+bound
                Z = numpy.sum(numpy.sum(mat_prod[s_min:s_max]))/M
                X = numpy.sum(numpy.sum(mat_neur1[s_min:s_max]))/M
                Y = numpy.sum(numpy.sum(mat_neur2[s_min:s_max]))/M
                cc_time[s_min] = (Z-X*Y)/numpy.sqrt((X*(1-X))*(Y*(1-Y)))
                xaxis[s_min]   = bin_size*idx
        else:
            M = float(num_bins*N)
            X = len(times1)/M
            Y = len(times2)/M
            Z = numpy.sum(numpy.sum(mat_neur1*mat_neur2))/M
            return (Z-X*Y)/numpy.sqrt((X*(1-X))*(Y*(1-Y)))
