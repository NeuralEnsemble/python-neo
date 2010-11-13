import numpy

# Function to calculate the euclidian distance between two positions
# For the moment, we suppose the cells to be located in the same grid
# of size NxN. Should then include a scaling parameter to allow
# distances between distincts populations ?
def distance(pos_1, pos_2, N=None):
    # If N is not None, it means that we are dealing with a toroidal space, 
    # and we have to take the min distance
    # on the torus.
    if N is None:
        dx = pos_1[0]-pos_2[0]
        dy = pos_1[1]-pos_2[1]
    else:
        dx = numpy.minimum(abs(pos_1[0]-pos_2[0]), N-(abs(pos_1[0]-pos_2[0])))
        dy = numpy.minimum(abs(pos_1[1]-pos_2[1]), N-(abs(pos_1[1]-pos_2[1])))
    return numpy.sqrt(dx*dx + dy*dy)


class PairsGenerator(object):
    """
    PairsGenerator(SpikeList, SpikeList, no_silent)
    This class defines the concept of PairsGenerator, that will be used by all
    the functions using pairs of cells. Functions get_pairs() will then be used
    to obtain pairs from the generator.

    Inputs:
        spk1      - First SpikeTrainList object to take cells from
        spk2      - Second SpikeTrainList object to take cells from
        no_silent - Boolean to say if only non silent cells should
                    be considered. False by default 
    
    Examples:
        >> p = PairsGenerator(spk1, spk1, True)
        >> p.get_pairs(100)
    
    See also AutoPairs, RandomPairs, CustomPairs, DistantDependentPairs
    """
    def __init__(self, spk1, spk2, no_silent=False):
        self.spk1      = spk1
        self.spk2      = spk2
        self.no_silent = no_silent
        self.pairs     = []
        self._get_id_lists()
    
    def _get_id_lists(self):
        self.ids_1 = set(self.spk1.id_list)
        self.ids_2 = set(self.spk2.id_list)
        if self.no_silent:
            n1 = set(self.spk1.select_ids("len(cell) == 0"))
            n2 = set(self.spk2.select_ids("len(cell) == 0"))
            self.ids_1 -= n1
            self.ids_2 -= n2
    
    def __len__(self):
        return len(self.pairs)
    
    def __iter__(self):
        return iter(self.pairs)
    
    def get_pairs(self, nb_pairs):
        """
        Function to obtain a certain number of cells from the generator
        
        Inputs:
            nb_pairs - int to specify the number of pairs desired
        
        Examples:
            >> res = p.get_pairs(100)
        """
        return _abstract_method(self)
    
    

class AutoPairs(PairsGenerator):
    """
    AutoPairs(SpikeList, SpikeList, no_silent). Inherits from PairsGenerator.
    Generator that will return pairs of the same elements (contained in the
    two SpikeList) selected twice. 
    
    Inputs:
        spk1      - First SpikeList object to take cells from
        spk2      - Second SpikeList object to take cells from
        no_silent - Boolean to say if only non silent cells should
                    be considered. False by default 
    
    Examples:
        >> p = AutoPairs(spk1, spk1, True)
        >> p.get_pairs(4)
            [[1,1],[2,2],[4,4],[5,5]]
    
    See also RandomPairs, CustomPairs, DistantDependentPairs
    """
    
    def __init__(self, spk1, spk2, no_silent=False):
        PairsGenerator.__init__(self, spk1, spk2, no_silent)
    
    def get_pairs(self, nb_pairs):
        cells = numpy.random.permutation(list(self.ids_1.intersection(self.ids_2)))
        N     = len(cells)
        if nb_pairs > N:
            if not self.no_silent:
                print "Only %d distinct pairs can be extracted. Turn no_silent to True." %N
        try:
            pairs      = numpy.zeros((N, 2),int)
            pairs[:,0] = cells[0:N]
            pairs[:,1] = cells[0:N]      
        except Exception:
            pairs  = array([None, None])
        return pairs


class RandomPairs(PairsGenerator):
    """
    RandomPairs(SpikeList, SpikeList, no_silent, no_auto). Inherits from PairsGenerator.
    Generator that will return random pairs of elements.
        
    Inputs:
        spk1      - First SpikeList object to take cells from
        spk2      - Second SpikeList object to take cells from
        no_silent - Boolean to say if only non silent cells should
                    be considered. False by default 
        no_auto   - Boolean to say if pairs with the same element (id,id) should
                    be remove. True by default, i.e those pairs are discarded
        
    Examples:
        >> p = RandomPairs(spk1, spk1, True, False)
        >> p.get_pairs(4)
            [[1,3],[2,5],[1,4],[5,5]]
        >> p = RandomPairs(spk1, spk1, True, True)
        >> p.get_pairs(3)
            [[1,3],[2,5],[1,4]]
        
        
    See also RandomPairs, CustomPairs, DistantDependentPairs
    """
    def __init__(self, spk1, spk2, no_silent=False, no_auto=True):
        PairsGenerator.__init__(self, spk1, spk2, no_silent)
        self.no_auto = no_auto
    
    def get_pairs(self, nb_pairs):
        cells1 = numpy.array(list(self.ids_1), int)
        cells2 = numpy.array(list(self.ids_2), int)
        pairs  = numpy.zeros((0,2), int)
        N1     = len(cells1)
        N2     = len(cells2)
        T      = min(N1,N2)
        while len(pairs) < nb_pairs:
            N = min(nb_pairs-len(pairs), T)
            tmp_pairs  = numpy.zeros((N, 2),int)
            tmp_pairs[:,0] = cells1[numpy.floor(numpy.random.uniform(0, N1, N)).astype(int)]
            tmp_pairs[:,1] = cells2[numpy.floor(numpy.random.uniform(0, N2, N)).astype(int)]
            if self.no_auto:
                idx = numpy.where(tmp_pairs[:,0] == tmp_pairs[:,1])[0]
                pairs = numpy.concatenate((pairs, numpy.delete(tmp_pairs, idx, axis=0)))
            else:
                pairs = numpy.concatenate((pairs, tmp_pairs))
        return pairs



class DistantDependentPairs(PairsGenerator):
    """
    DistantDependentPairs(SpikeList, SpikeList, no_silent, no_auto). Inherits from PairsGenerator.
    Generator that will return pairs of elements according to the distances between the cells. The
    dimensions attribute of the SpikeList should be not empty.
        
    Inputs:
        spk1      - First SpikeList object to take cells from
        spk2      - Second SpikeList object to take cells from
        no_silent - Boolean to say if only non silent cells should
                    be considered. False by default 
        no_auto   - Boolean to say if pairs with the same element (id,id) should
                    be remove. True by default, i.e those pairs are discarded
        length    - the lenght (in mm) covered by the extend of spk1 and spk2. Currently, spk1
                    and spk2 should cover the same surface. Default is spk1.length
        d_min     - the minimal distance between cells
        d_max     - the maximal distance between cells
            
    Examples:
        >> p = DistantDependentPairs(spk1, spk1, True, False)
        >> p.get_pairs(4, d_min=0, d_max = 50)
            [[1,3],[2,5],[1,4],[5,5]]
        >> p = DistantDependentPairs(spk1, spk1, True, True, lenght=1)
        >> p.get_pairs(3, d_min=0.25, d_max=0.35)
            [[1,3],[2,5],[1,4]]
        
        
    See also RandomPairs, CustomPairs, AutoPairs
    """
    def __init__(self, spk1, spk2, no_silent=False, no_auto=True, lenght=1., d_min=0, dmax=1e6):
        PairsGenerator.__init__(self, spk1, spk2, no_silent)
        self.lenght  = lenght
        self.no_auto = no_auto
        self.d_min   = d_min
        self.d_max   = d_max
    
    def set_bounds(self, d_min, d_max):
        self.d_min = d_min
        self.d_max = d_max
    
    def get_pairs(self, nb_pairs):
        """
        Function to obtain a certain number of cells from the generator
        
        Inputs:
            nb_pairs - int to specify the number of pairs desired
            
        The length parameter of the DistantDependentPairs should be defined correctly. It is the extent of the grid.
        
        Examples:
            >> res = p.get_pairs(100, 0.3, 0.5)
        """
        cells1 = numpy.array(list(self.ids_1), int)
        cells2 = numpy.array(list(self.ids_2), int)
        pairs  = numpy.zeros((0,2), int)
        N1     = len(cells1)
        N2     = len(cells2)
        T      = min(N1,N2)
        while len(pairs) < nb_pairs:
            N         = min(nb_pairs-len(pairs), T)
            cell1     = cells1[numpy.floor(numpy.random.uniform(0, N1, N)).astype(int)]
            cell2     = cells2[numpy.floor(numpy.random.uniform(0, N1, N)).astype(int)]
            pos_cell1 = numpy.array(self.spk1.id2position(cell1))*self.lenght/self.spk1.dimensions[0]
            pos_cell2 = numpy.array(self.spk2.id2position(cell2))*self.lenght/self.spk2.dimensions[0]
            dist      = distance(pos_cell1, pos_cell2, self.lenght)
            idx       = numpy.where((dist >= self.d_min) & (dist < self.d_max))[0]
            N         = len(idx)
            if N > 0:
                tmp_pairs = numpy.zeros((N, 2),int)
                tmp_pairs[:,0] = cell1[idx]
                tmp_pairs[:,1] = cell2[idx]
                if self.no_auto:
                    idx = numpy.where(tmp_pairs[:,0] == tmp_pairs[:,1])[0]
                    pairs = numpy.concatenate((pairs, numpy.delete(tmp_pairs, idx, axis=0)))
                else:
                    pairs = numpy.concatenate((pairs, tmp_pairs))
        return pairs


class CustomPairs(PairsGenerator):
    """
    CustomPairs(SpikeList, SpikeList, pairs). Inherits from PairsGenerator.
    Generator that will return custom pairs of elements.
        
    Inputs:
        spk1  - First SpikeList object to take cells from
        spk2  - Second SpikeList object to take cells from
        pairs - A list of tuple that will be the pairs returned
                when get_pairs() function will be used.

    Examples:
        >> p = CustomPairs(spk1, spk1, [(i,i) for i in xrange(100)])
        >> p.get_pairs(4)
            [[1,1],[2,2],[3,3],[4,4]]
        
    See also RandomPairs, CustomPairs, DistantDependentPairs, AutoPairs
    """
    def __init__(self, spk1, spk2, pairs=[[],[]]):
        PairsGenerator.__init__(self, spk1, spk2)
        self.pairs = numpy.array(pairs)

    def get_pairs(self, nb_pairs):
        if nb_pairs > len(self.pairs):
            print "Trying to select too much pairs..."
        return self.pairs[0:nb_pairs] 