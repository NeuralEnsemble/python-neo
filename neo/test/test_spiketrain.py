from __future__ import absolute_import
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from neo.core.spiketrain import check_has_dimensions_time, SpikeTrain
import quantities as pq
import numpy
import numpy as np
from neo.test.tools import assert_arrays_equal

class TestFunctions(unittest.TestCase):

    def test__check_has_dimensions_time(self):
        a = numpy.arange(3) * pq.ms
        b = numpy.arange(3) * pq.mV
        c = numpy.arange(3) * pq.mA
        d = numpy.arange(3) * pq.minute
        check_has_dimensions_time(a)
        self.assertRaises(ValueError, check_has_dimensions_time, b)
        self.assertRaises(ValueError, check_has_dimensions_time, c)
        check_has_dimensions_time(d)
        self.assertRaises(ValueError, check_has_dimensions_time, a, b, c, d)
        

class TestConstructor(unittest.TestCase):
    def test__create_empty(self):
        t_start = 0.0
        t_stop = 10.0
        st = SpikeTrain([ ], t_start=t_start, t_stop=t_stop, units = 's')
    
    def test__create_from_list(self):
        times = range(10)
        t_start = 0.0
        t_stop = 10.0
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="ms")
        self.assertEqual(st.t_start, t_start*pq.ms)
        self.assertEqual(st.t_stop, t_stop*pq.ms)
        assert_arrays_equal(st, times*pq.ms)

    def test__create_from_array(self):
        times = numpy.arange(10)
        t_start = 0.0*pq.s
        t_stop = 10000.0*pq.ms
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s")
        self.assertEqual(st.t_stop, t_stop)
        assert_arrays_equal(st, times*pq.s)

    def test__create_from_array_without_units_should_raise_ValueError(self):
        times = numpy.arange(10)
        t_start = 0.0*pq.s
        t_stop = 10.0*pq.s
        self.assertRaises(ValueError, SpikeTrain, times, t_start=t_start, t_stop=t_stop)

    def test__create_from_quantity_array(self):
        times = numpy.arange(10) * pq.ms
        t_start = 0.0*pq.ms
        t_stop = 12.0*pq.ms
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop)
        assert_arrays_equal(st, times)
    
    def test_defaults(self):
        # default recommended attributes
        st = SpikeTrain([3,4,5], units='sec')
        self.assertEqual(st.dtype, numpy.float)
        self.assertEqual(st.sampling_rate, 1.0 * pq.Hz)
        self.assertEqual(st.waveforms, None)
        self.assertEqual(st.left_sweep, None)
    
    def test_default_tstart(self):
        # t start defaults to zero
        st = SpikeTrain([3,4,5]*pq.s)
        self.assertEqual(st.t_start, 0.*pq.s)
        
        # unless otherwise specified
        st = SpikeTrain([3,4,5]*pq.s, t_start=2.)
        self.assertEqual(st.t_start, 2.*pq.s)
    
    def test_default_tstop(self):
        st = SpikeTrain([3,4,5]*pq.s)
        self.assertEqual(st.t_stop, 5.*pq.s)
        
        st = SpikeTrain([3,5,4]*pq.s)
        self.assertEqual(st.t_stop, 5.*pq.s)
        
        st = SpikeTrain([3,5,4]*pq.s, t_stop=10)
        self.assertEqual(st.t_stop, 10.*pq.s)
        
        st = SpikeTrain([3,5,4]*pq.s, t_stop=10000.*pq.ms)
        self.assertEqual(st.t_stop, 10.*pq.s)

        st = SpikeTrain([3,5,4], units='sec', t_stop=10000.*pq.ms)
        self.assertEqual(st.t_stop, 10.*pq.s)
    
    def test_sort(self):
        wf = np.array([[0., 1.], [2., 3.], [4., 5.]])
        st = SpikeTrain([3,4,5]*pq.s, waveforms=wf, name='n')
        st.sort()
        assert_arrays_equal(st, [3,4,5]*pq.s)
        assert_arrays_equal(st.waveforms, wf)
        self.assertEqual(st.name, 'n')
        self.assertEqual(st.t_stop, 5.0 * pq.s)
        
        st = SpikeTrain([3,5,4]*pq.s, waveforms=wf, name='n')
        st.sort()
        assert_arrays_equal(st, [3,4,5]*pq.s)
        assert_arrays_equal(st.waveforms, wf[[0,2,1]])
        self.assertEqual(st.name, 'n')
        self.assertEqual(st.t_start, 0.0 * pq.s)
        self.assertEqual(st.t_stop, 5.0 * pq.s)
    
    def test_slice(self):
        wf = np.array([[0., 1.], [2., 3.], [4., 5.]])
        st = SpikeTrain([3,4,5]*pq.s, waveforms=wf)
        
        # slice spike train, keep sliced spike times
        st2 = st[1:2]        
        assert_arrays_equal(st[1:2], st2)
        
        # but keep everything else pristine
        self.assertEqual(st.name, st2.name)
        self.assertEqual(st.description, st2.description)
        self.assertEqual(st.annotations, st2.annotations)
        self.assertEqual(st.file_origin, st2.file_origin)
        self.assertEqual(st.dtype, st2.dtype)
        
        # even the things which we might expect to change
        assert_arrays_equal(st.waveforms, st2.waveforms)
        self.assertEqual(st.t_start, st2.t_start)
        self.assertEqual(st.t_stop, st2.t_stop)
    
    def test_set_universally_recommended_attributes(self):
        st = SpikeTrain([3,4,5], units='sec', name='Name', description='Desc',
            file_origin='crack.txt')
        self.assertEqual(st.name, 'Name')
        self.assertEqual(st.description, 'Desc')
        self.assertEqual(st.file_origin, 'crack.txt')

    def test_autoset_universally_recommended_attributes(self):
        st = SpikeTrain([3,4,5]*pq.s)
        self.assertEqual(st.name, None)
        self.assertEqual(st.description, None)
        self.assertEqual(st.file_origin, None)
    
    def testannotations(self):
        st = SpikeTrain([3,4,5]*pq.s)
        self.assertEqual(st.annotations, {})
        
        st = SpikeTrain([3,4,5]*pq.s, ratname='Phillippe')
        self.assertEqual(st.annotations, {'ratname': 'Phillippe'})
    
    def test_change_with_copy_default(self):
        # Default is copy = True
        # Changing spike train does not change data
        # Data source is quantity
        data = [3,4,5] * pq.s
        st = SpikeTrain(data)
        st[0] = 99 * pq.s
        self.assertEqual(st[0], 99*pq.s)
        self.assertEqual(data[0], 3*pq.s)
    
    def test_change_with_copy_false(self):
        # Changing spike train also changes data, because it is a view
        # Data source is quantity
        data = [3,4,5] * pq.s
        st = SpikeTrain(data, copy=False)
        st[0] = 99 * pq.s
        self.assertEqual(st[0], 99*pq.s)
        self.assertEqual(data[0], 99*pq.s)

    def test_change_with_copy_false_and_rescale_true(self):
        # Changing spike train also changes data, because it is a view
        # Data source is quantity
        data = [3,4,5] * pq.s
        st = SpikeTrain(data, units='ms', copy=False)
        st[0] = 99000 * pq.ms
        self.assertEqual(st[0], 99000*pq.ms)
        self.assertEqual(data[0], 99000*pq.ms)
    
    def test_change_with_copy_true(self):
        # Changing spike train does not change data
        # Data source is quantity
        data = [3,4,5] * pq.s
        st = SpikeTrain(data, copy=True)
        st[0] = 99 * pq.s
        self.assertEqual(st[0], 99*pq.s)
        self.assertEqual(data[0], 3*pq.s)

    def test_change_with_copy_default_and_data_not_quantity(self):
        # Default is copy = True
        # Changing spike train does not change data
        # Data source is array
        # Array and quantity are tested separately because copy default
        # is different for these two.
        data = [3,4,5]
        st = SpikeTrain(data, units='sec')
        st[0] = 99 * pq.s
        self.assertEqual(st[0], 99*pq.s)
        self.assertEqual(data[0], 3*pq.s)
    
    def test_change_with_copy_false_and_data_not_quantity(self):
        # Changing spike train also changes data, because it is a view
        # Data source is array
        # Array and quantity are tested separately because copy default
        # is different for these two.
        data = numpy.array([3.0, 4.0, 5.0]) # must be float, otherwise will get copy, not view
        st = SpikeTrain(data, units='sec', copy=False)
        st[0] = 99 * pq.s
        self.assertEqual(st[0], 99*pq.s)
        self.assertEqual(data[0], 99)
    
    def test_change_with_copy_true_and_data_not_quantity(self):
        # Changing spike train does not change data
        # Data source is array
        # Array and quantity are tested separately because copy default
        # is different for these two.
        data = [3,4,5]
        st = SpikeTrain(data, units='sec', copy=True)
        st[0] = 99 * pq.s
        self.assertEqual(st[0], 99*pq.s)
        self.assertEqual(data[0], 3)
    
    def test_changing_slice_changes_original_spiketrain(self):
        # If we slice a spiketrain and then change the slice, the
        # original spiketrain should change.
        # Whether the original data source changes is dependent on the
        # copy parameter.
        # This is compatible with both numpy and quantity default behavior.
        data = [3,4,5] * pq.s
        st = SpikeTrain(data, copy=True)
        st2 = st[1:3]
        st2[0] = 99 * pq.s
        self.assertEqual(st[1], 99*pq.s)
        self.assertEqual(st2[0], 99*pq.s)
        self.assertEqual(data[1], 4*pq.s)

    def test_changing_slice_changes_original_spiketrain_with_copy_false(self):
        # If we slice a spiketrain and then change the slice, the
        # original spiketrain should change.
        # Whether the original data source changes is dependent on the
        # copy parameter.
        # This is compatible with both numpy and quantity default behavior.
        data = [3,4,5] * pq.s
        st = SpikeTrain(data, copy=False)
        st2 = st[1:3]
        st2[0] = 99 * pq.s
        self.assertEqual(st[1], 99*pq.s)
        self.assertEqual(st2[0], 99*pq.s)
        self.assertEqual(data[1], 99*pq.s)




if __name__ == "__main__":
    unittest.main()