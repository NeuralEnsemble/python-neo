from __future__ import absolute_import
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from neo.core.spiketrain import check_has_dimensions_time, SpikeTrain
import quantities as pq
import numpy
from neo.test.tools import assert_arrays_equal
import sys

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
        st = SpikeTrain([ ], t_start=t_start, t_stop=t_stop, units='s')
    
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
    
    def test__create_with_times_outside_tstart_tstop_should_raise_Exception(self):
        t_start = 23
        t_stop = 77
        ok = SpikeTrain(numpy.arange(t_start, t_stop), units='ms', t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, SpikeTrain, numpy.arange(t_start-5, t_stop), units='ms', t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, SpikeTrain, numpy.arange(t_start, t_stop+5), units='ms', t_start=t_start, t_stop=t_stop)
    
    def test_defaults(self):
        # default recommended attributes
        st = SpikeTrain([3,4,5], units='sec', t_stop=10.0)
        self.assertEqual(st.dtype, numpy.float)
        self.assertEqual(st.sampling_rate, 1.0 * pq.Hz)
        self.assertEqual(st.waveforms, None)
        self.assertEqual(st.left_sweep, None)
    
    def test_default_tstart(self):
        # t start defaults to zero
        st = SpikeTrain([3,4,5]*pq.s, t_stop=8000*pq.ms)
        self.assertEqual(st.t_start, 0.*pq.s)
        
        # unless otherwise specified
        st = SpikeTrain([3,4,5]*pq.s, t_start=2.0, t_stop=8)
        self.assertEqual(st.t_start, 2.*pq.s)
    
    def test_tstop_units_conversion(self):
        st = SpikeTrain([3,5,4]*pq.s, t_stop=10)
        self.assertEqual(st.t_stop, 10.*pq.s)
        
        st = SpikeTrain([3,5,4]*pq.s, t_stop=10000.*pq.ms)
        self.assertEqual(st.t_stop, 10.*pq.s)

        st = SpikeTrain([3,5,4], units='sec', t_stop=10000.*pq.ms)
        self.assertEqual(st.t_stop, 10.*pq.s)
    
    def test_sort(self):
        wf = numpy.array([[0., 1.], [2., 3.], [4., 5.]])
        st = SpikeTrain([3,4,5]*pq.s, waveforms=wf, name='n', t_stop=10.0)
        st.sort()
        assert_arrays_equal(st, [3,4,5]*pq.s)
        assert_arrays_equal(st.waveforms, wf)
        self.assertEqual(st.name, 'n')
        self.assertEqual(st.t_stop, 10.0 * pq.s)
        
        st = SpikeTrain([3,5,4]*pq.s, waveforms=wf, name='n', t_stop=10.0)
        st.sort()
        assert_arrays_equal(st, [3,4,5]*pq.s)
        assert_arrays_equal(st.waveforms, wf[[0,2,1]])
        self.assertEqual(st.name, 'n')
        self.assertEqual(st.t_start, 0.0 * pq.s)
        self.assertEqual(st.t_stop, 10.0 * pq.s)
    
    def test_slice(self):
        wf = numpy.array([[0., 1.], [2., 3.], [4., 5.]])
        st = SpikeTrain([3,4,5]*pq.s, waveforms=wf, name='n', arb='arbb', t_stop=10.0)
        
        # slice spike train, keep sliced spike times
        st2 = st[1:2]        
        assert_arrays_equal(st[1:2], st2)
        
        # but keep everything else pristine
        self.assertEqual(st.name, st2.name)
        self.assertEqual(st.description, st2.description)
        self.assertEqual(st.annotations, st2.annotations)
        self.assertEqual(st.file_origin, st2.file_origin)
        self.assertEqual(st.dtype, st2.dtype)
        self.assertEqual(st.t_start, st2.t_start)
        self.assertEqual(st.t_stop, st2.t_stop)
        
        # except we update the waveforms
        assert_arrays_equal(st.waveforms[1:2], st2.waveforms)

    def test_slice_to_end(self):
        wf = numpy.array([[0., 1.], [2., 3.], [4., 5.]])
        st = SpikeTrain([3,4,5]*pq.s, waveforms=wf, name='n', arb='arbb', t_stop=12.3)
        
        # slice spike train, keep sliced spike times
        st2 = st[1:]        
        assert_arrays_equal(st[1:], st2)
        
        # but keep everything else pristine
        self.assertEqual(st.name, st2.name)
        self.assertEqual(st.description, st2.description)
        self.assertEqual(st.annotations, st2.annotations)
        self.assertEqual(st.file_origin, st2.file_origin)
        self.assertEqual(st.dtype, st2.dtype)
        self.assertEqual(st.t_start, st2.t_start)
        self.assertEqual(st.t_stop, st2.t_stop)
        
        # except we update the waveforms
        assert_arrays_equal(st.waveforms[1:], st2.waveforms)
        

    def test_slice_from_beginning(self):
        wf = numpy.array([[0., 1.], [2., 3.], [4., 5.]])
        st = SpikeTrain([3,4,5]*pq.s, waveforms=wf, name='n', arb='arbb', t_stop=23.4*pq.s)
        
        # slice spike train, keep sliced spike times
        st2 = st[:2]        
        assert_arrays_equal(st[:2], st2)
        
        # but keep everything else pristine
        self.assertEqual(st.name, st2.name)
        self.assertEqual(st.description, st2.description)
        self.assertEqual(st.annotations, st2.annotations)
        self.assertEqual(st.file_origin, st2.file_origin)
        self.assertEqual(st.dtype, st2.dtype)
        self.assertEqual(st.t_start, st2.t_start)
        self.assertEqual(st.t_stop, st2.t_stop)
        
        # except we update the waveforms
        assert_arrays_equal(st.waveforms[:2], st2.waveforms)

    def test_slice_negative_idxs(self):
        wf = numpy.array([[0., 1.], [2., 3.], [4., 5.]])
        st = SpikeTrain([3,4,5]*pq.s, waveforms=wf, name='n', arb='arbb', t_stop=10.0)
        
        # slice spike train, keep sliced spike times
        st2 = st[:-1]        
        assert_arrays_equal(st[:-1], st2)
        
        # but keep everything else pristine
        self.assertEqual(st.name, st2.name)
        self.assertEqual(st.description, st2.description)
        self.assertEqual(st.annotations, st2.annotations)
        self.assertEqual(st.file_origin, st2.file_origin)
        self.assertEqual(st.dtype, st2.dtype)
        self.assertEqual(st.t_start, st2.t_start)
        self.assertEqual(st.t_stop, st2.t_stop)
        
        # except we update the waveforms
        assert_arrays_equal(st.waveforms[:-1], st2.waveforms)
    
    def test_set_universally_recommended_attributes(self):
        st = SpikeTrain([3,4,5], units='sec', name='Name', description='Desc',
            file_origin='crack.txt', t_stop=99.9)
        self.assertEqual(st.name, 'Name')
        self.assertEqual(st.description, 'Desc')
        self.assertEqual(st.file_origin, 'crack.txt')

    def test_autoset_universally_recommended_attributes(self):
        st = SpikeTrain([3,4,5]*pq.s, t_stop=10.0)
        self.assertEqual(st.name, None)
        self.assertEqual(st.description, None)
        self.assertEqual(st.file_origin, None)
    
    def testannotations(self):
        st = SpikeTrain([3,4,5]*pq.s, t_stop=11.1)
        self.assertEqual(st.annotations, {})
        
        st = SpikeTrain([3,4,5]*pq.s, t_stop=11.1, ratname='Phillippe')
        self.assertEqual(st.annotations, {'ratname': 'Phillippe'})
    
    def test_change_with_copy_default(self):
        # Default is copy = True
        # Changing spike train does not change data
        # Data source is quantity
        data = [3,4,5] * pq.s
        st = SpikeTrain(data, t_stop=100.0)
        st[0] = 99 * pq.s
        self.assertEqual(st[0], 99*pq.s)
        self.assertEqual(data[0], 3*pq.s)
    
    def test_change_with_copy_false(self):
        # Changing spike train also changes data, because it is a view
        # Data source is quantity
        data = [3,4,5] * pq.s
        st = SpikeTrain(data, copy=False, t_stop=100.0)
        st[0] = 99 * pq.s
        self.assertEqual(st[0], 99*pq.s)
        self.assertEqual(data[0], 99*pq.s)

    def test_change_with_copy_false_and_fake_rescale(self): 
        # Changing spike train also changes data, because it is a view
        # Data source is quantity
        data = [3000,4000,5000] * pq.ms
        # even though we specify units, it still returns a view
        st = SpikeTrain(data, units='ms', copy=False, t_stop=100000)
        st[0] = 99000 * pq.ms
        self.assertEqual(st[0], 99000*pq.ms)
        self.assertEqual(data[0], 99000*pq.ms)

    def test_change_with_copy_false_and_rescale_true(self):
        # When rescaling, a view cannot be returned        
        # Changing spike train also changes data, because it is a view
        data = [3,4,5] * pq.s
        self.assertRaises(ValueError, SpikeTrain, data, units='ms', copy=False, 
            t_stop=10000)

    def test_init_with_rescale(self):
        data = [3,4,5] * pq.s
        st = SpikeTrain(data, units='ms', t_stop=6000)
        self.assertEqual(st[0], 3000*pq.ms)
        self.assertEqual(st._dimensionality, pq.ms._dimensionality)
        self.assertEqual(st.t_stop, 6000*pq.ms)
    
    def test_change_with_copy_true(self):
        # Changing spike train does not change data
        # Data source is quantity
        data = [3,4,5] * pq.s
        st = SpikeTrain(data, copy=True, t_stop=100)
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
        st = SpikeTrain(data, units='sec', t_stop=100)
        st[0] = 99 * pq.s
        self.assertEqual(st[0], 99*pq.s)
        self.assertEqual(data[0], 3*pq.s)
    
    def test_change_with_copy_false_and_data_not_quantity(self):
        # Changing spike train also changes data, because it is a view
        # Data source is array
        # Array and quantity are tested separately because copy default
        # is different for these two.
        data = numpy.array([3, 4, 5])        
        st = SpikeTrain(data, units='sec', copy=False, dtype=numpy.int, t_stop=101)
        st[0] = 99 * pq.s
        self.assertEqual(st[0], 99*pq.s)
        self.assertEqual(data[0], 99)

    def test_change_with_copy_false_and_dtype_change(self):
        # You cannot change dtype and request a view
        data = numpy.array([3, 4, 5])        
        self.assertRaises(ValueError, SpikeTrain, data, units='sec',
            copy=False, t_stop=101)
    
    def test_change_with_copy_true_and_data_not_quantity(self):
        # Changing spike train does not change data
        # Data source is array
        # Array and quantity are tested separately because copy default
        # is different for these two.
        data = [3,4,5]
        st = SpikeTrain(data, units='sec', copy=True, t_stop=123.4)
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
        st = SpikeTrain(data, copy=True, t_stop=99.9)
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
        st = SpikeTrain(data, copy=False, t_stop=100.0)
        st2 = st[1:3]
        st2[0] = 99 * pq.s
        self.assertEqual(st[1], 99*pq.s)
        self.assertEqual(st2[0], 99*pq.s)
        self.assertEqual(data[1], 99*pq.s)

    def test__changing_spiketime_should_check_time_in_range(self):
        data = [3,4,5] * pq.ms
        st = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        self.assertRaises(ValueError, st.__setitem__, 0, 10.1*pq.ms)
        self.assertRaises(ValueError, st.__setitem__, 1, 5.0*pq.s)
        self.assertRaises(ValueError, st.__setitem__, 2, 5.0*pq.s)
        self.assertRaises(ValueError, st.__setitem__, 0, 0)

    def test__changing_multiple_spiketimes(self):
        data = [3,4,5] * pq.ms
        st = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        st[:] = [7,8,9] * pq.ms
        assert_arrays_equal(st, numpy.array([7,8,9]))

    def test__changing_multiple_spiketimes_should_check_time_in_range(self):
        data = [3,4,5] * pq.ms
        st = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        if sys.version_info[0] == 2:
            self.assertRaises(ValueError, st.__setslice__, 0, 3, [3,4,11] * pq.ms)
            self.assertRaises(ValueError, st.__setslice__, 0, 3, [0,4,5] * pq.ms)
    
    def test__different_dtype_for_t_start_and_array(self):
        data = numpy.array([0,9.9999999], dtype = numpy.float64) * pq.s
        #This is OK.
        st = SpikeTrain(data.astype(numpy.float64), copy=True, t_start=data[0], t_stop=data[1])
        #This use to bug
        st = SpikeTrain(data.astype(numpy.float32), copy=True, t_start=data[0], t_stop=data[1])
        
    


if __name__ == "__main__":
    unittest.main()
