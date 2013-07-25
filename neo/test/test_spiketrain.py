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
    def result_spike_check(self, st, st_out, t_start_out, t_stop_out,
                           dtype, units):
        assert_arrays_equal(st, st_out)
        self.assertEqual(st.t_start, t_start_out)
        self.assertEqual(st.t_stop, t_stop_out)

        self.assertEqual(st.units, units)
        self.assertEqual(st.t_start.units, units)
        self.assertEqual(st.t_stop.units, units)

        self.assertEqual(st.dtype, dtype)
        self.assertEqual(st.t_stop.dtype, dtype)
        self.assertEqual(st.t_start.dtype, dtype)

    def test__create_empty(self):
        t_start = 0.0
        t_stop = 10.0
        st = SpikeTrain([], t_start=t_start, t_stop=t_stop, units='s')

        dtype = numpy.float64
        units = 1 * pq.s
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = [] * units
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_empty_no_t_start(self):
        t_start = 0.0
        t_stop = 10.0
        st = SpikeTrain([ ], t_stop=t_stop, units='s')

        dtype = numpy.float64
        units = 1 * pq.s
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = [] * units
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_list(self):
        times = range(10)
        t_start = 0.0*pq.s
        t_stop = 10000.0*pq.ms
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="ms")

        dtype = numpy.float64
        units = 1 * pq.ms
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times * units
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_list_set_dtype(self):
        times = range(10)
        t_start = 0.0*pq.s
        t_stop = 10000.0*pq.ms
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                        units="ms", dtype='f4')

        dtype = numpy.float32
        units = 1 * pq.ms
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.astype(dtype)
        st_out = pq.Quantity(times, units=units, dtype=dtype)
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_list_no_start_stop_units(self):
        times = range(10)
        t_start = 0.0
        t_stop = 10000.0
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="ms")

        dtype = numpy.float64
        units = 1 * pq.ms
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = times * units
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_list_no_start_stop_units_set_dtype(self):
        times = range(10)
        t_start = 0.0
        t_stop = 10000.0
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                        units="ms", dtype='f4')

        dtype = numpy.float32
        units = 1 * pq.ms
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = pq.Quantity(times, units=units, dtype=dtype)
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_array(self):
        times = numpy.arange(10)
        t_start = 0.0*pq.s
        t_stop = 10000.0*pq.ms
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s")

        dtype = numpy.int
        units = 1 * pq.s
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times * units
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_array_with_dtype(self):
        times = numpy.arange(10, dtype='f4')
        t_start = 0.0*pq.s
        t_stop = 10000.0*pq.ms
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s")

        dtype = numpy.float32
        units = 1 * pq.s
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times * units
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_array_set_dtype(self):
        times = numpy.arange(10)
        t_start = 0.0*pq.s
        t_stop = 10000.0*pq.ms
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                        units="s", dtype='f4')

        dtype = numpy.float32
        units = 1 * pq.s
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.astype(dtype)
        st_out = times.astype(dtype) * units
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_array_no_start_stop_units(self):
        times = numpy.arange(10)
        t_start = 0.0
        t_stop = 10000.0
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s")

        dtype = numpy.int
        units = 1 * pq.s
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = times * units
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_array_no_start_stop_units_with_dtype(self):
        times = numpy.arange(10, dtype='f4')
        t_start = 0.0
        t_stop = 10000.0
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s")

        dtype = numpy.float32
        units = 1 * pq.s
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = times * units
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_array_no_start_stop_units_set_dtype(self):
        times = numpy.arange(10)
        t_start = 0.0
        t_stop = 10000.0
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                        units="s", dtype='f4')

        dtype = numpy.float32
        units = 1 * pq.s
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times.astype(dtype) * units
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array(self):
        times = numpy.arange(10) * pq.ms
        t_start = 0.0*pq.s
        t_stop = 12.0*pq.ms
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop)

        dtype = numpy.float64
        units = 1 * pq.ms
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_with_dtype(self):
        times = numpy.arange(10, dtype='f4') * pq.ms
        t_start = 0.0*pq.s
        t_stop = 12.0*pq.ms
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop)

        dtype = numpy.float32
        units = 1 * pq.ms
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.astype(dtype)
        st_out = times.astype(dtype)
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_set_dtype(self):
        times = numpy.arange(10) * pq.ms
        t_start = 0.0*pq.s
        t_stop = 12.0*pq.ms
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                        dtype='f4')

        dtype = numpy.float32
        units = 1 * pq.ms
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.astype(dtype)
        st_out = times.astype(dtype)
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_no_start_stop_units(self):
        times = numpy.arange(10) * pq.ms
        t_start = 0.0
        t_stop = 12.0
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop)

        dtype = numpy.float64
        units = 1 * pq.ms
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = times
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_no_start_stop_units_with_dtype(self):
        times = numpy.arange(10, dtype='f4') * pq.ms
        t_start = 0.0
        t_stop = 12.0
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop)

        dtype = numpy.float32
        units = 1 * pq.ms
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times.astype(dtype)
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_no_start_stop_units_set_dtype(self):
        times = numpy.arange(10) * pq.ms
        t_start = 0.0
        t_stop = 12.0
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                        dtype='f4')

        dtype = numpy.float32
        units = 1 * pq.ms
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times.astype(dtype)
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)


    def test__create_from_quantity_array_units(self):
        times = numpy.arange(10) * pq.ms
        t_start = 0.0*pq.s
        t_stop = 12.0*pq.ms
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units='s')

        dtype = numpy.float64
        units = 1 * pq.s
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_units_with_dtype(self):
        times = numpy.arange(10, dtype='f4') * pq.ms
        t_start = 0.0*pq.s
        t_stop = 12.0*pq.ms
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                        units='s')

        dtype = numpy.float32
        units = 1 * pq.s
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.rescale(units).astype(dtype)
        st_out = times.rescale(units).astype(dtype)
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_units_set_dtype(self):
        times = numpy.arange(10) * pq.ms
        t_start = 0.0*pq.s
        t_stop = 12.0*pq.ms
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                        units='s', dtype='f4')

        dtype = numpy.float32
        units = 1 * pq.s
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.rescale(units).astype(dtype)
        st_out = times.rescale(units).astype(dtype)
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_units_no_start_stop_units(self):
        times = numpy.arange(10) * pq.ms
        t_start = 0.0
        t_stop = 12.0
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units='s')

        dtype = numpy.float64
        units = 1 * pq.s
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_units_no_start_stop_units_set_dtype(self):
        times = numpy.arange(10) * pq.ms
        t_start = 0.0
        t_stop = 12.0
        st = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                        units='s', dtype='f4')

        dtype = numpy.float32
        units = 1 * pq.s
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times.rescale(units).astype(dtype)
        self.result_spike_check(st, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_list_without_units_should_raise_ValueError(self):
        times = range(10)
        t_start = 0.0*pq.s
        t_stop = 10000.0*pq.ms
        self.assertRaises(ValueError, SpikeTrain, times,
                          t_start=t_start, t_stop=t_stop)

    def test__create_from_array_without_units_should_raise_ValueError(self):
        times = numpy.arange(10)
        t_start = 0.0*pq.s
        t_stop = 10000.0*pq.ms
        self.assertRaises(ValueError, SpikeTrain, times,
                          t_start=t_start, t_stop=t_stop)

    def test__create_from_array_with_incompatible_units_ValueError(self):
        times = numpy.arange(10) * pq.km
        t_start = 0.0*pq.s
        t_stop = 10000.0*pq.ms
        self.assertRaises(ValueError, SpikeTrain, times,
                          t_start=t_start, t_stop=t_stop)

    def test__create_with_times_outside_tstart_tstop_ValueError(self):
        t_start = 23
        t_stop = 77
        ok = SpikeTrain(numpy.arange(t_start, t_stop), units='ms',
                                     t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, SpikeTrain,
                          numpy.arange(t_start-5, t_stop), units='ms',
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, SpikeTrain,
                          numpy.arange(t_start, t_stop+5), units='ms',
                          t_start=t_start, t_stop=t_stop)

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


class TestSorting(unittest.TestCase):

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


class TestSlice(unittest.TestCase):

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

class TestTimeSlice(unittest.TestCase):

    def test_time_slice_typical(self):
        st = SpikeTrain([0.1,0.5,1.2,3.3,6.4,7] * pq.ms, t_stop=10.0)

        # time_slice spike train, keep sliced spike times
        # this is the typical time slice falling somewhere in the middle of spikes
        t_start = 0.12* pq.ms
        t_stop = 3.5 * pq.ms
        st2 = st.time_slice(t_start,t_stop)
        assert_arrays_equal(st2, SpikeTrain([0.5,1.2,3.3] * pq.ms, t_stop=3.3))

        # but keep everything else pristine
        self.assertEqual(st.name, st2.name)
        self.assertEqual(st.description, st2.description)
        self.assertEqual(st.annotations, st2.annotations)
        self.assertEqual(st.file_origin, st2.file_origin)
        self.assertEqual(st.dtype, st2.dtype)
        self.assertEqual(t_start, st2.t_start)
        self.assertEqual(t_stop, st2.t_stop)

    def test_time_slice_differnt_units(self):
        st = SpikeTrain([0.1,0.5,1.2,3.3,6.4,7] * pq.ms, t_stop=10.0)

        # time_slice spike train, keep sliced spike times
        t_start = 0.00012* pq.s
        t_stop = 0.0035 * pq.s
        st2 = st.time_slice(t_start,t_stop)
        assert_arrays_equal(st2, SpikeTrain([0.5,1.2,3.3] * pq.ms, t_stop=3.3))

        # but keep everything else pristine
        self.assertEqual(st.name, st2.name)
        self.assertEqual(st.description, st2.description)
        self.assertEqual(st.annotations, st2.annotations)
        self.assertEqual(st.file_origin, st2.file_origin)
        self.assertEqual(st.dtype, st2.dtype)
        self.assertEqual(t_start, st2.t_start)
        self.assertEqual(t_stop, st2.t_stop)

    def test_time_slice_matching_ends(self):
        st = SpikeTrain([0.1,0.5,1.2,3.3,6.4,7] * pq.ms, t_stop=10.0)

        # time_slice spike train, keep sliced spike times
        t_start = 0.1* pq.ms
        t_stop = 7.0 * pq.ms
        st2 = st.time_slice(t_start,t_stop)
        assert_arrays_equal(st, st2)

        # but keep everything else pristine
        self.assertEqual(st.name, st2.name)
        self.assertEqual(st.description, st2.description)
        self.assertEqual(st.annotations, st2.annotations)
        self.assertEqual(st.file_origin, st2.file_origin)
        self.assertEqual(st.dtype, st2.dtype)
        self.assertEqual(t_start, st2.t_start)
        self.assertEqual(t_stop, st2.t_stop)

    def test_time_slice_out_of_boundries(self):
        st = SpikeTrain([0.1,0.5,1.2,3.3,6.4,7] * pq.ms, t_stop=10.0,
            t_start=0.1)

        # time_slice spike train, keep sliced spike times
        t_start = 0.01* pq.ms
        t_stop = 70.0 * pq.ms
        st2 = st.time_slice(t_start,t_stop)
        assert_arrays_equal(st, st2)

        # but keep everything else pristine
        self.assertEqual(st.name, st2.name)
        self.assertEqual(st.description, st2.description)
        self.assertEqual(st.annotations, st2.annotations)
        self.assertEqual(st.file_origin, st2.file_origin)
        self.assertEqual(st.dtype, st2.dtype)
        self.assertEqual(st.t_start, st2.t_start)
        self.assertEqual(st.t_stop, st2.t_stop)

    def test_time_slice_empty(self):
        st = SpikeTrain([] * pq.ms, t_stop=10.0)

        # time_slice spike train, keep sliced spike times
        t_start = 0.01* pq.ms
        t_stop = 70.0 * pq.ms
        st2 = st.time_slice(t_start,t_stop)
        assert_arrays_equal(st, st2)

        # but keep everything else pristine
        self.assertEqual(st.name, st2.name)
        self.assertEqual(st.description, st2.description)
        self.assertEqual(st.annotations, st2.annotations)
        self.assertEqual(st.file_origin, st2.file_origin)
        self.assertEqual(st.dtype, st2.dtype)
        self.assertEqual(t_start, st2.t_start)
        self.assertEqual(st.t_stop, st2.t_stop)

    def test_time_slice_none_stop(self):
        st = SpikeTrain([0.1,0.5,1.2,3.3,6.4,7] * pq.ms, t_stop=10.0,
            t_start=0.1)

        # time_slice spike train, keep sliced spike times
        t_start = 1 * pq.ms
        st2 = st.time_slice(t_start,None)
        assert_arrays_equal([1.2,3.3,6.4,7] * pq.ms, st2)

        # but keep everything else pristine
        self.assertEqual(st.name, st2.name)
        self.assertEqual(st.description, st2.description)
        self.assertEqual(st.annotations, st2.annotations)
        self.assertEqual(st.file_origin, st2.file_origin)
        self.assertEqual(st.dtype, st2.dtype)
        self.assertEqual(t_start, st2.t_start)
        self.assertEqual(st.t_stop, st2.t_stop)

    def test_time_slice_none_start(self):
        st = SpikeTrain([0.1,0.5,1.2,3.3,6.4,7] * pq.ms, t_stop=10.0,
            t_start=0.1)

        # time_slice spike train, keep sliced spike times
        t_stop = 1 * pq.ms
        st2 = st.time_slice(None,t_stop)
        assert_arrays_equal([0.1,0.5] * pq.ms, st2)

        # but keep everything else pristine
        self.assertEqual(st.name, st2.name)
        self.assertEqual(st.description, st2.description)
        self.assertEqual(st.annotations, st2.annotations)
        self.assertEqual(st.file_origin, st2.file_origin)
        self.assertEqual(st.dtype, st2.dtype)
        self.assertEqual(st.t_start, st2.t_start)
        self.assertEqual(t_stop, st2.t_stop)

    def test_time_slice_none_both(self):
        st = SpikeTrain([0.1,0.5,1.2,3.3,6.4,7] * pq.ms, t_stop=10.0,
            t_start=0.1)

        # time_slice spike train, keep sliced spike times
        st2 = st.time_slice(None,None)
        assert_arrays_equal(st, st2)

        # but keep everything else pristine
        self.assertEqual(st.name, st2.name)
        self.assertEqual(st.description, st2.description)
        self.assertEqual(st.annotations, st2.annotations)
        self.assertEqual(st.file_origin, st2.file_origin)
        self.assertEqual(st.dtype, st2.dtype)
        self.assertEqual(st.t_start, st2.t_start)
        self.assertEqual(st.t_stop, st2.t_stop)

class TestAttributesAnnotations(unittest.TestCase):

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

class TestChanging(unittest.TestCase):

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
            copy=False, t_stop=101, dtype=numpy.float64)

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
        # This is compatible with both np and quantity default behavior.
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
        # This is compatible with both np and quantity default behavior.
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

    def test__rescale(self):
        data = [3,4,5] * pq.ms
        st = SpikeTrain(data, t_start=0.5, t_stop=10.0)
        newst = st.rescale(pq.s)
        assert_arrays_equal(st, newst)
        self.assertEqual(newst.units, 1 * pq.s)

    def test__rescale_same_units(self):
        data = [3,4,5] * pq.ms
        st = SpikeTrain(data, t_start=0.5, t_stop=10.0)
        newst = st.rescale(pq.ms)
        assert_arrays_equal(st, newst)
        self.assertEqual(newst.units, 1 * pq.ms)

    def test__rescale_incompatible_units_ValueError(self):
        data = [3,4,5] * pq.ms
        st = SpikeTrain(data, t_start=0.5, t_stop=10.0)
        self.assertRaises(ValueError, st.rescale, pq.m)


class TestMiscellaneous(unittest.TestCase):
    def test__different_dtype_for_t_start_and_array(self):
        data = numpy.array([0,9.9999999], dtype = numpy.float64) * pq.s
        data16 = data.astype(numpy.float16)
        data32 = data.astype(numpy.float32)
        data64 = data.astype(numpy.float64)
        t_start = data[0]
        t_stop = data[1]
        t_start16 = data[0].astype(dtype=numpy.float16)
        t_stop16 = data[1].astype(dtype=numpy.float16)
        t_start32 = data[0].astype(dtype=numpy.float32)
        t_stop32 = data[1].astype(dtype=numpy.float32)
        t_start64 = data[0].astype(dtype=numpy.float64)
        t_stop64 = data[1].astype(dtype=numpy.float64)
        t_start_custom = 0.0
        t_stop_custom = 10.0
        t_start_custom16 = numpy.array(t_start_custom, dtype=numpy.float16)
        t_stop_custom16 = numpy.array(t_stop_custom, dtype=numpy.float16)
        t_start_custom32 = numpy.array(t_start_custom, dtype=numpy.float32)
        t_stop_custom32 = numpy.array(t_stop_custom, dtype=numpy.float32)
        t_start_custom64 = numpy.array(t_start_custom, dtype=numpy.float64)
        t_stop_custom64 = numpy.array(t_stop_custom, dtype=numpy.float64)

        #This is OK.
        st = SpikeTrain(data64, copy=True, t_start=t_start, t_stop=t_stop)

        st = SpikeTrain(data16, copy=True, t_start=t_start, t_stop=t_stop,
                        dtype=numpy.float16)
        st = SpikeTrain(data16, copy=True, t_start=t_start, t_stop=t_stop,
                        dtype=numpy.float32)

        st = SpikeTrain(data32, copy=True, t_start=t_start, t_stop=t_stop,
                        dtype=numpy.float16)
        st = SpikeTrain(data32, copy=True, t_start=t_start, t_stop=t_stop,
                        dtype=numpy.float32)

        st = SpikeTrain(data32, copy=True, t_start=t_start16, t_stop=t_stop16)
        st = SpikeTrain(data32, copy=True, t_start=t_start16, t_stop=t_stop16,
                        dtype=numpy.float16)
        st = SpikeTrain(data32, copy=True, t_start=t_start16, t_stop=t_stop16,
                        dtype=numpy.float32)
        st = SpikeTrain(data32, copy=True, t_start=t_start16, t_stop=t_stop16,
                        dtype=numpy.float64)

        st = SpikeTrain(data32, copy=True, t_start=t_start32, t_stop=t_stop32)
        st = SpikeTrain(data32, copy=True, t_start=t_start32, t_stop=t_stop32,
                        dtype=numpy.float16)
        st = SpikeTrain(data32, copy=True, t_start=t_start32, t_stop=t_stop32,
                        dtype=numpy.float32)
        st = SpikeTrain(data32, copy=True, t_start=t_start32, t_stop=t_stop32,
                        dtype=numpy.float64)

        st = SpikeTrain(data32, copy=True, t_start=t_start64, t_stop=t_stop64,
                        dtype=numpy.float16)
        st = SpikeTrain(data32, copy=True, t_start=t_start64, t_stop=t_stop64,
                        dtype=numpy.float32)

        st = SpikeTrain(data16, copy=True,
                        t_start=t_start_custom, t_stop=t_stop_custom)
        st = SpikeTrain(data16, copy=True,
                        t_start=t_start_custom, t_stop=t_stop_custom,
                        dtype=numpy.float16)
        st = SpikeTrain(data16, copy=True,
                        t_start=t_start_custom, t_stop=t_stop_custom,
                        dtype=numpy.float32)
        st = SpikeTrain(data16, copy=True,
                        t_start=t_start_custom, t_stop=t_stop_custom,
                        dtype=numpy.float64)

        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom, t_stop=t_stop_custom)
        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom, t_stop=t_stop_custom,
                        dtype=numpy.float16)
        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom, t_stop=t_stop_custom,
                        dtype=numpy.float32)
        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom, t_stop=t_stop_custom,
                        dtype=numpy.float64)

        st = SpikeTrain(data16, copy=True,
                        t_start=t_start_custom, t_stop=t_stop_custom)
        st = SpikeTrain(data16, copy=True,
                        t_start=t_start_custom, t_stop=t_stop_custom,
                        dtype=numpy.float16)
        st = SpikeTrain(data16, copy=True,
                        t_start=t_start_custom, t_stop=t_stop_custom,
                        dtype=numpy.float32)
        st = SpikeTrain(data16, copy=True,
                        t_start=t_start_custom, t_stop=t_stop_custom,
                        dtype=numpy.float64)

        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom16, t_stop=t_stop_custom16)
        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom16, t_stop=t_stop_custom16,
                        dtype=numpy.float16)
        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom16, t_stop=t_stop_custom16,
                        dtype=numpy.float32)
        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom16, t_stop=t_stop_custom16,
                        dtype=numpy.float64)

        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom32, t_stop=t_stop_custom32)
        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom32, t_stop=t_stop_custom32,
                        dtype=numpy.float16)
        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom32, t_stop=t_stop_custom32,
                        dtype=numpy.float32)
        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom32, t_stop=t_stop_custom32,
                        dtype=numpy.float64)

        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom64, t_stop=t_stop_custom64)
        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom64, t_stop=t_stop_custom64,
                        dtype=numpy.float16)
        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom64, t_stop=t_stop_custom64,
                        dtype=numpy.float32)
        st = SpikeTrain(data32, copy=True,
                        t_start=t_start_custom64, t_stop=t_stop_custom64,
                        dtype=numpy.float64)

        #This use to bug - see ticket #38
        st = SpikeTrain(data16, copy=True, t_start=t_start, t_stop=t_stop)
        st = SpikeTrain(data16, copy=True, t_start=t_start, t_stop=t_stop,
                        dtype=numpy.float64)

        st = SpikeTrain(data32, copy=True, t_start=t_start, t_stop=t_stop)
        st = SpikeTrain(data32, copy=True, t_start=t_start, t_stop=t_stop,
                        dtype=numpy.float64)

        st = SpikeTrain(data32, copy=True, t_start=t_start64, t_stop=t_stop64)
        st = SpikeTrain(data32, copy=True, t_start=t_start64, t_stop=t_stop64,
                        dtype=numpy.float64)


if __name__ == "__main__":
    unittest.main()
