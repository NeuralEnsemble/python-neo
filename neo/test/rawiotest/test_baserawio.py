import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from neo.rawio.baserawio import BaseRawWithBufferApiIO, _signal_stream_dtype


class BufferReader(BaseRawWithBufferApiIO):
    def _get_analogsignal_buffer_description(self, block_index, seg_index, buffer_id):
        return self.description


@unittest.skipIf(h5py is None, "requires h5py")
class TestHDF5BufferSelection(unittest.TestCase):
    def test_stream_selection_applied_once(self):
        samples = np.arange(30, dtype="uint16").reshape(6, 5)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "signals.h5"
            with h5py.File(filename, "w") as file:
                file["time_first"] = samples
                file["channel_first"] = samples.T
            for time_axis, dataset in [(0, "time_first"), (1, "channel_first")]:
                for buffer_slice in [
                    None,
                    slice(0, -1),
                    slice(-1, None),
                    np.array([0, 2, 4]),
                    np.array([True, False, True, False, True]),
                ]:
                    with self.subTest(time_axis=time_axis, buffer_slice=buffer_slice):
                        reader = BufferReader()
                        reader.header = {
                            "signal_streams": np.array([("signals", "0", "0")], dtype=_signal_stream_dtype)
                        }
                        reader.description = {
                            "type": "hdf5",
                            "file_path": str(filename),
                            "hdf5_path": dataset,
                            "time_axis": time_axis,
                        }
                        reader._stream_buffer_slice = {"0": buffer_slice}
                        expected = samples[1:4]
                        if buffer_slice is not None:
                            expected = expected[:, buffer_slice]
                        for channel_indexes in [None, [0], slice(0, 1), [-1, 0]]:
                            selected = expected if channel_indexes is None else expected[:, channel_indexes]
                            np.testing.assert_array_equal(
                                reader._get_analogsignal_chunk(0, 0, 1, 4, 0, channel_indexes), selected
                            )
                        reader._hdf5_analogsignal_buffers[0][0]["0"].close()
