import builtins
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from neo.rawio.openephysbinaryrawio import (
    OpenEphysBinaryRawIO,
    _read_mtscomp_metadata,
    _resolve_continuous_storage,
    explore_folder,
)
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

import numpy as np
import pytest


def _write_synthetic_open_ephys_recording(
    dataset_folder,
    recordings,
    channel_names_by_stream,
    sample_rate=100.0,
    experiment_index=1,
):
    """Create a small raw Open Ephys dataset for compression tests."""
    dataset_folder = Path(dataset_folder)
    for recording_index, data_by_stream in enumerate(recordings, start=1):
        recording_folder = dataset_folder / f"experiment{experiment_index}" / f"recording{recording_index}"
        continuous_metadata = []
        for stream_name, data in data_by_stream.items():
            data = np.asarray(data, dtype=np.int16)
            stream_folder = recording_folder / "continuous" / stream_name
            stream_folder.mkdir(parents=True)
            data.tofile(stream_folder / "continuous.dat")
            np.save(stream_folder / "sample_numbers.npy", np.arange(data.shape[0], dtype=np.int64))

            channels = [
                {
                    "channel_name": channel_name,
                    "bit_volts": 0.195,
                    "units": "uV",
                }
                for channel_name in channel_names_by_stream[stream_name]
            ]
            continuous_metadata.append(
                {
                    "folder_name": stream_name,
                    "sample_rate": sample_rate,
                    "dtype": "int16",
                    "channels": channels,
                }
            )

        with open(recording_folder / "structure.oebin", "w", encoding="utf8") as file:
            json.dump({"continuous": continuous_metadata, "events": []}, file)

    return dataset_folder


def _compress_open_ephys_recording(source, destination, remove_dat=True, chunk_duration=1.0):
    """Copy an Open Ephys fixture and compress each continuous.dat in the copy."""
    mtscomp = pytest.importorskip("mtscomp")
    source = Path(source)
    destination = Path(destination)
    shutil.copytree(source, destination)

    for structure_path in destination.rglob("structure.oebin"):
        with open(structure_path, encoding="utf8") as file:
            structure = json.load(file)
        for stream_info in structure["continuous"]:
            stream_folder = structure_path.parent / "continuous" / stream_info["folder_name"]
            dat_path = stream_folder / "continuous.dat"
            mtscomp.compress(
                dat_path,
                stream_folder / "continuous.cbin",
                stream_folder / "continuous.ch",
                sample_rate=stream_info["sample_rate"],
                n_channels=len(stream_info["channels"]),
                dtype=np.int16,
                chunk_duration=chunk_duration,
                n_threads=1,
                check_after_compress=False,
                quiet=True,
            )
            if remove_dat:
                dat_path.unlink()

    return destination


class TestOpenEphysBinaryRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = OpenEphysBinaryRawIO
    entities_to_download = ["openephysbinary"]
    entities_to_test = [
        "openephysbinary/v0.5.3_two_neuropixels_stream",
        "openephysbinary/v0.4.4.1_with_video_tracking",
        "openephysbinary/v0.5.x_two_nodes",
        "openephysbinary/v0.6.x_neuropixels_multiexp_multistream",
        "openephysbinary/v0.6.x_neuropixels_with_sync",
        "openephysbinary/v0.6.x_neuropixels_missing_folders",
        "openephysbinary/v0.6.x_onebox_neuropixels",
        "openephysbinary/neural_and_non_neural_data_mixed",
    ]

    def test_sync(self):
        # The sync trace is always split off into its own -SYNC stream; the parent
        # AP stream has 384 channels (384 neural, SYNC excluded).
        rawio = OpenEphysBinaryRawIO(self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync"))
        rawio.parse_header()
        stream_name = [s_name for s_name in rawio.header["signal_streams"]["name"] if "AP" in s_name][0]
        stream_index = list(rawio.header["signal_streams"]["name"]).index(stream_name)

        chunk = rawio.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=stream_index
        )
        assert chunk.shape[1] == 384

    def test_sync_channel_access(self):
        """Sync channels are exposed as their own streams."""
        rawio = OpenEphysBinaryRawIO(self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync"))
        rawio.parse_header()

        # Find sync channel streams
        sync_stream_names = [s_name for s_name in rawio.header["signal_streams"]["name"] if "SYNC" in s_name]
        assert len(sync_stream_names) > 0, "No sync channel streams found"

        # Get the stream index for the first sync channel
        sync_stream_index = list(rawio.header["signal_streams"]["name"]).index(sync_stream_names[0])

        # Check that we can access the sync channel data
        chunk = rawio.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=100, stream_index=sync_stream_index
        )

        # Sync channel should have only one channel
        assert chunk.shape[1] == 1, f"Expected sync channel to have 1 channel, got {chunk.shape[1]}"

    def test_missing_folders(self):
        # missing folders should raise a warning
        with self.assertWarns(UserWarning):
            rawio = OpenEphysBinaryRawIO(self.get_local_path("openephysbinary/v0.6.x_neuropixels_missing_folders"))
            rawio.parse_header()

    def test_multiple_ttl_events_parsing(self):
        rawio = OpenEphysBinaryRawIO(self.get_local_path("openephysbinary/v0.6.x_neuropixels_with_sync"))
        rawio.parse_header()
        rawio.header = rawio.header
        # Testing co
        # This is the TTL events from the NI Board channel
        ttl_events = rawio._evt_streams[0][0][1]
        assert "rising" in ttl_events.keys()
        assert "labels" in ttl_events.keys()
        assert "durations" in ttl_events.keys()
        assert "timestamps" in ttl_events.keys()

        # Check that durations of different event streams are correctly parsed:
        assert np.allclose(ttl_events["durations"][ttl_events["labels"] == "1"], 0.5, atol=0.001)
        assert np.allclose(ttl_events["durations"][ttl_events["labels"] == "6"], 0.025, atol=0.001)
        assert np.allclose(ttl_events["durations"][ttl_events["labels"] == "7"], 0.016666, atol=0.001)

    def test_separating_stream_for_non_neural_data(self):
        rawio = OpenEphysBinaryRawIO(self.get_local_path("openephysbinary/neural_and_non_neural_data_mixed"))
        rawio.parse_header()
        # Check that the non-neural data stream is correctly separated
        assert len(rawio.header["signal_streams"]["name"]) == 2
        assert rawio.header["signal_streams"]["name"].tolist() == ["Rhythm_FPGA-100.0", "Rhythm_FPGA-100.0_ADC"]


def test_resolve_continuous_storage(tmp_path):
    dat_path = tmp_path / "continuous.dat"
    cbin_path = tmp_path / "continuous.cbin"
    ch_path = tmp_path / "continuous.ch"

    dat_path.touch()
    assert _resolve_continuous_storage(tmp_path) == {"type": "raw", "file_path": dat_path}

    cbin_path.touch()
    ch_path.touch()
    # Raw data takes precedence when both representations are present.
    assert _resolve_continuous_storage(tmp_path) == {"type": "raw", "file_path": dat_path}

    dat_path.unlink()
    assert _resolve_continuous_storage(tmp_path) == {
        "type": "mtscomp",
        "file_path": cbin_path,
        "metadata_path": ch_path,
    }


@pytest.mark.parametrize(
    ("present_filename", "match"),
    [
        ("continuous.cbin", "required metadata file"),
        ("continuous.ch", "no signal data file"),
        (None, "No signal data file"),
    ],
)
def test_resolve_continuous_storage_missing_files(tmp_path, present_filename, match):
    if present_filename is not None:
        (tmp_path / present_filename).touch()
    with pytest.raises(FileNotFoundError, match=match):
        _resolve_continuous_storage(tmp_path)


def test_raw_and_mtscomp_open_ephys_signals_are_equal(tmp_path):
    sample_count = 245
    probe_data = (np.arange(sample_count * 3, dtype=np.int16) - 400).reshape(sample_count, 3)
    mixed_data = (np.arange(sample_count * 2, dtype=np.int16) + 800).reshape(sample_count, 2)
    raw_folder = _write_synthetic_open_ephys_recording(
        tmp_path / "raw",
        [{"ProbeA-AP": probe_data, "Rhythm": mixed_data}],
        {
            "ProbeA-AP": ["AP0", "AP1", "Probe_SYNC"],
            "Rhythm": ["CH0", "ADC0"],
        },
    )
    compressed_folder = _compress_open_ephys_recording(raw_folder, tmp_path / "compressed")

    # Even invalid compressed artifacts must not affect a valid raw recording.
    for dat_path in raw_folder.rglob("continuous.dat"):
        (dat_path.parent / "continuous.cbin").write_bytes(b"not mtscomp data")
        (dat_path.parent / "continuous.ch").write_text("not JSON", encoding="utf8")

    _, raw_streams, _, _, _ = explore_folder(raw_folder)
    raw_stream_info = raw_streams[0][0]["continuous"]["ProbeA-AP"]
    assert raw_stream_info["raw_filename"] == raw_stream_info["data_filename"]
    assert raw_stream_info["raw_filename"].endswith("continuous.dat")

    _, compressed_streams, _, _, _ = explore_folder(compressed_folder)
    compressed_stream_info = compressed_streams[0][0]["continuous"]["ProbeA-AP"]
    assert "raw_filename" not in compressed_stream_info
    assert compressed_stream_info["data_filename"].endswith("continuous.cbin")

    raw_reader = OpenEphysBinaryRawIO(raw_folder)
    compressed_reader = OpenEphysBinaryRawIO(compressed_folder)
    raw_reader.parse_header()
    compressed_reader.parse_header()

    assert raw_reader.uses_mtscomp is False
    assert compressed_reader.uses_mtscomp is True
    with pytest.raises(AttributeError):
        compressed_reader.uses_mtscomp = False
    assert not list(compressed_folder.rglob("continuous.dat"))
    np.testing.assert_array_equal(raw_reader.header["signal_buffers"], compressed_reader.header["signal_buffers"])
    np.testing.assert_array_equal(raw_reader.header["signal_streams"], compressed_reader.header["signal_streams"])
    np.testing.assert_array_equal(raw_reader.header["signal_channels"], compressed_reader.header["signal_channels"])
    raw_buffer_description = raw_reader.get_analogsignal_buffer_description(0, 0, "0")
    assert set(raw_buffer_description) == {
        "type",
        "file_path",
        "dtype",
        "order",
        "file_offset",
        "shape",
    }
    assert raw_buffer_description["type"] == "raw"
    assert raw_buffer_description["file_path"].endswith("continuous.dat")
    assert raw_reader.segment_t_start(0, 0) == compressed_reader.segment_t_start(0, 0)
    assert raw_reader.segment_t_stop(0, 0) == compressed_reader.segment_t_stop(0, 0)

    for stream_index in range(raw_reader.signal_streams_count()):
        signal_size = raw_reader.get_signal_size(0, 0, stream_index)
        assert signal_size == compressed_reader.get_signal_size(0, 0, stream_index)
        for i_start, i_stop in ((0, 1), (signal_size - 1, signal_size), (95, 105), (35, 190)):
            raw_chunk = raw_reader.get_analogsignal_chunk(0, 0, i_start, i_stop, stream_index)
            compressed_chunk = compressed_reader.get_analogsignal_chunk(0, 0, i_start, i_stop, stream_index)
            np.testing.assert_array_equal(raw_chunk, compressed_chunk)

    stream_names = raw_reader.header["signal_streams"]["name"].tolist()
    neural_stream_index = stream_names.index("ProbeA-AP")
    mixed_neural_stream_index = stream_names.index("Rhythm")
    adc_stream_index = stream_names.index("Rhythm_ADC")
    sync_stream_index = stream_names.index("ProbeA-APSYNC")

    neural = compressed_reader.get_analogsignal_chunk(0, 0, 0, sample_count, neural_stream_index)
    mixed_neural = compressed_reader.get_analogsignal_chunk(0, 0, 0, sample_count, mixed_neural_stream_index)
    adc = compressed_reader.get_analogsignal_chunk(0, 0, 0, sample_count, adc_stream_index)
    sync = compressed_reader.get_analogsignal_chunk(0, 0, 0, sample_count, sync_stream_index)
    np.testing.assert_array_equal(neural, probe_data[:, :2])
    np.testing.assert_array_equal(mixed_neural, mixed_data[:, :1])
    np.testing.assert_array_equal(adc, mixed_data[:, 1:2])
    np.testing.assert_array_equal(sync, probe_data[:, 2:3])

    contiguous = compressed_reader.get_analogsignal_chunk(
        0, 0, 20, 40, neural_stream_index, channel_indexes=slice(0, 2)
    )
    non_contiguous = compressed_reader.get_analogsignal_chunk(0, 0, 20, 40, neural_stream_index, channel_indexes=[1, 0])
    np.testing.assert_array_equal(contiguous, probe_data[20:40, :2])
    np.testing.assert_array_equal(non_contiguous, probe_data[20:40, [1, 0]])
    assert not list(compressed_folder.rglob("continuous.dat"))


def test_mtscomp_readers_are_cached_per_block_segment_and_buffer(tmp_path):
    channel_names_by_stream = {"AP": ["AP0", "AP1"], "LFP": ["LFP0"]}
    raw_folder = tmp_path / "raw_multi"
    expected = {}
    for block_index in range(2):
        recordings = []
        expected[block_index] = {}
        for seg_index, sample_count in enumerate((125 + 10 * block_index, 217 + 10 * block_index)):
            offset = 1000 * (2 * block_index + seg_index)
            ap = (np.arange(sample_count * 2, dtype=np.int16) + offset).reshape(sample_count, 2)
            lfp = (np.arange(sample_count, dtype=np.int16) - offset).reshape(sample_count, 1)
            recordings.append({"AP": ap, "LFP": lfp})
            expected[block_index][seg_index] = {"AP": ap, "LFP": lfp}

        _write_synthetic_open_ephys_recording(
            raw_folder,
            recordings,
            channel_names_by_stream,
            experiment_index=block_index + 1,
        )

    compressed_folder = _compress_open_ephys_recording(raw_folder, tmp_path / "compressed_multi")
    reader = OpenEphysBinaryRawIO(compressed_folder)
    reader.parse_header()

    stream_names = reader.header["signal_streams"]["name"].tolist()
    for block_index in range(2):
        for seg_index in range(2):
            for stream_name in ("AP", "LFP"):
                stream_index = stream_names.index(stream_name)
                first = reader.get_analogsignal_chunk(block_index, seg_index, 5, 25, stream_index)
                cached_reader = reader._mtscomp_analogsignal_buffers[block_index][seg_index][str(stream_index)]
                second = reader.get_analogsignal_chunk(block_index, seg_index, 10, 30, stream_index)
                assert reader._mtscomp_analogsignal_buffers[block_index][seg_index][str(stream_index)] is cached_reader
                np.testing.assert_array_equal(first, expected[block_index][seg_index][stream_name][5:25])
                np.testing.assert_array_equal(second, expected[block_index][seg_index][stream_name][10:30])

    cached_readers = [
        compressed_reader
        for block_readers in reader._mtscomp_analogsignal_buffers.values()
        for segment_readers in block_readers.values()
        for compressed_reader in segment_readers.values()
    ]
    assert len(cached_readers) == 8
    assert len({id(compressed_reader) for compressed_reader in cached_readers}) == 8


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda metadata: metadata.update(n_channels=3), "declares 2 channels.*declares 3"),
        (lambda metadata: metadata.update(dtype="float32"), "declares dtype int16.*declares float32"),
        (lambda metadata: metadata.update(sample_rate=123.0), "declares sample rate 100.0.*declares 123.0"),
        (lambda metadata: metadata.pop("chunk_bounds"), "missing required key chunk_bounds"),
        (lambda metadata: metadata.update(chunk_bounds=[0, 100, 50]), "monotonically increasing"),
        (lambda metadata: metadata.update(chunk_bounds=[0, "invalid"]), "integer sample indices"),
    ],
)
def test_invalid_mtscomp_metadata_fails_during_header_parsing(tmp_path, mutation, match):
    data = np.arange(400, dtype=np.int16).reshape(200, 2)
    raw_folder = _write_synthetic_open_ephys_recording(
        tmp_path / "raw",
        [{"AP": data}],
        {"AP": ["AP0", "AP1"]},
    )
    compressed_folder = _compress_open_ephys_recording(raw_folder, tmp_path / "compressed")
    metadata_path = next(compressed_folder.rglob("continuous.ch"))
    with open(metadata_path, encoding="utf8") as file:
        metadata = json.load(file)
    mutation(metadata)
    with open(metadata_path, "w", encoding="utf8") as file:
        json.dump(metadata, file)

    with pytest.raises(ValueError, match=match):
        OpenEphysBinaryRawIO(compressed_folder).parse_header()


def test_mtscomp_is_only_required_for_trace_access_and_reader_closes(tmp_path):
    data = np.arange(400, dtype=np.int16).reshape(200, 2)
    raw_folder = _write_synthetic_open_ephys_recording(
        tmp_path / "raw",
        [{"AP": data}],
        {"AP": ["AP0", "AP1"]},
    )
    structure_path = next(raw_folder.rglob("structure.oebin"))
    event_folder = structure_path.parent / "events" / "Messages"
    event_folder.mkdir(parents=True)
    np.save(event_folder / "timestamps.npy", np.array([10, 20], dtype=np.int64))
    np.save(event_folder / "text.npy", np.array([b"start", b"stop"]))
    with open(structure_path, encoding="utf8") as file:
        structure = json.load(file)
    structure["events"] = [
        {
            "folder_name": "Messages",
            "channel_name": "Messages",
            "sample_rate": 100.0,
        }
    ]
    with open(structure_path, "w", encoding="utf8") as file:
        json.dump(structure, file)

    compressed_folder = _compress_open_ephys_recording(raw_folder, tmp_path / "compressed")
    raw_reader = OpenEphysBinaryRawIO(raw_folder)
    reader = OpenEphysBinaryRawIO(compressed_folder)

    original_import = builtins.__import__

    def import_without_mtscomp(name, *args, **kwargs):
        if name == "mtscomp":
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_without_mtscomp):
        raw_reader.parse_header()
        reader.parse_header()
        raw_chunk = raw_reader.get_analogsignal_chunk(0, 0, 0, 10, 0)
        np.testing.assert_array_equal(raw_chunk, data[:10])

        assert raw_reader.event_count(0, 0, 0) == reader.event_count(0, 0, 0) == 2
        raw_events = raw_reader.get_event_timestamps(0, 0, 0)
        event_timestamps, event_durations, event_labels = reader.get_event_timestamps(0, 0, 0)
        np.testing.assert_array_equal(raw_events[0], event_timestamps)
        assert raw_events[1] is event_durations is None
        np.testing.assert_array_equal(raw_events[2], event_labels)
        np.testing.assert_array_equal(event_timestamps, [10, 20])
        assert event_durations is None
        np.testing.assert_array_equal(event_labels, ["start", "stop"])
        with pytest.raises(ImportError, match="pip install mtscomp"):
            reader.get_analogsignal_chunk(0, 0, 0, 10, 0)

    reader.get_analogsignal_chunk(0, 0, 0, 10, 0)
    mtscomp_reader = reader._mtscomp_analogsignal_buffers[0][0]["0"]
    reader.get_analogsignal_chunk(0, 0, 10, 20, 0)
    assert reader._mtscomp_analogsignal_buffers[0][0]["0"] is mtscomp_reader
    assert not mtscomp_reader.cdata.closed

    reader._close_mtscomp_analogsignal_buffers()
    assert mtscomp_reader.cdata.closed
    assert not hasattr(reader, "_mtscomp_analogsignal_buffers")


def test_read_mtscomp_metadata_rejects_invalid_scalar_fields(tmp_path):
    metadata_path = tmp_path / "continuous.ch"
    valid_metadata = {
        "n_channels": 2,
        "sample_rate": 100.0,
        "dtype": "int16",
        "chunk_bounds": [0, 10],
    }
    invalid_values = (
        ("n_channels", 0, "positive integer"),
        ("n_channels", True, "positive integer"),
        ("sample_rate", np.inf, "finite and positive"),
        ("sample_rate", 0, "finite and positive"),
        ("dtype", "not-a-dtype", "NumPy dtype"),
        ("chunk_bounds", [], "non-empty list"),
        ("chunk_bounds", [1, 10], "must start at 0"),
    )
    for key, value, match in invalid_values:
        metadata = valid_metadata.copy()
        metadata[key] = value
        with open(metadata_path, "w", encoding="utf8") as file:
            json.dump(metadata, file)
        with pytest.raises(ValueError, match=match):
            _read_mtscomp_metadata(metadata_path)


if __name__ == "__main__":
    unittest.main()
