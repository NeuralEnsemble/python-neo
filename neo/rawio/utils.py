import mmap
import numpy as np


def get_memmap_shape(filename, dtype, num_channels=None, offset=0):
    dtype = np.dtype(dtype)
    with open(filename, mode="rb") as f:
        f.seek(0, 2)
        flen = f.tell()
        bytes = flen - offset
        if bytes % dtype.itemsize != 0:
            raise ValueError("Size of available data is not a multiple of the data-type size.")
        size = bytes // dtype.itemsize
        if num_channels is None:
            shape = (size,)
        else:
            shape = (size // num_channels, num_channels)
        return shape


def get_memmap_chunk_from_opened_file(fid, num_channels, start, stop, dtype, file_offset=0):
    """
    Utility function to get a chunk as a memmap array directly from an opened file.
    Using this instead memmap can avoid memmory consumption when multiprocessing.

    Similar mechanism is used in spikeinterface.

    """
    bytes_per_sample = num_channels * dtype.itemsize

    # Calculate byte offsets
    start_byte = file_offset + start * bytes_per_sample
    end_byte = file_offset + stop * bytes_per_sample

    # Calculate the length of the data chunk to load into memory
    length = end_byte - start_byte

    # The mmap offset must be a multiple of mmap.ALLOCATIONGRANULARITY
    memmap_offset, start_offset = divmod(start_byte, mmap.ALLOCATIONGRANULARITY)
    memmap_offset *= mmap.ALLOCATIONGRANULARITY

    # Adjust the length so it includes the extra data from rounding down
    # the memmap offset to a multiple of ALLOCATIONGRANULARITY
    length += start_offset

    memmap_obj = mmap.mmap(fid.fileno(), length=length, access=mmap.ACCESS_READ, offset=memmap_offset)

    arr = np.ndarray(
        shape=((stop - start), num_channels),
        dtype=dtype,
        buffer=memmap_obj,
        offset=start_offset,
    )

    return arr
