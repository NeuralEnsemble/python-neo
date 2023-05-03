import mmap
import numpy as np

def get_memmap_shape(filename, dtype, num_channels=None, offset=0):
    dtype = np.dtype(dtype)
    with open(filename, mode='rb') as f:
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

def create_memmap_buffer(fid, shape, dtype, offset=0):
    """
    A function that mimic the np.memmap but:
      * use an already opened file as input without checking the file size.
      * it handles also only the ready only case
    This should be faster.
    """
    dtype = np.dtype(dtype)
    size = np.prod(shape, dtype='int64')
    bytes = dtype.itemsize * size
    start = offset - offset % mmap.ALLOCATIONGRANULARITY
    bytes -= start
    array_offset = offset - start
    mmap_buffer = mmap.mmap(fid.fileno(), bytes, access=mmap.ACCESS_READ, offset=start)
    arr = np.ndarray.__new__(np.ndarray, shape, dtype=dtype, buffer=mmap_buffer, offset=array_offset, order='c')
    return arr
