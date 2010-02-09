def _adc_to_double(Chan, data):
	'''
	Scales a Son Adc channel (int16) to double precision floating point.

	Chan is a Channel instance.

	Applies the scale and offset supplied in Chan.fhead to the data
	contained in 'data'. These values are derived form the channel
	header on disc.
		
		OUT = (DATA * SCALE/6553.6) + OFFSET

	Chan.info will be updated with fields for the min and max values.
	'''
	if data.dtype.name != 'int16': raise TypeError, '16 bit integer expected'
	s = Chan.info.scale / 6553.6
	out = (data.astype('double') * s) + Chan.info.offset
	Chan.info.ymax = (data.max().astype('double') * s) + Chan.info.offset
	Chan.info.ymin = (data.min().astype('double') * s) + Chan.info.offset
	# Chan.info.kind = 9
	return out

def _real_to_adc(data):
	'''
	Converts a floating point array to int16.
	'''
	from scipy import array, polyfit
	# find min and max of input:
	a = array([data.min(), data.max()])
	# find slope and intercept for the line through (-32768, min)
	# and (32767, max):
	scale = polyfit([-32768, 32767], a, 1)
	data = (data-scale[1])/scale[0] # y = ax+b so find x = (y-b)/a
	data = data.round()             # round to nearest integer
	# check that int16 conversion can't lead to overflow:
	if (data.max() > 32767) | (data.min() < -32768):
		raise ValueError, 'Outside 16bit-integer range'
	data = data.astype('int16')    # convert to int16
	# hout.scale = scale[0]*6553.6 # adjust slope to conform to SON scale format...
	# hout.offset = scale[1]       # ... and set offset
	# hout.kind = 1                # set kind to ADC channel
	return data

def data(Chan, start=None, stop=None, timeunits='seconds', as_float=True):
	'''
	Reads data from a RealWave or Adc channel from a Son file.

	Chan is a Channel intance.

	'start', 'stop' set the data limits to be returned, in blocks for
	continuous data or in epochs for triggered data. If only start
	is set, data from only the block/epoch selected will be returned.

	'timeunits' scales time to appropriate unit. Valid options are 'ticks',
	'microseconds', 'milliseconds' or 'seconds'.

	'as_float=True' returns data as floating point values (scaling and
	applying offset for Adc channel data). Else, data will be of type int16.

	RETURNS data, which can be a simple vector (for continuously sampled
	data) or a two-dimensional matrix with each epoch (frame) of data
	in a separate row (if sampling was triggered).
	'''
	from scipy import array, io, histogram, zeros
	fid = Chan.fhead.fid
	blockheader = Chan.blockheader
	SizeOfHeader = 20 # block header is 20 bytes long
	# sample interval in clock ticks:
	SampleInterval = (blockheader[2,0]-blockheader[1,0])/(blockheader[4,0]-1)

	# =======================================================
	# = Set data types according to channel type to be read =
	# =======================================================
	if Chan.info.kind == 1:   datatype = 'h' # Adc channel, 'int16'
	elif Chan.info.kind == 9: datatype = 'f' # RealWave channel, 'single'

	# ============================================
	# = Check for discontinuities in data record =
	# ============================================
	NumFrames = 1 # number of frames. Initialize to one
	Frame = [1]
	for i in range(Chan.info.blocks-1): 
		IntervalBetweenBlocks = blockheader[1, i+1] - blockheader[2, i]
		if IntervalBetweenBlocks > SampleInterval: # if true, data is discontinuous
			NumFrames += 1          # count discontinuities (NumFrames)
			Frame.append(NumFrames) # record the frame number that each block belongs to
		else:
			Frame.append(Frame[i]) # pad between discontinuities
	Frame = array(Frame)

	# =================================
	# = Set start and stop boundaries =
	# =================================
	if not start and not stop:
		FramesToReturn = NumFrames
		Chan.info.npoints = zeros(FramesToReturn)
		startEpoch = 0 # read all data
		endEpoch = Chan.info.blocks
	elif start and not stop:
		if NumFrames == 1: # read one epoch
			startEpoch = start-1
			endEpoch = start
		else:
			FramesToReturn = 1
			Chan.info.npoints = 0
			indx = arange(Frame.size)
			startEpoch = indx[Frame == start][0]
			endEpoch = indx[Frame == start][-1] + 1
	elif start and stop:
		if NumFrames == 1: # read a range of epochs
			startEpoch = start-1
			endEpoch = stop
		else:
			FramesToReturn = stop-start + 1
			Chan.info.npoints = zeros(FramesToReturn)
			indx = arange(Frame.size)
			startEpoch = indx[Frame == start][0]
			endEpoch = indx[Frame == stop][-1] + 1
	
	# Make sure we are in range if using 'start' and 'stop'
	if (startEpoch > Chan.info.blocks) | (startEpoch > endEpoch):
		raise ValueError, 'Invalid start and/or stop'
	if endEpoch > Chan.info.blocks: endEpoch = Chan.info.blocks

	# =============
	# = Read data =
	# =============
	if NumFrames == 1:
		# ++ Continuous sampling - one frame only. Epochs correspond to
		# blocks in the SON file.
		# sum of samples in all blocks:
		NumberOfSamples = sum(blockheader[4, startEpoch:endEpoch])
		# pre-allocate memory for data:
		data = zeros(NumberOfSamples, datatype)
		# read data:
		pointer = 0
		for i in range(startEpoch, endEpoch):
			fid.seek(blockheader[0, i] + SizeOfHeader)
			data[pointer : pointer+blockheader[4, i]] =\
					io.fread(fid, blockheader[4, i], datatype)
			pointer += blockheader[4, i]
		# set extra channel information:
		Chan.info.mode    = 'continuous'
		Chan.info.epochs  = [startEpoch+1, endEpoch]
		Chan.info.npoints = NumberOfSamples
		Chan.info.start   = blockheader[1, startEpoch] # first data point (clock ticks)
		Chan.info.stop    = blockheader[2, endEpoch-1] # end of data (clock ticks)
		Chan.info.Epochs = '%i--%i of %i blocks' %(startEpoch+1, endEpoch,\
				Chan.info.blocks)

	else:
		# ++ Frame-based data -  multiple frames. Epochs correspond to
		# frames of data.
		# sum of samples in required epochs:
		NumberOfSamples = sum(blockheader[4, startEpoch:endEpoch])
		# maximum data points to a frame:
		FrameLength = histogram(Frame, range(startEpoch,endEpoch))[0].max() *\
				blockheader[4, startEpoch:endEpoch].max()
		# pre-allocate memory:
		data = zeros([FramesToReturn, FrameLength], datatype)
		Chan.info.start = zeros(FramesToReturn)
		Chan.info.stop = zeros(FramesToReturn)
		# read data:
		pointer = 0 # pointer into data array for each disk data block
		index = 0   # epoch counter
		for i in range(startEpoch, endEpoch):
			fid.seek(blockheader[0, i] + SizeOfHeader)
			data[index, pointer : pointer+blockheader[4, i]] =\
					io.fread(fid, blockheader[4, i], datatype)
			Chan.info.npoints[index] = Chan.info.npoints[index]+blockheader[4,i]
			try: Frame[i+1]
			except IndexError:
				Chan.info.stop[index] = blockheader[2, i] # time at eof
			else:
				if Frame[i+1] == Frame[i]:
					pointer += blockheader[4, i] # increment pointer or...
				else:
					Chan.info.stop[index] = blockheader[2, i] # end time for this frame
					if i < endEpoch-1:
						pointer = 0 # begin new frame
						index += 1
						# time of first data point in next frame (clock ticks):
						Chan.info.start[index] = blockheader[1, i+1]
		# set extra channel information:
		Chan.info.mode = 'triggered'
		Chan.info.start[0] = blockheader[1, startEpoch] 
		Chan.info.Epochs = '%i--%i of %i epochs' %(startEpoch+1, endEpoch,\
				NumFrames)

	# ================
	# = Convert time =
	# ================
	Chan.info.start = Chan.fhead._ticks_to_seconds(Chan.info.start, timeunits)
	Chan.info.stop =  Chan.fhead._ticks_to_seconds(Chan.info.stop, timeunits)
	Chan.info.timeunits = timeunits

	# =========================
	# = Scale and return data =
	# =========================
	if as_float and Chan.info.kind == 1:     # Adc
			data = _adc_to_double(Chan, data)
	if not as_float and Chan.info.kind == 9: # RealWave
			data = _real_to_adc(data)
	return data
