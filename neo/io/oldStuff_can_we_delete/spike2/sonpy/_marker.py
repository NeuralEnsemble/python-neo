def data(Chan, start=None, stop=None, timeunits='seconds', as_float=True):
	'''
	Reads data from an event/marker channel (ie, Event, Marker, AdcMark,
	RealMark, or TextMark channel) from a Son file.

	'start', 'stop' set the data limits to be returned, in blocks. If
	only start is set, data from only that block will be returned.

	'timeunits' scales time to appropriate unit. Valid options are 'ticks',
	'microseconds', 'milliseconds' or 'seconds'.

	'as_float' only makes sense for AdcMark or RealMark channels. If
	True, returns data as floating point values (scaling and applying
	offset for Adc data). Else, data will be of type int16.
	'''
	from scipy import io, zeros
	fid = Chan.fhead.fid
	blockheader = Chan.blockheader
	SizeOfHeader = 20 # Block header is 20 bytes long
	
	# ====================================
	# = Set start and end blocks to read =
	# ====================================
	if not start and not stop:
		startBlock, endBlock = 0, Chan.info.blocks
	elif start and not stop:
		startBlock, endBlock = start-1, start
	elif start and stop:
		startBlock, endBlock = start-1, min([stop, Chan.info.blocks])

	# == Sum of samples in required blocks ==
	nItems = sum(blockheader[4, startBlock:endBlock])
	
	# =============
	# = Read data =
	# =============
	#	+ Event data +
	if Chan.info.kind in [2, 3, 4]:
		# pre-allocate memory:
		timings = zeros(nItems, 'int32')
		# read data:
		pointer = 0
		for block in range(startBlock, endBlock):
			fid.seek(blockheader[0, block] + SizeOfHeader)
			timings[pointer : pointer+blockheader[4, block]] =\
					io.fread(fid, blockheader[4, block], 'l')
			pointer += blockheader[4, block]

	#	+ Marker data +
	elif Chan.info.kind == 5:
		# pre-allocate memory:
		timings = zeros(nItems, 'int32')
		markers = zeros([nItems, 4], 'uint8')
		# read data:
		count = 0
		for block in range(startBlock, endBlock):
			fid.seek(blockheader[0, block] + SizeOfHeader) # start of block
			for i in range(blockheader[4, block]):         # loop for each marker
				timings[count] = io.fread(fid, 1, 'l')     # time
				markers[count] = io.fread(fid, 4, 'B')     # 4x marker bytes
				count += 1
		markers = [chr(x) for x in markers[:,0]]

	#	+ AdcMark data +
	elif Chan.info.kind == 6:
		nValues = Chan.info.nExtra/2 # 2 because 2 bytes per int16 value
		# pre-allocate memory:
		timings = zeros(nItems, 'int32')
		markers = zeros([nItems, 4], 'uint8')
		adc     = zeros([nItems, nValues], 'int16')
		# read data:
		count = 0
		for block in range(startBlock, endBlock):
			fid.seek(blockheader[0, block] + SizeOfHeader) # start of block
			for i in range(blockheader[4, block]):         # loop for each marker
				timings[count] = io.fread(fid, 1, 'l')     # time
				markers[count] = io.fread(fid, 4, 'B')     # 4x marker bytes
				adc[count]     = io.fread(fid, nValues, 'h')
				count += 1
		if as_double:
			from _waveform import _adc_to_double
			adc = _adc_to_double(Chan, adc)

	#	+ RealMark data +
	elif Chan.info.kind == 7:
		nValues = Chan.info.nExtra/4 # each value has 4 bytes (single precision)
		# pre-allocate:
		timings = zeros(nItems, 'int32')
		markers = zeros([nItems, 4], 'uint8')
		real =    zeros([nItems, nValues], 'single')
		# read data:
		count = 0
		for block in range(startBlock, endBlock):
			fid.seek(blockheader[0, block] + SizeOfHeader) # start of block
			for i in range(blockheader[4, block]):         # loop for each marker
				timings[count] = io.fread(fid, 1, 'l')     # time
				markers[count] = io.fread(fid, 4, 'B')     # 4x marker bytes
				real[count]    = io.fread(fid, nValues, 'f')
				count += 1
		if not as_double:
			from _waveform import _real_to_adc
			real = _real_to_adc(real)

	#	+ TextMark data +
	elif Chan.info.kind == 8:
		# pre-allocate memory:
		timings = zeros(nItems, 'int32')
		markers = zeros([nItems, 4], 'uint8')
		text = zeros([nItems, Chan.info.nExtra], 'S1')
		# read data:
		count = 0
		for block in range(startBlock, endBlock):
			fid.seek(blockheader[0, block] + SizeOfHeader) # start of block
			for i in range(blockheader[4, block]):         # loop for each marker
				timings[count] = io.fread(fid, 1, 'l')     # time
				markers[count] = io.fread(fid, 4, 'B')     # 4x marker bytes
				text[count]    = io.fread(fid, Chan.info.nExtra, 'c')
				count += 1

	# ================
	# = Convert time =
	# ================
	timings = Chan.fhead._ticks_to_seconds(timings, timeunits)
	Chan.info.timeunits = timeunits
	Chan.info.Epochs = '%i--%i of %i block(s)'\
			%(startBlock+1, endBlock, Chan.info.blocks)

	# ===============
	# = Return data =
	# ===============
	if Chan.info.kind in [2, 3, 4]:
		data = timings
	elif Chan.info.kind == 5:
		data = zip(timings, markers)
	elif Chan.info.kind == 6:
		data = zip(timings, markers, adc)
	elif Chan.info.kind == 7:
		data = zip(timings, markers, real)
	elif Chan.info.kind == 8:
		data = zip(timings, markers, text)
	return data
