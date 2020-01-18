from gym import spaces
import numpy as np

class BoardBox(spaces.Box):
	"""General state space for board games"""
	def __init__(self, low_num, high_num, shape=None, dtype=np.int64, turn=0, draw=0, checkered=True):
		assert dtype is not None, 'dtype must be explicitly provided. '
		self.dtype = np.dtype(dtype)

		def build_checkered_board(w, h, c):
			re = np.r_[w * [1, 0]] * c  # even-numbered rows
			ro = np.r_[w * [0, 1]] * c  # odd-numbered rows
			return np.row_stack(h * (re, ro))

		if checkered:
			low = build_checkered_board(4, 4, low_num)
			high = build_checkered_board(4, 4, high_num)

		self.low_num = low_num
		self.high_num = high_num

		if shape is None:
			assert low.shape == high.shape, 'box dimension mismatch. '
			self.shape = low.shape
			self.low = low
			self.high = high
		else:
			assert np.isscalar(low) and np.isscalar(high), 'box requires scalar bounds. '
			self.shape = tuple(shape)
			self.low = np.full(self.shape, low)
			self.high = np.full(self.shape, high)

		def _get_precision(dtype):
			if np.issubdtype(dtype, np.floating):
				return np.finfo(dtype).precision
			else:
				return np.inf

		low_precision = _get_precision(self.low.dtype)
		high_precision = _get_precision(self.high.dtype)
		dtype_precision = _get_precision(self.dtype)
		if min(low_precision, high_precision) > dtype_precision:
			logger.warn("Box bound precision lowered by casting to {}".format(self.dtype))
		self.low = self.low.astype(self.dtype)
		self.high = self.high.astype(self.dtype)

		# Boolean arrays which indicate the interval type for each coordinate
		self.bounded_below = -np.inf < self.low
		self.bounded_above = np.inf > self.high

		self.draw = draw
		self.turn = turn

		super(spaces.Box, self).__init__(self.shape, self.dtype)

	def sample(self):
		high = self.high if self.dtype.kind == 'f' \
			else self.high.astype('int64') + 1
		sample = np.empty(self.shape)

		# Masking arrays which classify the coordinates according to interval
		# type
		unbounded = ~self.bounded_below & ~self.bounded_above
		upp_bounded = ~self.bounded_below & self.bounded_above
		low_bounded = self.bounded_below & ~self.bounded_above
		bounded = self.bounded_below & self.bounded_above

		# Vectorized sampling by interval type
		sample[unbounded] = self.np_random.normal(
			size=unbounded[unbounded].shape)

		sample[low_bounded] = self.np_random.exponential(
			size=low_bounded[low_bounded].shape) + self.low[low_bounded]

		sample[upp_bounded] = -self.np_random.exponential(
			size=upp_bounded[upp_bounded].shape) + self.high[upp_bounded]

		sample[bounded] = self.np_random.uniform(low=self.low[bounded],
																						 high=high[bounded],
																						 size=bounded[bounded].shape)
		if self.dtype.kind == 'i':
			sample = np.floor(sample)

		# Added this onto the original sample function

		X, Y = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]))
		Z = sample.astype(self.dtype)

		self.turn = np.random.choice([0, 1])
		self.draw = np.random.choice(np.arange(41))
		if self.draw == 0:
			self.draw = None

		new_sample = np.zeros((self.high_num + 3, self.shape[0], self.shape[1]))
		new_sample[Z, X, Y] = 1
		new_sample[-2] = self.turn

		if self.draw is not None:
			new_sample[-1, (self.draw - 1) // self.shape[0], (self.draw - 1) % self.shape[1]] = 1

		return new_sample[1:, :, :]