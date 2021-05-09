import tensorflow as tf
from utils import Conv2D as SpecializedConv2D

expand = lambda r: lambda x: (x,) * r if tf.rank(x) == 0 else x

class Border:
    default = None

    def __init__(self, width, stride=1, rank=2, register=None):
        assert self.default is not None
        self.rank, self.register = rank, register or tf.Variable
        width, stride = map(expand(rank), (width, stride))
        self.stride = tf.convert_to_tensor(stride)
        self.default = tf.convert_to_tensor(self.default)
        self.sizes = tuple(((i - 1) // 2, i // 2) for i in width)
        self.values = [[self.initialize(size, i, axis, bool(end))
                        for end, size in enumerate(self.sizes[axis])]
                       for axis, i in enumerate(width)]

    # @tf.function(input_signature=[tf.TensorSpec([None], tf.int32)])
    def __call__(self, shape):
        tf.debugging.assert_equal(tf.shape(shape), (self.rank,))
        return self.builder(shape, [])

    def builder(self, shape, built):
        axis = len(built)
        if axis == self.rank:
            built = tf.meshgrid(*built, indexing="ij")
            res = built[0]
            for i in built[1:]:
                res = self.compose(res, i)
            return res

        size, stride = shape[axis], self.stride[axis]
        (start, end), (ssize, esize) = self.values[axis], self.sizes[axis]
        start, end = self.conv(start, True), self.conv(end, False)
        if ssize + esize <= size:
            middle = tf.repeat(self.default, size - ssize - esize)
        elif tf.math.maximum(ssize, esize) <= size:
            msize = ssize + esize - size
            middle = self.overlap(start[-msize:], end[:msize])
            start, end = start[:-msize], end[msize:]
        else:
            middle = self.overlap(start[:size], end[esize - size:])
            start, end = tf.zeros((0,)), tf.zeros((0,))

        reps = (size + stride - 1) // stride
        pad = tf.nn.relu((reps - 1) * stride + ssize + esize + 1 - size)
        offset = ssize - pad // 2
        ssize, msize = map(tf.size, (start, middle))
        start = start[offset::stride]
        middle = middle[(stride + offset - ssize) % stride::stride]
        end = end[(stride + offset - ssize - msize) % stride::stride]

        return tf.concat((
            self.builder(shape, built + [start]),
            self.builder(shape, built + [middle]),
            self.builder(shape, built + [end]),
        ), len(built))

    def initialize(self, size, width, axis, end):
        raise NotImplementedError()

    @classmethod
    def conv(_, x, reverse):
        raise NotImplementedError()

    @classmethod
    def compose(_, a, b):
        raise NotImplementedError()

    @classmethod
    def overlap(_, a, b):
        raise NotImplementedError()

class BorderReweight(Border):
    default = 1.

    def initialize(self, size, width, axis, end):
        name = f"border_reweight_axis{axis}_{'end' if end else 'start'}"
        res = tf.range(width - size, width)[::-1 if end else 1]
        res = tf.cast((res + 1) / res, tf.float32)
        return self.register(res, name=name)

    @classmethod
    def conv(cls, x, reverse):
        return tf.math.cumprod(x, reverse=reverse)

    @classmethod
    def compose(_, a, b):
        return a * b

    @classmethod
    def overlap(_, a, b):
        return a * b / (a + b - a * b)

# double counts the corners, which is unavoidable with O(n) parameters for an
# n by n kernel
# Proof: if you conv a kernel with ones, you can recover all the kernel weights
# by subtracting adjacent values towards the relevant corner therefore you need
# O(n^2) information to suplement the lost corners
class BorderOffset(Border):
    default = 0.

    def initialize(self, size, width, axis, end):
        name = f"border_bias_axis{axis}_{'end' if end else 'start'}"
        return self.register(tf.zeros((size,)), name=name)

    @classmethod
    def conv(cls, x, reverse):
        return tf.math.cumsum(x, reverse=reverse)

    @classmethod
    def compose(_, a, b):
        return a + b

    @classmethod
    def overlap(_, a, b):
        return a + b

class BorderConv:
    def __init__(self, *args, activation=None, **kw):
        super().__init__(*args, **kw)
        assert self.padding == "same" and all(
            i == 1 for i in self.dilation_rate)
        self._activation = tf.keras.activations.get(activation)
        if self._activation is None:
            self._activation = lambda x: x

        self.small = bool(tf.reduce_all(self.kernel_size == tf.constant(1)))
        if self.small:
            return

        self.border_weight = BorderReweight(
            self.kernel_size, self.strides, self.rank)
        if self.use_bias:
            self.border_bias = BorderOffset(
                self.kernel_size, self.strides, self.rank)

    def builder(self, input_shape):
        input_shape = input_shape[2:] if self._channels_first else \
            input_shape[1:-1]
        weight = tf.expand_dims(self.border_weight(
            input_shape)[tf.newaxis, ...], self._get_channel_axis())
        bias = 0. if not self.use_bias else tf.expand_dims(self.border_bias(
            input_shape)[tf.newaxis, ...], self._get_channel_axis())
        return weight, bias

    def call(self, inputs):
        res = super().call(inputs)
        if self.small:
            return self._activation(res)

        weight, bias = self.builder(tf.shape(inputs))
        return self._activation(weight * res + bias)

class Conv1D(BorderConv, tf.keras.layers.Conv1D):
    pass

class Conv2D(BorderConv, SpecializedConv2D):
    pass

class Conv3D(BorderConv, tf.keras.layers.Conv3D):
    pass

class DepthwiseConv2D(BorderConv, tf.keras.layers.DepthwiseConv2D):
    pass
