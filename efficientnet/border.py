import tensorflow as tf
from functools import partial
from utils import Conv2D as SpecializedConv2D

expand = lambda r: lambda x: (x,) * r if tf.rank(x) == 0 else x

class Border(tf.Module):
    default = None

    def __init__(self, width, stride=1, rank=2, disjoint=False, register=None,
                 name=None):
        super().__init__(name=name)
        assert self.default is not None
        self.rank, self.register = rank, register or tf.Variable
        width, stride = map(expand(rank), (width, stride))
        self.stride = tf.convert_to_tensor(stride)
        self.sizes = tuple(((i - 1) // 2, i // 2) for i in width)
        self.disjoint, self.cache = disjoint, None
        with self.name_scope:
            self.values = [[self.initialize(size, i, axis, disjoint, bool(end))
                            for end, size in enumerate(self.sizes[axis])]
                           for axis, i in enumerate(width)]

    def slices(self, shape):
        tf.debugging.assert_equal(tf.size(shape), self.rank)
        res = []
        for axis in range(self.rank):
            size, stride = shape[axis], self.stride[axis]
            start, end = self.sizes[axis]

            reps = (size + stride - 1) // stride - 1
            pad = tf.nn.relu(reps * stride + start + end + 1 - size)
            offset, diff = start - pad // 2, size - start - end
            reset = (offset + end - size) % stride
            midset = (offset - start) % stride

            if diff >= 0:
                data = (0, (0, 0), (0, 0), (offset, start), (reset, end),
                        (diff - midset - 1) // stride + 1)
            elif tf.math.maximum(start, end) <= size:
                data = (1, (reset + size - end, start), (reset, -diff),
                        (offset, size - end), (midset - diff, end), 0)
            else:
                data = (2, (offset, size), (offset + end - size, end),
                        (0, 0), (0, 0), 0)

            res.append(tuple(slice(*i, stride) if type(i) is tuple else i
                             for i in data))
        return tuple(res)

    def set_shape(self, shape):
        self.cache = self.slices(shape)

    def __call__(self, shape=None):
        assert self.cache is not None or shape is not None
        return self.build(self.cache or self.slices(shape))

    def build(self, bounds, axis=0, sides=(), expand=()):
        if axis == self.rank:
            if not sides:
                return self.default(list(zip(*expand))[1])

            sides = tf.meshgrid(*sides, indexing="ij")
            res = sides[0]
            for side in sides[1:]:
                res = self.compose(res, side)
            for axis, repeats in expand:
                res = tf.repeat(tf.expand_dims(res, axis), repeats, axis)
            return res

        if self.disjoint:
            start, end = self.values[axis]
        else:
            start = self.conv(self.values[axis][0], True)
            end = self.conv(self.values[axis][1], False)
        builder, bound = partial(self.build, bounds, axis + 1), bounds[axis]
        if bound[0] == 0:
            built = builder(sides, expand + ((axis, bound[5]),))
            built = built if bound[3].start >= bound[3].stop else tf.concat((
                builder(sides + (start[bound[3]],), expand), built), axis)
            return built if bound[4].start >= bound[4].stop else tf.concat((
                built, builder(sides + (end[bound[4]],), expand)), axis)
        else:
            data = self.overlap(start[bound[1]], end[bound[2]])
            if bound[0] == 1:
                data = tf.concat((start[bound[3]], data, end[bound[4]]), 0)
            return builder(sides + (data,), expand)

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
    def initialize(self, size, width, axis, disjoint, end):
        name = f"border_reweight_axis{axis}_{'end' if end else 'start'}"
        res = tf.range(width - size, width)[::-1 if end else 1]
        res = tf.cast(width / res if disjoint else (res + 1) / res, tf.float32)
        return self.register(res, name=name)

    @classmethod
    def default(cls, *args, **kw):
        return tf.ones(*args, **kw)

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
# O(n^2) information to supplement the lost corners
class BorderOffset(Border):
    def initialize(self, size, width, axis, disjoint, end):
        name = f"border_bias_axis{axis}_{'end' if end else 'start'}"
        return self.register(tf.zeros((size,)), name=name)

    @classmethod
    def default(cls, *args, **kw):
        return tf.zeros(*args, **kw)

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
    def __init__(self, *args, activation=None, disjoint=False, **kw):
        super().__init__(*args, **kw)
        assert self.padding == "same" and all(
            i == 1 for i in self.dilation_rate)
        self.disjoint = disjoint
        self.small = bool(tf.reduce_all(self.kernel_size == tf.constant(1)))
        self.spacial = slice(2, None) if self._channels_first else slice(1, -1)
        self._activation = tf.keras.activations.get(activation)
        if self._activation is None:
            self._activation = lambda x: x

    def build(self, input_shape):
        super().build(input_shape)
        if self.small:
            return

        self.border_weight = BorderReweight(
            self.kernel_size, self.strides, self.rank, self.disjoint)
        self._border_weight_values = self.border_weight.values

        if self.use_bias:
            self.border_bias = BorderOffset(
                self.kernel_size, self.strides, self.rank, self.disjoint)
            self._border_bias_values = self.border_bias.values

        input_shape = input_shape[self.spacial]
        if None not in input_shape:
            self.border_weight.set_shape(input_shape)
            self.border_bias.set_shape(input_shape)

    def builder(self, shape):
        weight = tf.expand_dims(self.border_weight(
            shape)[tf.newaxis, ...], self._get_channel_axis())
        bias = 0. if not self.use_bias else tf.expand_dims(
            self.border_bias(shape)[tf.newaxis, ...], self._get_channel_axis())
        return weight, bias

    def call(self, inputs):
        res = super().call(inputs)
        if self.small:
            return self._activation(res)

        weight, bias = self.builder(tf.shape(inputs)[self.spacial])
        return self._activation(weight * res + bias)

class Conv1D(BorderConv, tf.keras.layers.Conv1D):
    pass

class Conv2D(BorderConv, SpecializedConv2D):
    pass

class Conv3D(BorderConv, tf.keras.layers.Conv3D):
    pass

class DepthwiseConv2D(BorderConv, tf.keras.layers.DepthwiseConv2D):
    pass
