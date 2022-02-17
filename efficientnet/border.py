import tensorflow as tf
from .utils import Conv2D as SpecializedConv2D

def nslice(rank, dim):
    start = tuple(slice(None) for i in range(dim))
    end = tuple(slice(None) for i in range(rank - dim - 1))
    def inner(*args):
        return start + (slice(*args),) + end
    return inner

class Dimension(tf.Module):
    def __init__(self, rank, primary, kernel, stride, size=None, channels=1,
                 channel_axis=None, disjoint=False, register=None, name=None):
        name = name or f"axis{primary}"
        super().__init__(name=name)
        self.rank, self.primary, self.kernel = rank, primary, kernel
        self.stride, self.size, self.disjoint = stride, size, disjoint
        self.channels, self.channel_axis = channels, channel_axis
        self.register = register or tf.Variable
        with self.name_scope:
            self.initialize()

    def expand(self, tensor):
        shape = tf.ones((self.rank,), dtype=tf.int64)
        shape = tf.tensor_scatter_nd_update(shape, [[self.primary]], [-1])
        tensor = tf.reshape(tensor, shape)
        if self.channel_axis is not None:
            tensor = tf.repeat(tensor, self.channels, self.channel_axis)
        return tensor

    def group(self, tensor, start, prefix=False):
        flip = slice(None, None, -1) if prefix else slice(None)
        tensor = tf.concat((tensor, self.default((self.stride - 1,)))[flip], 0)
        end = tf.size(tensor) - (tf.size(tensor) - start) % self.stride
        return self.reduce(tf.reshape(tensor[start:end], (-1, self.stride)), 1)

    def consolidate(self, middle, start, end, dim=None, rank=None):
        if middle > 0:
            return middle, start, end

        dim = self.primary if dim is None else dim
        rank = self.rank if rank is None else rank
        empty = tf.constant([], shape=[int(i != dim) for i in range(rank)])
        if middle == 0:
            return 0, empty, tf.concat((start, end), dim)

        idx = nslice(rank, dim)
        over = self.overlap(start[idx(middle, None)], end[idx(-middle)])
        return 0, empty, tf.concat(
            (start[idx(middle)], over, end[idx(-middle, None)]), dim)

    def pad(self, size):
        pad = tf.nn.relu(self.kernel - 1 - ((size - 1) % self.stride))
        start = (self.kernel - 1) // 2 - pad // 2
        end = (pad - 1) // 2 % self.stride
        return -(-size // self.stride), start, end

    def initialize(self, start, end):
        if self.size is not None:
            out, ss, es = self.pad(self.size)
            if self.disjoint:
                start, end = start[ss::self.stride], end[es::self.stride]
                start, end = start[:out], end[-out:]
            else:
                start, end = self.group(start, ss), self.group(end, es, True)
                if tf.size(end) > out:
                    over = tf.size(end) - out + 1
                    end = tf.concat(([self.reduce(end[:over])], end[over:]), 0)
                    if tf.size(start) > out:
                        edge = self.reduce(start[out - 1:])
                        start = tf.concat((start[:out - 1], [edge]), 0)

            self.middle = out - tf.size(start) - tf.size(end)
            if self.disjoint:
                self.middle, start, end = self.consolidate(
                    self.middle, start, end, 0, 1)

        self.start, self.end = self.expand(start), self.expand(end)
        if tf.size(start) > 0:
            self.start = self.register(self.start, name="start")
        if tf.size(end) > 0:
            self.end = self.register(self.end, name="end")

    def __call__(self, size=None):
        if self.size is None:
            assert size is not None
            if self.disjoint:
                start, end = self.start, self.end
            else:
                start = self.conv(self.start, True)
                end = self.conv(self.end, False)

            out, ss, es = self.pad(size)
            idx = nslice(self.rank, self.primary)
            start = start[idx(ss, None, self.stride)]
            end = end[idx(es, None, self.stride)]
            start, end = start[idx(out)], end[idx(-out, None)]
            return self.consolidate(out - tf.shape(start)[self.primary] -
                                    tf.shape(end)[self.primary], start, end)
        elif self.disjoint:
            return self.middle, self.start, self.end
        return self.consolidate(self.middle, self.conv(self.start, True),
                                self.conv(self.end, False))

class Reweight(Dimension):
    def initialize(self):
        res = tf.range((self.kernel + 1) // 2, self.kernel, dtype=tf.float32)
        res = tf.cast(self.kernel, tf.float32) / res if self.disjoint \
            else (res + 1) / res
        super().initialize(res[(self.kernel + 1) % 2:], res[::-1])

    def conv(self, x, reverse):
        if tf.size(x) == 0:
            return x

        return tf.math.cumprod(x, self.primary, reverse=reverse)

    @classmethod
    def default(cls, *args, **kw):
        return tf.ones(*args, **kw)

    @classmethod
    def compose(cls, a, b):
        return a * b

    @classmethod
    def overlap(cls, a, b):
        return a * b / (a + b - a * b)

    @classmethod
    def reduce(cls, *args, **kwargs):
        return tf.math.reduce_prod(*args, **kwargs)

class Offset(Dimension):
    def initialize(self):
        start = tf.zeros(((self.kernel - 1) // 2,))
        end = tf.zeros((self.kernel // 2,))
        super().initialize(start, end)

    def conv(self, x, reverse):
        if tf.size(x) == 0:
            return x

        return tf.math.cumsum(x, self.primary, reverse=reverse)

    @classmethod
    def default(cls, *args, **kw):
        return tf.zeros(*args, **kw)

    @classmethod
    def compose(cls, a, b):
        return a + b

    @classmethod
    def overlap(cls, a, b):
        return a + b

    @classmethod
    def reduce(cls, *args, **kwargs):
        return tf.math.reduce_sum(*args, **kwargs)

class Border(tf.Module):
    def __init__(self, rank, kernel, stride, size=None, empty=(), channels=1,
                 channel_axis=None, disjoint=False, register=None, name=None):
        super().__init__(name=name)
        self.rank = rank
        size = (None,) * rank if size is None else size
        empty = tuple(rank + i if i < 0 else i for i in empty)
        channel_axis = rank + channel_axis if channel_axis is not None \
            and channel_axis < 0 else channel_axis
        self.channels, self.channel_axis = channels, channel_axis
        ax = [i for i in range(rank) if i not in empty and i != channel_axis]
        with self.name_scope:
            self.ax = tuple(self.base(
                rank, dim, kernel[i], stride[i], size[dim], channels,
                channel_axis, disjoint, register) for i, dim in enumerate(ax))

    def __call__(self, size=None):
        ax = [ax(None if size is None else size[ax.primary]) for ax in self.ax]
        def build(idx=0, sides=(), expand=()):
            if idx == len(self.ax):
                if not sides:
                    shape = [1] * self.rank
                    if self.channel_axis is not None:
                        shape[self.channel_axis] = self.channels
                    for i, val in expand:
                        shape[i] = val
                    return self.base.default(shape)

                res = sides[0]
                for side in sides[1:]:
                    res = self.base.compose(res, side)
                for axis, repeats in expand:
                    res = tf.repeat(res, repeats, axis)
                return res

            middle, start, end = ax[idx]
            if middle == 0:
                return build(idx + 1, sides + (end,), expand)
            else:
                dim = self.ax[idx].primary
                res = build(idx + 1, sides, expand + ((dim, middle),))
                res = res if tf.size(start) == 0 else tf.concat(
                    (build(idx + 1, sides + (start,), expand), res), dim)
                return res if tf.size(end) == 0 else tf.concat(
                    (res, build(idx + 1, sides + (end,), expand)), dim)
        return build()

class BorderReweight(Border):
    base = Reweight

class BorderOffset(Border):
    base = Offset

class BorderConv:
    def __init__(self, *args, activation=None, disjoint=False, **kw):
        super().__init__(*args, **kw)
        assert self.padding == "same" and all(
            i == 1 for i in self.dilation_rate)
        self.disjoint = disjoint
        self.small = bool(tf.reduce_all(self.kernel_size == tf.constant(1)))
        self._activation = tf.keras.activations.get(activation)
        if self._activation is None:
            self._activation = lambda x: x

    def build(self, input_shape):
        super().build(input_shape)
        if not self.small:
            channel_axis, zeroed = self._get_channel_axis(), (lambda _: 0.)
            self.border_weight = BorderReweight(
                self.rank + 2, self.kernel_size, self.strides, input_shape,
                (0,), self.filters, channel_axis, self.disjoint)
            self.border_bias = zeroed if not self.use_bias else BorderOffset(
                self.rank + 2, self.kernel_size, self.strides, input_shape,
                (0,), self.filters, channel_axis, self.disjoint)

    def call(self, inputs):
        res = super().call(inputs)
        if not self.small:
            shape = tf.shape(inputs)
            res = self.border_weight(shape) * res + self.border_bias(shape)
        return self._activation(res)

class Conv1D(BorderConv, tf.keras.layers.Conv1D):
    pass

class Conv2D(BorderConv, SpecializedConv2D):
    pass

class Conv3D(BorderConv, tf.keras.layers.Conv3D):
    pass

class DepthwiseConv2D(BorderConv, tf.keras.layers.DepthwiseConv2D):
    pass
