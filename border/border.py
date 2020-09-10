import tensorflow as tf

expand = lambda r: lambda x: (x,) * r if tf.rank(x) == 0 else x

class Border:
    default = None

    def __init__(self, width, stride=1, rank=2, register=None):
        assert self.default is not None and tf.rank(self.default) == 0
        self.rank, self.register = rank, register or tf.Variable
        width, stride = map(expand(rank), (width, stride))
        self.stride = tf.convert_to_tensor(stride)
        self.values = [[self.initialize(size, i, axis, bool(end))
                        for end, size in enumerate(((i - 1) // 2, i // 2))]
                       for axis, i in enumerate(width)]

    # @tf.function(input_signature=[tf.TensorSpec([None], tf.int32)])
    def __call__(self, shape):
        tf.debugging.assert_equal(tf.shape(shape), (self.rank,))
        return self._build(shape, [])

    def _build(self, shape, built):
        axis = len(built)
        if axis == self.rank:
            built = tf.meshgrid(*built, indexing="ij")
            res = built[0]
            for i in built[1:]:
                res = self.compose(res, i)
            return res

        size, stride = shape[axis], self.stride[axis]
        start, end = self.values[axis]
        ssize, esize = map(tf.size, (start, end))
        start, end = self.conv(start, False), self.conv(end, True)
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
            self._build(shape, built + [start]),
            self._build(shape, built + [middle]),
            self._build(shape, built + [end]),
        ), len(built))

    # @tf.function(input_signature=[
    #     tf.TensorSpec([None]), tf.TensorSpec([], tf.bool)])
    def conv(self, inp, end):
        out = inp
        if tf.size(inp) > 1:
            for i in tf.range(1, tf.size(inp)):
                j = ((inp, i), (out, 0))
                (left, offset), (right, lim) = j if end else j[::-1]
                update = self.compose(right[i:], left[:-i])
                ind = tf.range(offset, tf.size(inp) - lim)[:, tf.newaxis]
                out = tf.tensor_scatter_nd_update(out, ind, update)
        return out

    def initialize(self, size, width, axis, end):
        raise NotImplementedError()

    @classmethod
    def compose(_, a, b):
        raise NotImplementedError()

    @classmethod
    def overlap(_, a, b):
        raise NotImplementedError()

class BorderReweight(Border):
    default = tf.constant(1.)

    def initialize(self, size, width, axis, end):
        name = f"border_reweight_axis{axis}_{'end' if end else 'start'}"
        res = tf.range(width - size, width)[::-1 if end else 1]
        res = tf.cast((res + 1) / res, tf.float32)
        return self.register(res, name=name)

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
    default = tf.constant(0.)

    def initialize(self, size, width, axis, end):
        name = f"border_bias_axis{axis}_{'end' if end else 'start'}"
        return self.register(tf.zeros((size,)), name=name)

    @classmethod
    def compose(_, a, b):
        return a + b

    @classmethod
    def overlap(_, a, b):
        return a + b
