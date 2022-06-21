import tensorflow as tf
from functools import partial, wraps, update_wrapper
from inspect import signature, Parameter
from tensorflow_addons.image import transform
import tensorflow_probability as tfp
from math import radians, pi

# adds a `cond` attribute to a callable, which provides a static graph
# conditional that will execute the provided method or not based on a tf.Tensor
class CondCall:
    def __init__(self, parent, f, bypass=False):
        assert type(bypass) == bool
        self.f, self.parent, self.bypass = f, parent, bypass
        update_wrapper(self, f)

    def __call__(self, *args, **kw):
        return self.f(*args, **kw)

    def cond(self, tensor, im, *args, **kw):
        if self.bypass:
            return self(im, *args, **kw)

        identity = self.parent.variables
        res, self.parent.variables = tf.cond(
            tensor, lambda: (self(im, *args, **kw), self.parent.variables),
            lambda: (im, identity))
        return res

# list wrapper that will always execute classes where `required == True` and
# can sample a fixed number of the optional augmentations
class OpsList:
    def __init__(self, parent, ops):
        self.parent, self.ops = parent, ops
        self.required = [i.required for i in ops]
        self.choosable = len(self) - sum(self.required)
        self.offset = tf.range(self.choosable) + tf.cumsum(tf.cast(
            self.required, tf.int32))[tf.math.logical_not(self.required)]

    # calls the augmentation's caller method and pulls the method signature
    # from call
    def __getitem__(self, i):
        wrapped = partial(self.ops[i].caller, self.parent)
        wrapped = wraps(self.ops[i].call)(wrapped)
        wrapped = partial(wrapped, self.ops[i])
        return CondCall(self.parent, wrapped, self.required[i])

    # chooses n out of the first m natural numbers
    def _sample(self, n, m):
        return tf.random.uniform_candidate_sampler(
            tf.range(m, dtype=tf.int64)[tf.newaxis, :], m, n, True, m
        ).sampled_candidates

    def sample(self, n):
        assert 0 <= n <= self.choosable
        chosen = tf.gather(self.offset, self._sample(n, self.choosable))
        updates = tf.repeat(True, tf.shape(chosen))
        mask = tf.scatter_nd(chosen[:, tf.newaxis], updates, (len(self),))
        return (wraps(op)(partial(op.cond, mask[i]))
                for i, op in enumerate(self))

    def __len__(self):
        return len(self.ops)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

# remaps an argument from [-1...0...1] to [lo...mid...hi] or [hi, mid, hi]
class Normalized:
    def __new__(cls, *args, **kw):
        self = super().__new__(cls)
        def decorator(f):
            self.__init__(f)
            # create mapping from initialized class arguments to decorated
            # method arguments
            self.normal = self.sig.bind_partial(*args, **kw).arguments
            for k, v in self.normal.items():
                kind = self.sig.parameters[k]
                if kind == Parameter.VAR_POSITIONAL:
                    self.normal[k] = tuple(self.parse(v) for v in v)
                if kind == Parameter.VAR_KEYWORD:
                    self.normal[k] = {k: self.parse(v) for k, v in v}
                else:
                    self.normal[k] = self.parse(v)
            return wraps(f)(self)
        return decorator

    def __init__(self, f):
        self.f, self.sig = f, signature(f)

    # remaps method defaults too
    def __call__(self, *args, **kw):
        bound = self.sig.bind(*args, **kw)
        bound.apply_defaults()
        bargs = bound.arguments
        for k, v in bargs.items():
            if k in self.normal and self.normal[k]:
                res = self.map(*self.normal[k], v)
                bargs[k] = res
        return self.f(*bound.args, **bound.kwargs)

    # translate length 0, 2, 3, or 4 tuple to fixed remapping specs
    # optional first argument: `int` or `float` depending on desired casting
    # other arguments specify [lo=hi (default)], mid, hi
    @classmethod
    def parse(cls, args):
        if len(args) == 0:
            return None

        if type(args[0]) is type:
            assert args[0] in (int, float)
            res, args = args[0] is int, args[1:]
        else:
            res = False

        assert 2 <= len(args) <= 3
        return (res,) + tuple((args[1],) + args if len(args) == 2 else args)

    # remaps an input magnitude to the output range
    @classmethod
    def map(cls, floor, lo, center, hi, v):
        lo, center, hi = (tf.constant(i, tf.float32) for i in (lo, center, hi))
        v, lo, hi = tf.math.abs(v), center, lo if v < 0 else hi
        if floor:
            v, lo, hi = tf.cond(
                lo < hi, lambda: (v, lo, hi), lambda: (1 - v, hi, lo))
            mapped = v * (hi - lo + 1) + lo - 0.5
            mapped = tf.cond(mapped == hi + 0.5, lambda: hi, lambda: tf.cond(
                mapped == lo - 0.5, lambda: lo, lambda: tf.math.round(mapped)))
            return tf.cast(mapped, tf.int32)
        else:
            return tf.cast(v * (hi - lo) + lo, tf.float32)

# skip the first two arguments (`self` and `im` for augmentation)
def normalize(*args, **kw):
    return Normalized((), (), *args, **kw)

class Augmentation:
    required, ops, track = False, (), ()

    def __new__(cls, *args, **kw):
        res = super().__new__(cls)
        augmenting_parents = {i: sum(
            issubclass(j, __class__) for j in i.__bases__)
            for i in cls.__mro__}
        grouped_leaves = sum((
            i.__bases__ for i, j in augmenting_parents.items() if j > 1), ())
        unique_path_leaves = set(i for i in grouped_leaves if max(
            augmenting_parents[j] for j in i.__mro__) == 1)
        ops = unique_path_leaves.union(set(
            i for i in cls.__mro__ if __class__ in i.__bases__ and
            augmenting_parents[i] > 1))
        ops = [i for i in cls.__mro__ if i in ops] if ops else (cls,)
        res.ops = OpsList(res, ops)
        res.track = tuple(set(sum((
            i.track for i in cls.__mro__ if issubclass(i, __class__)), ())))
        res.__init__(*args, **kw)
        return res

    def __init__(self):
        self.initialization = self.variables

    def caller(self, cls, *args, **kw):
        return cls.call(self, *args, **kw)

    @property
    def variables(self):
        return tuple(getattr(self, i) for i in self.track)

    @variables.setter
    def variables(self, value):
        for i, j in zip(self.track, value):
            setattr(self, i, j)

    def recall(self, mask):
        return mask

class Pipeline:
    def __call__(self, im):
        for op in self.ops:
            im = op(im)
        return im

class Convert01(Augmentation):
    required = True

    def call(self, im):
        return im / 255

class WaveletTransform(Augmentation):
    required = True

    def call(self, im):
        # 8 bit color, so the smallest wavelet will cycle once per 2 increments
        resolution = 8
        rescaled = pi * im[..., tf.newaxis]
        rescaled *= tf.reshape(tf.range(
            1, resolution + 1, dtype=tf.float32), (1, 1, 1, -1))
        return tf.concat(tf.cos(rescaled), -1)

class Reformat(Augmentation):
    required = True
    mean, norm = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def __init__(self, *args, data_format="channels_first", **kw):
        self.channels_last = data_format == "channels_last"
        super().__init__(*args, **kw)

    def call(self, im):
        im = (im - self.mean) / self.norm
        return im if self.channels_last else tf.transpose(im, [2, 0, 1])

# adjustments
class Blended(Augmentation):
    def caller(self, cls, im, *args, **kw):
        res = cls.call(self, im, *args, **kw)
        m, im0, im1 = res if len(res) == 3 else res + (im,)
        return tf.clip_by_value(im0 + m * (im1 - im0), 0., 1.)

class Color(Blended):
    @normalize((0.2, 1, 1.8))
    def call(self, im, m):
        return m, tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(im))

class Brightness(Blended):
    @normalize((0.2, 1, 1.8))
    def call(self, im, m):
        return m, tf.zeros_like(im)

class Sharpness(Blended):
    size, center = 3, 5

    def __init__(self, *args, **kw):
        kernel = tf.ones((self.size, self.size, 1, 1))
        kernel = tf.tensor_scatter_nd_update(
            kernel, [[self.size // 2, self.size // 2, 0, 0]], [self.center])
        kernel /= tf.reduce_sum(kernel)
        self.kernel = tf.tile(kernel, [1, 1, 3, 1])
        super().__init__(*args, **kw)

    @normalize((0.2, 1, 1.8))
    def call(self, im, m):
        res = tf.nn.depthwise_conv2d(
            im[tf.newaxis, ...], self.kernel, [1, 1, 1, 1], "SAME")
        return m, res[0]

class Contrast(Augmentation):
    @normalize((0.2, 1, 1.8))
    def call(self, im, m):
        return tf.image.adjust_contrast(im, m)

class Solarize(Augmentation):
    @normalize((0, 0.9))
    def call(self, im, m):
        return tf.where(im < m, im, 1 - im)

class SolarizeAdd(Augmentation):
    threshold = 0.5

    @normalize((0, 0.9))
    def call(self, im, m):
        return tf.where(
            im < self.threshold, tf.clip_by_value(im + m, 0, 1), im)

class Invert(Augmentation):
    def call(self, im):
        return 1 - im

class Posterize(Augmentation):
    @normalize((int, 0, 4))
    def call(self, im, m):
        shift = tf.cast(tf.math.round(8 - m), tf.uint8)
        res = tf.bitwise.right_shift(im, shift)
        return tf.bitwise.left_shift(res, shift)

class AutoContrast(Augmentation):
    def call(self, im):
        lo = tf.reduce_min(im, (0, 1), True)
        hi = tf.reduce_max(im, (0, 1), True)
        scale = tf.math.maximum(tf.math.divide_no_nan(1., hi - lo), 1.)
        offset = tf.where(lo == hi, 0., lo)
        return (im - offset) * scale

class Equalize(Augmentation):
    def call(self, im):
        return tf.stack([self._augment(im[..., c]) for c in range(3)], -1)

    def _augment(self, im):
        imi = tf.cast(im, tf.int32)
        hist = tf.histogram_fixed_width(imi, [0, 255], nbins=256)
        last = hist[tf.where(hist != 0)[-1, 0]]
        lut = tf.cumsum(hist)
        step = (lut[-1] - last) // 255
        if step == 0:
            return im

        lut = (lut + (step // 2)) // step
        lut = tf.clip_by_value(tf.concat([[0], lut[:-1]], 0), 0, 255)
        return tf.gather(tf.cast(lut, tf.uint8), imi)

class Cutout(Augmentation):
    track = ("cutout_center",)

    def __init__(self, *args, **kw):
        self.replace = tf.constant(Reformat.mean)[tf.newaxis, tf.newaxis, :]
        self.cutout_center = tf.zeros((2,))
        super().__init__(*args, **kw)

    def caller(self, cls, im, m):
        self.cutout_center = tf.random.uniform((2,))
        return cls.call(self, im, m)

    @normalize((0, 0.6))
    def call(self, im, m):
        shape = tf.shape(im)[:-1]
        shapef = tf.cast(shape, tf.float32)
        value = tf.cast(shapef * m // 2, tf.int32)
        center = tf.cast(self.cutout_center * shapef, tf.int32)
        bounds = tf.math.maximum([center - value, center + value], 0)
        bounds = tf.math.minimum(bounds, [shape - 1])
        padding = tf.transpose([[1], [-1]] * (bounds - [[0, 0], shape]))
        mask = tf.pad(tf.ones(bounds[1] - bounds[0]), padding)[..., tf.newaxis]
        return tf.where(mask > 0, self.replace, im)

    def recall(self, mask):
        initial, self.replace = self.replace, tf.zeros((1, 1, 1), mask.dtype)
        res = super().recall(self.call(mask, self.m))
        self.replace = initial
        return res

class Flip(Augmentation):
    required, track = True, ("flipped",)

    def __init__(self, *args, **kw):
        self.flipped = tf.cast(False, tf.bool)
        super().__init__(*args, **kw)

    def caller(self, cls, im, m):
        self.flipped = m > 0
        return cls.call(self, im, m)

    def call(self, im, m):
        return im if self.flipped else tf.image.flip_left_right(im)

    def recall(self, mask):
        return super().recall(self.call(mask, self.m))

class Adjust(
        Flip, Equalize, Posterize, Convert01, AutoContrast, Invert, Solarize,
        SolarizeAdd, Color, Contrast, Brightness, Sharpness, Cutout):
    pass

# transformations
class Transformation(Augmentation):
    def caller(self, cls, im, *args, **kw):
        self._transform @= cls.call(self, im, *args, **kw)
        return im

class ApplyTransform(Blended):
    required, track = True, ("_output", "_transform")

    def __init__(self, *args, **kw):
        self._output, self._transform = tf.zeros((2,), tf.int32), tf.eye(3)
        self.replace = tf.constant([[[125, 123, 114]]], tf.float32) / 255
        super().__init__(*args, **kw)

    def output(self, im):
        return tf.shape(im)[:-1] if tf.reduce_all(
            self._output == 0) else self._output

    def bounds(self, shape):
        bounds = tf.unstack([[0, 0], shape], axis=-1)
        bounds = tf.stack(tf.meshgrid(*bounds), -1)
        bounds = tf.cast(tf.reshape(bounds, (4, 2)), tf.float32)
        return tf.transpose(tf.concat((bounds, tf.ones((4, 1))), -1))

    def call(self, im):
        im = tf.concat((im, tf.ones(tf.shape(im)[:-1])[..., tf.newaxis]), -1)
        flat = tf.reshape(self._transform, (-1,))
        im = transform(
            im, flat[:-1] / flat[-1], "BILINEAR", output_shape=self.output(im))

        return im[..., -1:], self.replace, im[..., :-1]

    def recall(self, mask):
        flat = tf.reshape(self._transform, (-1,))
        return transform(mask, flat[:-1] / flat[-1], output_shape=self._output)

class TranslateX(Transformation):
    @normalize((-0.4, 0, 0.4))
    def call(self, im, m):
        m *= tf.cast(tf.shape(im)[1], tf.float32)
        return [[1, 0, m], [0, 1, 0], [0, 0, 1]]

class TranslateY(Transformation):
    @normalize((-0.4, 0, 0.4))
    def call(self, im, m):
        m *= tf.cast(tf.shape(im)[0], tf.float32)
        return [[1, 0, 0], [0, 1, m], [0, 0, 1]]

class Translate(TranslateX, TranslateY):
    pass

class ShearX(Transformation):
    @normalize((-0.3, 0, 0.3))
    def call(self, im, m):
        return [[1, m, 0], [0, 1, 0], [0, 0, 1]]

class ShearY(Transformation):
    @normalize((-0.3, 0, 0.3))
    def call(self, im, m):
        return [[1, 0, 0], [m, 1, 0], [0, 0, 1]]

class Shear(ShearX, ShearY):
    pass

class Rotate(Transformation):
    @normalize((radians(-30), 0, radians(30)))
    def call(self, im, m):
        return [[tf.cos(m), -tf.sin(m), 0],
                [tf.sin(m),  tf.cos(m), 0],
                [0,          0,         1]]

class PaddedRotate(Rotate):
    @normalize((radians(-30), 0, radians(30)))
    def call(self, im, m):
        res = tf.convert_to_tensor(Rotate.call(self, im, m))
        bounds = res @ self.bounds(tf.shape(im)[:-1])
        bounds = tf.stack([tf.reduce_min(bounds, 1), tf.reduce_max(bounds, 1)])
        self._output = tf.cast(bounds[1] - bounds[0], tf.int32)[:-1]
        return res @ [[1, 0, bounds[0][1]], [0, 1, bounds[0][0]], [0, 0, 1]]

class Reshape(ApplyTransform):
    required = True

    def __init__(self, shape, *args, **kw):
        self.shape, self._shape = tf.convert_to_tensor(shape), tuple(shape)
        super().__init__(*args, **kw)

    def caller(self, cls, im, *args, **kw):
        im = super().caller(cls, im, *args, **kw)
        im.set_shape(self._shape + (None,))
        return im

class Stretch(Reshape):
    def call(self, im):
        ratio = tf.cast(self.output(im) / self.shape, tf.float32)
        self._output = self.shape
        self._transform @= tf.linalg.diag(tf.concat((ratio[::-1], (1.,)), 0))
        return super().call(im)

class Crop(Reshape):
    def __init__(self, *args, a=9., b=1., distort=True, recrop=True, **kw):
        dist = tfp.distributions.Beta(a, b)
        iid = lambda: dist.sample((2,))
        same = lambda: tf.repeat(dist.sample(), 2)
        ones = lambda: tf.constant([1., 1.])
        self.crop_sample = ones if not recrop else iid if distort else same
        self.distort = distort
        super().__init__(*args, **kw)

    def call(self, im):
        self._output, bounds = self.shape, self.bounds(self.shape[::-1])

        valid = tf.cast(tf.shape(im)[:-1][::-1], tf.float32)
        crop = (self._transform @ bounds)[:-1] / valid[:, tf.newaxis]
        extrema = tf.stack([tf.reduce_min(crop, 1), tf.reduce_max(crop, 1)])
        limit = 1 / (extrema[1] - extrema[0])
        scale = self.crop_sample() * (
            limit if self.distort else tf.reduce_min(limit))
        offset = -valid * scale * extrema[0] + \
            valid * tf.random.uniform((2,)) * (limit - scale) / limit
        self._transform = [[scale[0], 0,        offset[0]],
                           [0,        scale[1], offset[1]],
                           [0,        0,        1        ]] @ self._transform
        return super().call(im)

class CenterCrop(Reshape):
    def call(self, im):
        return tf.keras.preprocessing.image.smart_resize(im, self.shape)

class PaddedTransforms(PaddedRotate, Translate, Shear, Stretch):
    pass

class CroppedTransforms(Rotate, Shear, Crop):
    pass

# endpoints
class Randomize:
    def __init__(self, *args, n=3, m=.4, **kw):
        self.n, self.m = self.ops.choosable if n == -1 else n, m
        super().__init__(*args, **kw)

    def __call__(self, im, mask=None):
        self.variables = self.initialization
        for i, op in enumerate(self.ops.sample(self.n)):
            inputs = len(signature(op).parameters) - 1
            m = self.m * tf.math.sign(tf.random.uniform((inputs,)) - 0.5)
            im = op(im, *tf.unstack(m))
        if mask is None:
            return im
        return im, self.recall(mask)

class RandAugmentPadded(Randomize, Adjust, PaddedTransforms, Reformat):
    pass

class RandAugmentCropped(Randomize, Adjust, CroppedTransforms, Reformat):
    pass

class PrepStretched(Pipeline, Convert01, Stretch, Reformat):
    pass

class PrepCropped(Pipeline, Convert01, CenterCrop, Reformat):
    pass
