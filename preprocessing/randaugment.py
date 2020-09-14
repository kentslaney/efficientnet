import tensorflow as tf
from tensorflow_addons.image import transform
from functools import partial
from collections import defaultdict
from border.border import BorderReweight
from math import pi

class Augmentation:
    # Collects the augment function from all inherited classes into ops list.
    # Not the most readable solution, but it lets ops interact with the base
    # instance, which is nice for interdependencies, eg composing transforms.
    def __new__(cls, *args, **kwargs):
        res, ops = super().__new__(cls), defaultdict(list)
        ops[cls].append((cls,))
        for i in cls.__mro__:
            parents = [j for j in i.__bases__ if __class__ in j.__mro__]
            base = [ops[i]] if len(parents) == 1 else [
                [(j, n) + ops[i][0][1:]] for n, j in enumerate(parents)]
            for p, b in zip(parents, base):
                ops[p] += b
        res.aug = tuple(i[0] for i in sorted(
            ops[__class__], key=lambda x: x[::-1]))
        res.req = tuple(int(i.required) for i in res.aug)
        res.inp = tuple(partial(i.inputs, res, i) for i in res.aug)
        res.ops = tuple((
            lambda i, j: lambda im, m: i.augment(res, im, *j(m)))(*i)
                        for i in zip(res.aug, res.inp))
        return res

    level = ()
    flips = True
    required = False
    integers = False
    offset = 0
    def inputs(self, cls, m):
        if cls.integers:
            res = tuple(m * (j - i + 0.8) + i - 0.4 for i, j in cls.level)
            res = tuple(tf.cast(tf.math.round(i), tf.int32) for i in res)
        else:
            res = tuple(m * (j - i) + i for i, j in cls.level)

        if cls.flips:
            res = tuple(cls.offset + i * (
                1 if tf.random.uniform(()) < 0.5 else -1) for i in res)
        return res

class Blended(Augmentation):
    def inputs(self, cls, m):
        return Augmentation.inputs(self, cls, m) + (partial(cls.alter, self),)

    def augment(self, im1, value, f):
        im0 = f(im1)
        if tf.rank(value) == 0 and value == int(value):
            return (im0, im1)[int(value)]
        return tf.clip_by_value(im0 + value * (im1 - im0), 0., 1.)

class Color(Blended):
    offset, level = 1, ((0, 0.8),)

    def alter(self, im):
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(im))

class Brightness(Blended):
    offset, level = 1, ((0, 0.8),)

    def alter(self, im):
        return tf.zeros_like(im)

class Sharpness(Blended):
    offset, level, size, center = 1, ((0, 0.9),), 3, 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(__class__, "kernel"):
            size, center = __class__.size, __class__.center

            kernel = tf.ones((size, size, 1, 1))
            kernel = tf.tensor_scatter_nd_update(
                kernel, [[size // 2, size // 2, 0, 0]], [center])
            kernel /= tf.reduce_sum(kernel)
            __class__.kernel = tf.tile(kernel, [1, 1, 3, 1])

            __class__.reweight = BorderReweight(
                size, register=lambda x, *a, **k: x)

    def alter(self, im):
        res = tf.nn.depthwise_conv2d(
            im[tf.newaxis, ...], __class__.kernel, [1, 1, 1, 1], "SAME")
        res *= __class__.reweight(tf.shape(im)[:-1])[
            tf.newaxis, ..., tf.newaxis]
        return res[0]

class Contrast(Augmentation):
    offset, level = 1, ((0, 0.9),)

    def augment(self, im, value):
        return tf.image.adjust_contrast(im, value)

class Solarize(Augmentation):
    level, flips = ((0, 0.9),), False

    def augment(self, im, value):
        return tf.where(im < value, im, 1 - im)

class SolarizeAdd(Augmentation):
    level, flips, threshold = ((0, 0.9),), False, 0.5

    def augment(self, im, value):
        return tf.where(im < __class__.threshold,
                        tf.clip_by_value(im + value, 0, 1), im)

class Invert(Augmentation):
    def augment(self, im):
        return 1 - im

class Convert01(Augmentation):
    required = True

    def augment(self, im):
        return im / 255

class Convert11(Augmentation):
    required = True

    def augment(self, im):
        return im * 2 - 1

class Posterize(Augmentation):
    level, flips, integers = ((0, 4),), False, True

    def augment(self, im, value):
        shift = tf.cast(8 - value, tf.uint8)
        res = tf.bitwise.right_shift(im, shift)
        return tf.bitwise.left_shift(res, shift)

class AutoContrast(Augmentation):
    def augment(self, im):
        lo = tf.reduce_min(im, (0, 1), True)
        hi = tf.reduce_max(im, (0, 1), True)
        scale = tf.math.maximum(tf.math.divide_no_nan(1., hi - lo), 1.)
        offset = tf.where(lo == hi, 0, lo)
        return (im - offset) * scale

class Equalize(Augmentation):
    def augment(self, im):
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
    level, flips = ((0, 0.6),), False
    replace = tf.constant([[[125, 123, 114]]], tf.float32) / 255

    def augment(self, im, value):
        shape = tf.shape(im)[:-1]
        shapef = tf.cast(shape, tf.float32)
        value = tf.cast(shapef * value // 2, tf.int32)
        center = tf.cast(tf.random.uniform((2,)) * shapef, tf.int32)
        bounds = tf.math.maximum([center - value, center + value], 0)
        bounds = tf.math.minimum(bounds, [shape - 1])
        padding = tf.transpose([[1], [-1]] * (bounds - [[0, 0], shape]))
        mask = tf.pad(tf.ones(bounds[1] - bounds[0]), padding)[..., tf.newaxis]
        return tf.where(mask > 0, __class__.replace, im)

class Transformation(Augmentation):
    flips = True

    def inputs(self, cls, m):
        return (partial(cls.transform, self),
                ) + Augmentation.inputs(self, cls, m)

    def augment(self, im, f, *args):
        self._transform = f(im, *args) @ self._transform
        return im

class ApplyTransform(Blended):
    required, _output, _transform = True, None, tf.eye(3)
    replace = tf.constant([[[125, 123, 114]]], tf.float32) / 255

    def inputs(self, cls, m):
        return ()

    def augment(self, im):
        if self._transform is __class__._transform and \
           self._output is __class__._output:
            return im
        im = tf.concat((im, tf.ones(tf.shape(im)[:-1])[..., tf.newaxis]), -1)
        flat = tf.reshape(self._transform, -1)
        im = transform(im, flat[:-1] / flat[-1], "BILINEAR", self._output)
        self._output, self._transform = __class__._output, __class__._transform
        return Blended.augment(
            self, im[..., :-1], im[..., -1:], lambda _: __class__.replace)

class TranslateX(Transformation):
    level = ((0, 0.4),)

    def transform(self, im, value):
        value *= tf.cast(tf.shape(im)[0], tf.float32)
        return [[1, 0, value], [0, 1, 0], [0, 0, 1]]

class TranslateY(Transformation):
    level = ((0, 0.4),)

    def transform(self, im, value):
        value *= tf.cast(tf.shape(im)[1], tf.float32)
        return [[1, 0, 0], [0, 1, value], [0, 0, 1]]

class Translate(TranslateX, TranslateY):
    pass

class ShearX(Transformation):
    level = ((0, 0.3),)

    def transform(self, im, value):
        return [[1, value, 0], [0, 1, 0], [0, 0, 1]]

class ShearY(Transformation):
    level = ((0, 0.3),)

    def transform(self, im, value):
        return [[1, 0, 0], [value, 1, 0], [0, 0, 1]]

class Shear(ShearX, ShearY):
    pass

class Rotate(Transformation):
    level = ((0, 30 * pi / 180),)

    def transform(self, im, v):
        return [[tf.cos(v), -tf.sin(v), 0],
                [tf.sin(v),  tf.cos(v), 0],
                [0, 0, 1]]

class RotPad(Rotate):
    def transform(self, im, v):
        res = tf.convert_to_tensor(Rotate.transform(self, im, v))
        bounds = tf.unstack([[0, 0], tf.shape(im)[:-1]], axis=-1)
        bounds = tf.reshape(tf.stack(tf.meshgrid(*bounds), -1), (4, 2))
        bounds = tf.concat((tf.cast(bounds, tf.float32), tf.ones((4, 1))), -1)
        bounds = res @ tf.transpose(bounds)
        bounds = tf.stack([tf.reduce_min(bounds, 1), tf.reduce_max(bounds, 1)])
        self._output = tf.cast(bounds[1] - bounds[0], tf.int32)[:-1]
        return res @ [[1, 0, bounds[0][1]], [0, 1, bounds[0][0]], [0, 0, 1]]

class Flip(Augmentation):
    level, integers, required = ((0, 1),), True, True

    def augment(self, im, value):
        return im if value > 0 else tf.image.flip_left_right(im)

class Randomize:
    def __init__(self, n=3, m=.4):
        super().__init__()
        self.n, self.m = len(self.ops) - sum(self.req) if n < 0 else n, m
        self.k = len(self.ops) - sum(self.req)
        self.r = tf.range(self.k) + tf.cumsum(self.req)[
            tf.math.logical_not(tf.cast(self.req, tf.bool))]

    def sample(self, n, m): # chooses n out of the first m natural numbers
        return tf.random.uniform_candidate_sampler(
            tf.range(m, dtype=tf.int64)[tf.newaxis, :], m, n, True, m
        ).sampled_candidates

    def __call__(self, im):
        chosen = self.sample(self.n, self.k)
        chosen = tf.gather(self.r, chosen)
        updates = tf.repeat(True, tf.shape(chosen)[0])
        mask = tf.scatter_nd(chosen[:, tf.newaxis], updates, (len(self.ops),))
        for i in range(len(self.ops)):
            if self.req[i] or mask[i]:
                im = self.ops[i](im, self.m)
        return im

class Adjust(Flip, Equalize, Posterize, Convert01, AutoContrast, Invert,
             Solarize, SolarizeAdd, Color, Contrast, Brightness, Sharpness,
             Cutout):
    pass

class RandAugment(Randomize, Adjust, RotPad, Translate, Shear, ApplyTransform):
    pass
