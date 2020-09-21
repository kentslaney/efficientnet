import tensorflow as tf
from tensorflow_addons.image import transform
import tensorflow_probability as tfp
from functools import partial
from math import radians
from .base import Augmentation, normalize
from .adjust import Blended

class Transformation(Augmentation):
    def caller(self, cls, im, *args, **kwargs):
        self._transform @= cls.call(self, im, *args, **kwargs)
        return im

class ApplyTransform(Blended):
    required, _output, _transform = True, tf.zeros((2,), tf.int32), tf.eye(3)
    replace = tf.constant([[[125, 123, 114]]], tf.float32) / 255
    track = ("_output", "_transform")

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
        im = transform(im, flat[:-1] / flat[-1], "BILINEAR", self.output(im))

        self._output, self._transform = __class__._output, __class__._transform
        return im[..., -1:], self.replace, im[..., :-1]

class TranslateX(Transformation):
    @normalize((-0.4, 0, 0.4))
    def call(self, im, m):
        value *= tf.cast(tf.shape(im)[1], tf.float32)
        return [[1, 0, m], [0, 1, 0], [0, 0, 1]]

class TranslateY(Transformation):
    @normalize((-0.4, 0, 0.4))
    def call(self, im, m):
        value *= tf.cast(tf.shape(im)[0], tf.float32)
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
        res = tf.convert_to_tensor(Rotate.transform(self, im, m))
        bounds = res @ self.bounds(tf.shape(im)[:-1])
        bounds = tf.stack([tf.reduce_min(bounds, 1), tf.reduce_max(bounds, 1)])
        self._output = tf.cast(bounds[1] - bounds[0], tf.int32)[:-1]
        return res @ [[1, 0, bounds[0][1]], [0, 1, bounds[0][0]], [0, 0, 1]]

class Reshape(ApplyTransform):
    required = True

    def __init__(self, shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = tf.convert_to_tensor(shape)

class Stretch(Reshape):
    def call(self, im):
        ratio = tf.cast(self.output(im) / self.shape, tf.float32)
        self._output = self.shape
        self._transform @= tf.linalg.diag(tf.concat((ratio[::-1], (1.,)), 0))
        return ApplyTransform.call(self, im)

class Crop(Reshape):
    def __init__(self, *args, a=9., b=1., distort=True, recrop=True, **kwargs):
        super().__init__(*args, **kwargs)
        dist = tfp.distributions.Beta(a, b)
        iid = lambda: dist.sample((2,))
        same = lambda: tf.repeat(dist.sample(), 2)
        ones = lambda: tf.constant([1., 1.])
        self.crop_sample = ones if not recrop else iid if distort else same
        self.distort = distort

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
        return ApplyTransform.call(self, im)

class CenterCrop(Reshape, Augmentation):
    def call(self, im):
        return tf.keras.preprocessing.image.smart_resize(im, self.shape)

class PaddedTransforms(PaddedRotate, Translate, Shear, Stretch):
    pass

class CroppedTransforms(Rotate, Shear, Crop):
    pass
