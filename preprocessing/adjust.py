# Copyright (C) 2020 by Kent Slaney <kent@slaney.org>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

import tensorflow as tf
from .base import Augmentation, normalize, Convert01, Group, Reformat
from functools import partial
from border.border import BorderReweight

class Blended(Augmentation):
    def caller(self, cls, im, *args, **kwargs):
        res = cls.call(self, im, *args, **kwargs)
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel = tf.ones((self.size, self.size, 1, 1))
        kernel = tf.tensor_scatter_nd_update(
            kernel, [[self.size // 2, self.size // 2, 0, 0]], [self.center])
        kernel /= tf.reduce_sum(kernel)
        self.kernel = tf.tile(kernel, [1, 1, 3, 1])
        self.reweight = BorderReweight(
            self.size, register=lambda x, *a, **k: tf.Variable(x, False))

    @normalize((0.2, 1, 1.8))
    def call(self, im, m):
        res = tf.nn.depthwise_conv2d(
            im[tf.newaxis, ...], self.kernel, [1, 1, 1, 1], "SAME")
        res *= self.reweight(tf.shape(im)[:-1])[tf.newaxis, ..., tf.newaxis]
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
    replace = tf.constant(Reformat.mean)[tf.newaxis, tf.newaxis, :]

    @normalize((0, 0.6))
    def call(self, im, value):
        shape = tf.shape(im)[:-1]
        shapef = tf.cast(shape, tf.float32)
        value = tf.cast(shapef * value // 2, tf.int32)
        center = tf.cast(tf.random.uniform((2,)) * shapef, tf.int32)
        bounds = tf.math.maximum([center - value, center + value], 0)
        bounds = tf.math.minimum(bounds, [shape - 1])
        padding = tf.transpose([[1], [-1]] * (bounds - [[0, 0], shape]))
        mask = tf.pad(tf.ones(bounds[1] - bounds[0]), padding)[..., tf.newaxis]
        return tf.where(mask > 0, __class__.replace, im)

class Flip(Augmentation):
    required = True

    @normalize((-1, 0, 1))
    def call(self, im, m):
        return im if m > 0 else tf.image.flip_left_right(im)

class Adjust(Group):
    ops = (Flip, Equalize, Posterize, Convert01, AutoContrast, Invert,
           Solarize, SolarizeAdd, Color, Contrast, Brightness, Sharpness,
           Cutout)
