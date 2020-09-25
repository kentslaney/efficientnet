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
from functools import partial
from .mbconv import MBConv
from math import ceil

class Block:
    def __init__(self, size=3, outputs=32, expand=6, strides=1, repeats=1):
        self.size, self.outputs, self.expand, self.strides, self.repeats = \
            size, outputs, expand, strides, repeats

    def keys(self):
        return ("size", "outputs", "expand", "strides")

    def __getitem__(self, key):
        return getattr(self, key)

class Embedding(tf.keras.Model):
    base = MBConv
    initialization = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_out', distribution='untruncated_normal')

    blocks = [
        Block(3, 16,  1, 1, 1),
        Block(3, 24,  6, 2, 2),
        Block(5, 40,  6, 2, 2),
        Block(3, 80,  6, 2, 3),
        Block(5, 112, 6, 1, 3),
        Block(5, 192, 6, 2, 4),
        Block(3, 320, 6, 1, 1),
    ]

    def __init__(self, width, depth, dropout=0.2, divisor=8, stem=32,
                 data_format='channels_first', pretranspose=False, **kwargs):
        super().__init__(**kwargs)
        self.width, self.depth, self.divisor, self.dropout, self.data_format, \
            self.pretranspose = width, depth, divisor, dropout, data_format, \
                pretranspose
        self.stem = self.round_filters(stem)
        self.total = sum(block.repeats for block in self.blocks)

        self.conv = partial(
            self.base.conv, padding='same', data_format=data_format,
            kernel_initializer=self.initialization)
        channel = -1 if data_format == 'channels_last' else 1
        self.bn = partial(self.base.bn, axis=channel)

        for block in self.blocks:
            block.outputs = self.round_filters(block.outputs)
            block.repeats = self.round_repeats(block.repeats)
        __class__._build(self)

    def kwargs(self, overall, repeat):
        return {
            **({"strides": 1} if repeat else {}),
            "dropout": self.dropout * overall / self.total,
            "data_format": self.data_format,
        }

    def round_filters(self, filters):
        filters *= self.width
        rounded = int(filters + self.divisor / 2)
        rounded = rounded // self.divisor * self.divisor
        rounded = max(self.divisor, rounded)

        if rounded < 0.9 * filters:
            rounded += depth_divisor
        return int(rounded)

    def round_repeats(self, repeats):
        return int(ceil(self.depth * repeats))

    def _build(self):
        self._stem_conv = self.conv(self.stem, 3, 2, use_bias=False)
        self._stem_bn = self.bn()

        self._blocks, i = [], 0
        for block in self.blocks:
            for j in range(block.repeats):
                self._blocks.append(self.base(**{
                    **block, **self.kwargs(i, j)}))
                i += 1

    def call(self, inputs, training):
        x = inputs
        if self.pretranspose:
            x = tf.transpose(x, [3, 0, 1, 2])

        x = self._stem_conv(x)
        x = tf.nn.swish(self._stem_bn(x, training))

        for block in self._blocks:
            x = block(x, training)
        return x

class Classifier(Embedding):
    dense_init = tf.keras.initializers.VarianceScaling(
        scale=1 / 3, mode='fan_out', distribution='uniform')
    drop = tf.keras.layers.Dropout

    def __init__(self, *args, head_drop=0.2, outputs=1000, head=1280, **kw):
        super().__init__(*args, **kw)
        self.outputs, self.head_drop = outputs, head_drop
        self.head = self.round_filters(head)
        self._head_drop = self.drop(head_drop)
        self.pool = tf.keras.layers.GlobalAveragePooling2D(
            data_format=self.data_format)
        __class__._build(self)

    def _build(self):
        self._head_conv = self.conv(self.head, 1, use_bias=False)
        self._head_bn = self.bn()
        self._fc = tf.keras.layers.Dense(
            self.outputs, "softmax", kernel_initializer=self.dense_init)

    def call(self, inputs, training):
        x = super().call(inputs, training)
        x = tf.nn.swish(self._head_bn(self._head_conv(x), training))
        return self._fc(self._head_drop(self.pool(x), training))
