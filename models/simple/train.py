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
from tensorflow_addons.optimizers import MovingAverage
from models.train import RandAugmentTrainer, TFDSTrainer
from cli.utils import RequiredLength
from functools import partial
from models.border.layers import Conv2D as BorderConv2D
from models.utils import Conv2D

tf.config.optimizer.set_jit(True)

class Model(tf.keras.Model):
    conv = Conv2D

    def __init__(self, outputs, data_format):
        super().__init__()
        self.conv = partial(self.conv, padding='same', data_format=data_format,
                            activation=tf.nn.relu)
        self.conv0 = self.conv(128, 4, 2)
        self.conv1 = self.conv(192, 4, 2)
        self.conv2 = self.conv(256, 4, 2)
        self.pool = tf.keras.layers.GlobalAveragePooling2D(data_format)
        self.dense = tf.keras.layers.Dense(outputs, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.dense(self.pool(x))

class Trainer(RandAugmentTrainer, TFDSTrainer):
    opt = lambda _, lr: MovingAverage(
        tf.keras.optimizers.RMSprop(lr, 0.9, 0.9, 0.001))

    def build(self, border_conv, **kwargs):
        super().build(**kwargs)
        self.mapper = lambda f: lambda x, y: (
            f(x), tf.one_hot(y, self.outputs))
        if border_conv:
            Model.conv = BorderConv2D
        self.model = Model(self.outputs, self.data_format)
        self.compile(tf.keras.losses.CategoricalCrossentropy(True, 0.1),
                     ["categorical_accuracy"])

    @classmethod
    def cli(cls, parser):
        parser.add_argument("--border-conv", action="store_true")
        super().cli(parser)
