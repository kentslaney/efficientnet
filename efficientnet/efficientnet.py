import tensorflow as tf
from functools import partial
from mbconv import MBConv
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
                 data_format='channels_first', pretranspose=False, **kw):
        self.width, self.depth = width, depth
        self.divisor, self.dropout = divisor, dropout
        self.data_format, self.pretranspose = data_format, pretranspose

        self.stem = self.round_filters(stem)
        self.total = sum(block.repeats for block in self.blocks)

        for block in self.blocks:
            block.outputs = self.round_filters(block.outputs)
            block.repeats = self.round_repeats(block.repeats)
        super().__init__(**kw)

    def kw(self, overall, repeat):
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

    def build(self, input_shape):
        self.conv = partial(
            self.base.conv, padding='same', data_format=self.data_format,
            kernel_initializer=self.initialization)
        channel = -1 if self.data_format == 'channels_last' else 1
        self.bn = partial(self.base.bn, axis=channel)

        self._stem_conv = self.conv(self.stem, 3, 2, use_bias=False)
        self._stem_bn = self.bn()
 
        self._blocks, i = [], 0
        for block in self.blocks:
            for j in range(block.repeats):
                self._blocks.append(self.base(**{**block, **self.kw(i, j)}))
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

    def __init__(self, *args, head_drop=0.2, head=1280, outputs=1000, **kw):
        super().__init__(*args, **kw)
        self.head_drop, self.outputs = head_drop, outputs
        self.head = self.round_filters(head)

    def build(self, input_shape):
        super().build(input_shape)
        self._head_drop = self.drop(self.head_drop)
        self._head_conv = self.conv(self.head, 1, use_bias=False)
        self._head_bn = self.bn()
        self._fc = tf.keras.layers.Dense(
            self.outputs, "softmax", kernel_initializer=self.dense_init)
        self.pool = tf.keras.layers.GlobalAveragePooling2D(
            data_format=self.data_format)

    def call(self, inputs, training):
        x = super().call(inputs, training)
        x = tf.nn.swish(self._head_bn(self._head_conv(x), training))
        return self._fc(self._head_drop(self.pool(x), training))
