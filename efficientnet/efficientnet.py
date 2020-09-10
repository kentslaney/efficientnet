import tensorflow as tf
from functools import partial
from MBConv import MBConv

class Block:
    def __init__(self, size=3, outputs=32, expand=6, strides=1, repeats=1):
        self.size, self.outputs, self.expand, self.strides, self.repeats = \
            size, outputs, expand, strides, repeats

    def __dict__(self):
        return {"size": self.size, "outputs": self.outputs,
                "expand": self.expand, "strides": self.strides}

class Embedding(tf.keras.Model):
    base = MBConv
    conv = tf.keras.layers.Conv2D
    bn = tf.keras.layers.BatchNormalization
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
                 data_format='channels_first', name=None):
        super().__init__(name)
        self.width, self.depth, self.divisor, self.dropout, self.connect, \
            self.data_format = width, depth, divisor, dropout, connect, \
                data_format
        self.stem = self.round_filters(stem)
        self.total = sum(block.repeats for block in self.blocks)

        self.conv = partial(self.conv, padding='same', data_format=data_format,
                            kernel_initializer=self.initialization)
        channel = -1 if data_format == 'channels_last' else 1
        self.bn = partial(self.bn, axis=channel)

        for block in self.blocks:
            block.outputs = self.round_filters(block.outputs)
            block.repeats = self.round_repeats(block.repeats)

    def kwargs(self, i):
        return {"dropout": self.dropout * i / self.total,
                "data_format": self.data_format}

    def round_filters(self, filters):
        filters *= self.width
        rounded = int(filters + self.divisor / 2)
        rounded = rounded // self.divisor * self.divisor
        rounded = max(self.divisor, rounded)

        if rounded < 0.9 * filters:
            rounded += depth_divisor
        return int(rounded)

    def round_repeats(self, repeats):
        return int(math.ceil(self.depth * repeats))

    def build(self, input_shape):
        self._stem_conv = self.conv(self.stem, 3, 2, use_bias=False)
        self._stem_bn = self.bn()

        self._blocks, i = [], 0
        for block in self.blocks:
            for _ in range(block.repeats):
                self._blocks.append(self.base(**block, **self.kwargs(i)))
                i += 1

    def call(self, inputs, training):
        x = self._stem_conv(inputs)
        x = tf.nn.swish(self._stem_bn(x, training))

        for i, block in enumerate(self._blocks):
            x = block.call(x, training)
        return x

class Classifier(Embedding):
    dense_init = tf.keras.initializers.VarianceScaling(
        scale=1 / 3, mode='fan_out', distribution='uniform')
    drop = tf.keras.layers.Dropout

    def __init__(*args, head_drop=0.2, outputs=1000, head=1280, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs, self.head_drop = outputs, head_drop
        self.head = self.round_filters(head)
        self._head_drop = self.drop(head_drop)
        self.pool = keras.layers.GlobalAveragePooling2D(
            data_format=self.data_format)

    def build(self, input_shape):
        super().build(input_shape)
        self._head_conv = self.conv(self.stem, 1, use_bias=False)
        self._head_bn = self.bn()
        self._fc = tf.keras.layers.Dense(
            self.outputs, kernel_initializer=self.dense_init)

    def call(self, inputs, training):
        x = super().call(inputs, training)
        return self._fc(self._head_drop(self.pool(x)))

