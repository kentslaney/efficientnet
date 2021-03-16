import tensorflow as tf
from functools import partial
from utils import Conv2D

class MBConv(tf.keras.layers.Layer):
    conv = Conv2D
    depthwise = tf.keras.layers.DepthwiseConv2D
    bn = tf.keras.layers.BatchNormalization
    initialization = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_out', distribution='untruncated_normal')
    drop = tf.keras.layers.Dropout

    def activation(self, *args, **kwargs):
        return tf.nn.swish(*args, **kwargs)

    def __init__(self, size=3, outputs=32, expand=6, residuals=True, strides=1,
                 se_ratio=0.25, dropout=0, data_format='channels_first', **kw):
        super().__init__(**kw)
        assert data_format in ('channels_first', 'channels_last')
        self.size, self.outputs, self.expand, self.strides, self.se_ratio, \
            self.dropout, self.data_format = size, outputs, expand, strides, \
                se_ratio, dropout, data_format
        self.channel = -1 if data_format == 'channels_last' else 1
        self.spacial = (1, 2) if data_format == 'channels_last' else (2, 3)
        self.residuals = residuals and bool(tf.reduce_all(
            strides == tf.constant(1)))

        self.bn = partial(self.bn, axis=self.channel)
        self.conv = partial(self.conv, padding='same', data_format=data_format,
                            kernel_initializer=self.initialization)
        self.depthwise = partial(self.depthwise, data_format=data_format,
                                 depthwise_initializer=self.initialization,
                                 padding='same')

    def pool(self, inputs):
        return tf.reduce_mean(inputs, self.spacial, True)

    def build(self, input_shape):
        input_channels = input_shape[self.channel]
        self.residuals = self.residuals and input_channels == self.outputs
        filters = input_channels * self.expand

        if self.expand != 1:
            self._expand_conv = self.conv(filters, 1, use_bias=False)
            self._expand_bn = self.bn()

        self._depthwise = self.depthwise(
            self.size, self.strides, use_bias=False)
        self._depthwise_bn = self.bn()

        if 0 < self.se_ratio <= 1:
            reduced = tf.math.maximum(1, tf.cast(
                input_channels * self.se_ratio, tf.int32))
            self._se_reduce = self.conv(reduced, 1, activation=self.activation)
            self._se_expand = self.conv(filters, 1, activation='sigmoid')

        self._project_conv = self.conv(self.outputs, 1, use_bias=False)
        self._project_bn = self.bn()
        self.drop = self.drop(self.dropout, noise_shape=(
            input_shape[0], 1, 1, 1)) if self.dropout > 0 else lambda x, *y: x

    def call(self, inputs, training=False):
        x = inputs

        if self.expand != 1:
            x = self._expand_conv(x)
            x = self.activation(self._expand_bn(x, training))

        x = self._depthwise(x)
        x = self.activation(self._depthwise_bn(x, training))

        if 0 < self.se_ratio <= 1:
            x *= self._se_expand(self._se_reduce(self.pool(x)))

        x = self._project_bn(self._project_conv(x), training)

        if self.residuals:
            x = self.drop(x, training) + inputs

        return x

    def get_config(self):
        config = super().get_config()
        keys = ("size", "outputs", "expand", "residuals", "strides",
                "se_ratio", "dropout", "data_format")
        config.update({i: getattr(self, i) for i in keys})
        return config
