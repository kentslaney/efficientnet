import tensorflow as tf
from functools import partial
from src.base import RandAugmentTrainer, TFDSTrainer
from src.utils import RequiredLength, cli_builder, perimeter_pixel
import tensorflow_probability as tfp

class LayerFlags:
    PUSH = object()
    POP = object()
    SKIP = object()

# arXiv:1711.10684v1
class ResUNet(tf.keras.Model):
    def __init__(self, data_format):
        super().__init__()
        self.conv = partial(
                tf.keras.layers.Conv2D, padding='same', data_format=data_format)
        self.channel = -1 if data_format == 'channels_last' else 1
        self.bn = partial(tf.keras.layers.BatchNormalization, axis=self.channel)
        self.act = lambda: tf.keras.layers.Activation("relu")
        self.upsample = lambda: tf.keras.layers.UpSampling2D((2, 2))
        self.build([16, 32, 64, 128, 256])

    def conv_block(self, filters, strides):
        return [
                self.bn(), self.act(),
                self.conv(filters, kernel_size=(3, 3), strides=strides)]

    def skip_block(self, filters, strides):
        return [
                self.conv(filters, kernel_size=(1, 1), strides=strides),
                self.bn(), self.act(), LayerFlags.SKIP]

    def shortcuts(self):
        def body(x):
            x = i(x)
            while (j := next(value)) != LayerFlags.SKIP:
                x = j(x)
            return x
        value = iter(self.skips)
        for i in value:
            yield body

    def build(self, filters):
        self.sequential = [
                LayerFlags.PUSH,
                self.conv(filters[0], kernel_size=(3, 3), strides=1),
                *self.conv_block(filters[0], 1)]
        self.skips = self.skip_block(filters[0], 1)
        for i in range(1, len(filters)):
            self.sequential += [
                    LayerFlags.PUSH,
                    *self.conv_block(filters[i], 2),
                    *self.conv_block(filters[i], 1)]
            if i < len(filters) - 1:
                self.skips += self.skip_block(filters[i], 2)
        self.sequential += [
                self.conv(filters[-1], kernel_size=(3, 3), strides=1),
                self.conv(filters[-1], kernel_size=(3, 3), strides=1)]
        for size in filters[:0:-1]:
            self.sequential += [
                    self.upsample(),
                    LayerFlags.POP,
                    *self.conv_block(size, 1),
                    *self.conv_block(size, 1),
                    LayerFlags.SKIP]
            self.skips += self.skip_block(size, 1)

    def call(self, x):
        stack, skip, preempted = [], None, True
        shortcut = self.shortcuts()
        for layer in self.sequential:
            match layer:
                case LayerFlags.PUSH if preempted:
                    stack.append(x)
                    skip = x
                    preempted = False
                case LayerFlags.PUSH:
                    x += next(shortcut)(skip)
                    skip = x
                    stack.append(x)
                case LayerFlags.POP:
                    x = tf.concat((x, stack.pop()), self.channel)
                    skip = x
                case LayerFlags.SKIP:
                    x += next(shortcut)(skip)
                case _:
                    x = layer(x)
        return x

    def compute_loss(self, x, y, y_pred, sample_weight):
        return self.loss.call(y, y_pred)

class InstanceLoss(tf.keras.Loss):
    # sample coefficient starting point = 100 / (size ** 2 * samples)
    def __init__(self, data_format, sample_coefficient=4e-4, samples=5, mean=3):
        super().__init__()
        self.data_format = data_format
        self.dist = tfp.distributions.Geometric(probs=(1 / mean,))
        self.channel = 3 if data_format == 'channels_last' else 1
        self.spacial = tuple(i for i in range(1, 4) if i != self.channel)
        self.sample_coefficient, self.samples = sample_coefficient, samples

    def call(self, y_true, y_pred):
        # penalize each labeled object's latent space variance
        tile = [y_pred.shape[i] if i == self.channel else 1 for i in range(4)]
        tile = tile[1:]
        @tf.function
        def outer(i):
            mask, pred, std = y_true[i], y_pred[i], 0.
            for j in tf.range(mask[0, 0, 0] & 31):
                obj = tf.cast(mask & 2 ** (5 + j), tf.bool)[..., 0]
                obj = tf.keras.ops.expand_dims(obj, self.channel - 1)
                obj = tf.tile(obj, tile)
                obj = tf.gather_nd(pred, tf.where(obj))
                std += tf.where(tf.size(obj) == 0, 0., tf.keras.ops.sum(
                    tf.keras.ops.std(obj)))
            return tf.keras.ops.sum(std)
        std = tf.map_fn(
                outer, tf.range(tf.shape(y_true)[0]),
                fn_output_signature=tf.float32)
        std = tf.keras.ops.sum(std)

        # geometric L_\inf pixel distribution as positive/negative samples
        sample_shape = y_pred.shape[:1] + tuple(
                y_pred.shape[i] for i in self.spacial) + (self.samples,)
        r = 1 + self.dist.sample(sample_shape)
        r = tf.cast(r[..., 0], tf.int32)
        theta = tf.keras.random.randint(sample_shape, 0, 2 ** 31 - 1)
        delta_r, delta_c = perimeter_pixel(r, theta % (8 * r))
        max_r, max_c = (tf.shape(y_pred)[i] for i in self.spacial)
        prefix, postfix = ((None,) * 2), 2 if self.channel == 1 else 1
        prefix, postfix = prefix[:postfix], prefix[postfix:]
        r, c = tf.meshgrid(tf.range(max_r), tf.range(max_c))
        r, c = r[(*prefix, :, :, *postfix)], c[(*prefix, :, :, *postfix)]
        r_, c_ = r + delta_r, c + delta_c
        oob = tf.keras.ops.logical_or(
                tf.keras.ops.logical_or(r_ < 0, c_ < 0),
                tf.keras.ops.logical_or(r_ >= max_r, c_ >= max_c))
        r_, c_ = tf.where(oob, r, r_), tf.where(oob, c, c_)
        batch_indices = tf.range(sample_shape[0])[:, *(None,) * 3]
        batch_indices = tf.broadcast_to(batch_indices, oob.shape)
        coords = tf.stack((batch_indices, r_, c_), -1)
        y_sample = tf.gather_nd(y_pred, coords)
        y_cmp = tf.gather_nd(y_true, coords)
        tile = [self.samples if i == self.channel else 1 for i in range(5)]
        y_ref = tf.keras.ops.tile(tf.expand_dims(y_pred, self.channel), tile)
        y_obj = tf.keras.ops.tile(tf.expand_dims(y_true, self.channel), tile)
        sample = (y_sample - y_ref) ** 2 * tf.where(y_obj == y_cmp, 1., -1.)
        sample = tf.keras.ops.sum(sample)

        return (std + self.sample_coefficient * sample) / tf.shape(y_true)[0]

class UNetTrainer(RandAugmentTrainer, TFDSTrainer):
    def opt(self, lr):
        return tf.keras.optimizers.Adam(lr)

    @cli_builder
    def __init__(self, learning_rate=1e-6, dataset="ref_coco", size=224, **kw):
        super().__init__(learning_rate=learning_rate, decay=decay,
                         augment=augment, dataset=dataset, size=size, **kw)

        self.model = ResUNet(self.data_format)
        self.compile(InstanceLoss(self.data_format))
