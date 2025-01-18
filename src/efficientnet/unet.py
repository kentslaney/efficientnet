import tensorflow as tf
from functools import partial
from .base import RandAugmentTrainer, TFDSTrainer
from .utils import RequiredLength, cli_builder, perimeter_pixel, Default
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
        self.upsample = lambda: tf.keras.layers.UpSampling2D(
                (2, 2), data_format=data_format)
        self._build([16, 32, 64, 128, 256])

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

    def _build(self, filters):
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
                *self.conv_block(filters[-1], 1),
                *self.conv_block(filters[-1], 1)]
        for size in filters[:0:-1]:
            self.sequential += [
                    self.upsample(),
                    LayerFlags.POP,
                    *self.conv_block(size, 1),
                    *self.conv_block(size, 1),
                    LayerFlags.SKIP]
            self.skips += self.skip_block(size, 1)
        self.sequential += [
                self.conv(filters[0], kernel_size=(1, 1), activation="sigmoid")]

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
    # sample_coefficient is the ratio of local variance to object variance
    # mean is of L_\inf distance of local samples (geometrically distributed)
    @cli_builder
    def __init__(
            self, data_format, sample_coefficient=4, samples=256, mean=16,
            reweight=1):
        super().__init__()
        self.data_format = data_format
        self.dist = tfp.distributions.Geometric(probs=(1 / mean,))
        self.channel = 3 if data_format == 'channels_last' else 1
        self.spacial = tuple(i for i in range(1, 4) if i != self.channel)
        self.sample_coefficient, self.samples = sample_coefficient, samples
        self.reweight = reweight

    def call(self, y_true, y_pred):
        # penalize each labeled object's latent space variance
        tile = [y_pred.shape[i] if i == self.channel else 1 for i in range(4)]
        tile = tile[1:]
        recast = [
                y_true.shape[i] if i == self.channel else y_pred.shape[i]
                for i in range(y_pred.ndim)]
        y_true = tf.function(input_signature=[tf.TensorSpec(recast, tf.int32)])(
                lambda x: x)(y_true)
        @tf.function
        def outer(arg):
            (mask, pred), std = arg, 0.
            lim = mask[0, 0, 0] & 31
            for j in tf.range(31):
                if j >= lim:
                    continue
                obj = tf.cast(mask & 2 ** (5 + j), tf.bool)[..., 0]
                obj = tf.keras.ops.expand_dims(obj, self.channel - 1)
                obj = tf.tile(obj, tile)
                obj = tf.gather_nd(pred, tf.where(obj))
                std += tf.where(tf.size(obj) == 0, 0., tf.keras.ops.mean(
                    tf.keras.ops.std(obj)))
            return tf.keras.ops.mean(std)
        std = tf.map_fn(outer, (y_true, y_pred), fn_output_signature=tf.float32)
        std = tf.keras.ops.mean(std)

        # geometric L_\inf pixel distribution as positive/negative samples
        sample_shape = tuple(
                self.samples if i == self.channel else y_pred.shape[i]
                for i in range(y_pred.ndim))
        r = 1 + self.dist.sample(sample_shape)
        r = tf.cast(r[..., 0], tf.int32)
        theta = tf.keras.random.randint(sample_shape, 0, 2 ** 31 - 1)
        delta_r, delta_c = perimeter_pixel(r, theta % (8 * r))
        max_r, max_c = (tf.shape(y_pred)[i] for i in self.spacial)
        prefix, postfix = ((None,) * 2), 2 if self.channel == 1 else 1
        prefix, postfix = prefix[:postfix], prefix[postfix:]
        r, c = tf.meshgrid(tf.range(max_r), tf.range(max_c))
        r = r[(*prefix, *(slice(None),) * 2, *postfix)]
        c = c[(*prefix, *(slice(None),) * 2, *postfix)]
        r_, c_ = r + delta_r, c + delta_c
        in_bounds = tf.keras.ops.logical_and(
                tf.keras.ops.logical_and(r_ > 0, c_ > 0),
                tf.keras.ops.logical_and(r_ < max_r, c_ < max_c))
        r_, c_ = tf.where(in_bounds, r_, r), tf.where(in_bounds, c_, c)
        batch_indices = tf.range(sample_shape[0])[(slice(None), *(None,) * 3)]
        batch_indices = tf.broadcast_to(batch_indices, in_bounds.shape)
        coords = tf.stack((batch_indices, r_, c_), -1)
        tile = [self.samples if i == self.channel else 1 for i in range(5)]
        sample_idx = tf.reshape(tf.range(self.samples), tile[:-1])
        sample_idx = tf.broadcast_to(sample_idx, sample_shape)[..., None]
        coords = tf.concat((
                coords[..., :self.channel], sample_idx,
                coords[..., self.channel:]), -1)
        coords = tf.expand_dims(coords, self.channel + 1)
        y_sample = tf.gather_nd(y_pred, coords)
        y_cmp = tf.gather_nd(y_true, coords)
        y_ref = tf.keras.ops.tile(tf.expand_dims(y_pred, self.channel), tile)
        y_obj = tf.keras.ops.tile(tf.expand_dims(y_true, self.channel), tile)
        cmp = y_obj == y_cmp
        reweight = self.reweight / (1 - tf.keras.ops.mean(
                tf.cast(cmp, tf.float32), self.channel, keepdims=True))
        sample = (y_sample - y_ref) ** 2 * tf.where(cmp, 1., -reweight)
        sample = tf.keras.ops.sum(sample) / tf.cast(
                tf.keras.ops.sum(in_bounds), tf.float32)

        return std + self.sample_coefficient * sample

class UNetTrainer(RandAugmentTrainer, TFDSTrainer):
    def opt(self, lr):
        return tf.keras.optimizers.Adam(lr)

    @classmethod
    def cli(cls, parser):
        parser.add_argument("--sample-coefficient", type=float)
        parser.add_argument("--samples", type=float)
        parser.add_argument("--samples-mean-dist", type=float)
        parser.add_argument("--samples-negative-reweight", type=float)
        super().cli(parser)

    @cli_builder
    def __init__(
            self, batch=2, learning_rate=1e-6, dataset="ref_coco", size=448,
            sample_coefficient=Default, samples=Default,
            samples_mean_dist=Default, samples_negative_reweight=Default, **kw):
        super().__init__(
                batch=batch, learning_rate=learning_rate, dataset=dataset,
                size=size, **kw)
        self.model = ResUNet(self.data_format)
        self.compile(InstanceLoss(
                self.data_format, sample_coefficient, samples,
                samples_mean_dist, samples_negative_reweight
            ))
