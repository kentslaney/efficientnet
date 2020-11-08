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

import os
import tensorflow as tf
import tensorflow_datasets as tfds
from cli.utils import strftime
from cli.data import augment_cli
from models.utils import tpu_prep
from preprocessing.augment import (
    PrepStretched, RandAugmentCropped, RandAugmentPadded)

class WarmedExponential(tf.keras.callbacks.Callback):
    def __init__(self, scale, units, warmup, decay, freq=64):
        self.scale, self.units, self.warmup, self.decay, self.freq = scale, \
            units, warmup, decay, freq
        self.step = tf.Variable(0, tf.int32)

    def on_train_batch_begin(self, batch, logs=None):
        self.step.assign_add(1)
        if self.step % self.freq == 1:
            x = tf.cast(self.step, tf.float32) / self.units
            lr = self.scale * x / self.warmup if x < self.warmup else \
                self.scale * self.decay ** (x - self.warmup)
            self.model.optimizer.lr.assign(lr)

class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'learning_rate': self.model.optimizer.lr})
        super().on_epoch_end(epoch, logs)

class Trainer:
    _path, _format, checkpoint = None, None, None
    mapper = lambda _, f: lambda x, *y: (f(x),) + y
    opt = tf.keras.optimizers.Adam

    def __init__(self, batch, learning_rate):
        self.batch, self.learning_rate = batch, learning_rate
        self.callbacks = []

    @property
    def path(value):
        return self._path

    @path.setter
    def path(self, value):
        path, name = value
        formatted = name.format(time=strftime())
        self._path = path = os.path.join(path, formatted)
        ckpts, logs = os.path.join(path, "ckpts"), os.path.join(path, "logs")
        os.makedirs(ckpts, exist_ok=True)

        prev = ((i.stat().st_ctime, i.path) for i in os.scandir(ckpts))
        prev = max(prev, default=(None, None))[1]
        if prev is not None:
            self.checkpoint = prev
        elif formatted != name:
            print(f'Writing to training directory {path}')

        self.callbacks += [
            LRTensorBoard(logs, update_freq=64),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(ckpts, "ckpt_{epoch}"))]

    @property
    def data_format(self):
        if self._format is None:
            return "channels_last" if self.tpu or not bool(
                tf.config.experimental.list_physical_devices('GPU')) \
                else "channels_first"
        return self._format

    @data_format.setter
    def data_format(self, value):
        self._format = value

    def decay(self):
        self.callbacks.append(WarmedExponential(
            self.learning_rate, self.decay_unit / self.batch, 1, 0.97))

    def distribute(self, strategy, tpu=False):
        self.strategy, self.tpu = strategy, tpu
        if tf.distribute.in_cross_replica_context():
            self.batch *= strategy.num_replicas_in_sync
        self.learning_rate *= self.batch
        self.opt = self.opt(self.learning_rate)
        self.opt = tf.tpu.CrossShardOptimizer(self.opt) if tpu else self.opt

    def preprocess(self, train, validation=None):
        tune = tf.data.experimental.AUTOTUNE
        options = tf.data.Options()
        options.experimental_deterministic = False
        if self.tpu:
            train, validation = map(tpu_prep, (train, validation))

        self.dataset = self.dataset.shuffle(2048).map(
            self.mapper(train), tune).batch(self.batch, True).prefetch(
                tune).with_options(options)

        if validation is not None and self.validation is not None:
            self.validation = self.validation.map(self.mapper(
                validation), tune).batch(self.batch, True).prefetch(
                    tune).with_options(options)

    def compile(self, *args, **kwargs):
        self.model.compile(self.opt, *args, **kwargs)

    def fit(self, epochs):
        if self.checkpoint is not None:
            self.model.load_weights(self.checkpoint)

        self.model.fit(self.dataset, validation_data=self.validation,
                       callbacks=self.callbacks, batch_size=self.batch,
                       epochs=epochs)

    @classmethod
    def cli(cls, parser):
        parser.add_argument("--decay-unit", type=float, default=3e6, help=(
            "base units for the learning rate in number of examples, warms up "
            "1 unit, decays 0.97"))

    def build(self, decay_unit, **kwargs):
        self.decay_unit = decay_unit

class TFDSTrainer(Trainer):
    @classmethod
    def _tfds_imagenet2012(cls, data_dir):
        data_dir = os.path.expanduser(
            tfds.core.constants.DATA_DIR if data_dir is None else data_dir)
        data_base = os.path.join(data_dir, "downloads", "manual")
        downloader = tfds.download.DownloadManager(download_dir=data_base)
        # https://archive.is/0Q3LX#selection-13351.0-13351.32
        key = "dd31405981ef5f776aa17412e1f0c112"
        url_base = f"http://image-net.org/challenges/LSVRC/2012/{key}/"
        files = ("ILSVRC2012_img_train.tar", "ILSVRC2012_img_val.tar")
        paths = [tfds.download.Resource(path=os.path.join(
            data_base, fname), url=url_base + fname) for fname in files]
        paths = [path for path in paths if not tf.io.gfile.exists(path.path)]
        if paths:
            downloader.download(paths)

    @classmethod
    def builder(cls, dataset, data_dir):
        if hasattr(cls, "_tfds_" + dataset):
            getattr(cls, "_tfds_" + dataset)(data_dir)

    def build(self, dataset=None, holdout=None, data_dir=None, **kwargs):
        self.builder(dataset, data_dir)
        data, info = tfds.load(dataset, data_dir=data_dir, with_info=True,
                               shuffle_files=True, as_supervised=True)
        data = [i[1] for i in sorted(
            data.items(), key=lambda x: -info.splits[x[0]].num_examples)]

        hold = -1 if holdout is False else -2
        for i in data[:hold]:
            data[hold] = data[hold].concatenate(i)
        data = data[hold:] if len(data) > abs(hold) - 1 else (data, None)
        data = (data[0], None) if holdout is True else data
        self.dataset, self.validation = data
        self.outputs = info.features["label"].num_classes
        super().build(**kwargs)

    @classmethod
    def cli(cls, parser):
        parser.add_argument("--dataset", default="imagenette/320px-v2", help=(
            "choose which TFDS dataset to train on; must be classification "
            "and support as_supervised"))

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument("--train-all", dest="holdout", action="store_false",
                           help="train on all available splits")
        group.add_argument(
            "--no-eval", dest="holdout", action="store_true", help=(
                "don't run evaluation steps, but hold still hold out one "
                "split"))
        parser.set_defaults(holdout=None)
        super().cli(parser)

class RandAugmentTrainer(Trainer):
    def build(self, size=None, augment=True, pad=False, **kwargs):
        self.size = (size, size) if type(size) == int else size
        self.augment, self.pad = augment, pad
        super().build(**kwargs)

    def preprocess(self):
        train = PrepStretched if not self.augment else \
            RandAugmentPadded if self.pad else RandAugmentCropped
        train = train(self.size, data_format=self.data_format)
        validation = PrepStretched(self.size, data_format=self.data_format)
        super().preprocess(train, validation)

    @classmethod
    def cli(self, parser):
        augment_cli(parser)
        super().cli(parser)
