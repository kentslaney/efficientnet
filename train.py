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

import data, os, importlib, argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from utils import (
    relpath, NoStrategy, PresetFlag, HelpFormatter, ExtendCLI, tpu_prep)
from preprocessing.augment import (
    PrepStretched, RandAugmentCropped, RandAugmentPadded)

class Trainer:
    _base, _format, checkpoint = None, None, None
    mapper = lambda _, f: lambda x, *y: (f(x),) + y
    opt = tf.keras.optimizers.Adam

    def __init__(self, batch, learning_rate):
        self.batch, self.learning_rate = batch, learning_rate
        self.callbacks = []

    @property
    def base(value):
        return self._base

    @base.setter
    def base(self, value):
        base, name = value
        formatted = name.format(time=strftime())
        self._base = base = os.path.join(os.path.abspath(value), formatted)
        ckpts, logs = os.path.join(base, "ckpts"), os.path.join(base, "logs")
        os.makedirs(ckpts, exist_ok=True)
        tb_manual_metrics = tf.summary.create_file_writer(
            os.path.join(logs, "metrics"))
        tb_manual_metrics.set_as_default()

        prev = ((i.stat().st_ctime, i.path) for i in os.scandir(ckpts))
        prev = max(prev, default=(None, None))[1]
        if prev is not None:
            self.checkpoint = prev
        elif formatted != name:
            print(f'Writing to training directory {formatted}')

        self.callbacks += [
            tf.keras.callbacks.TensorBoard(logs, update_freq=64),
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
        warmup, schedule = 5, tf.keras.optimizers.schedules.ExponentialDecay(
            self.learning_rate, 2.4, 0.97, True)
        def logger(epoch):
            lr = schedule(epoch) if epoch < warmup else epoch / warmup
            tf.summary.scalar('learning rate', data=lr, step=epoch)
            return lr

        self.callbacks.append(tf.keras.callbacks.LearningRateScheduler(logger))

    def distribute(self, strategy, tpu=False):
        self.strategy, self.tpu = strategy, tpu
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

        self.dataset = self.dataset.shuffle(1000).map(
            self.mapper(train), tune).batch(
                self.batch, True).prefetch(tune).with_options(options)

        if validation is not None and self.validation is not None:
            self.validation = self.validation.map(
                self.mapper(validation), tune).batch(
                    self.batch, True).prefetch(tune).with_options(options)

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
        pass

    def build(self, **kwargs):
        pass

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
        downloader.download([tfds.download.Resource(
            url=url_base + fname, path=os.path.join(data_base, fname))
                for fname in files])

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
        parser.add_argument(
            "--size", metavar="N", type=int, default=None, help=(
                "force the input image to be a certain size, will default to "
                "the recommended size for the preset if unset"))

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
        data.augment_cli(parser)
        super().cli(parser)

def cli_strategy(distribute):
    distribute, *devices = distribute or [None]
    assert not distribute or hasattr(tf.distribute, distribute)
    devices = None if len(devices) == 1 else \
        devices[1] if distribute == "OneDeviceStrategy" else devices

    tpu = distribute == "TPUStrategy"
    if tpu:
        devices = tf.distribute.cluster_resolver.TPUClusterResolver(*devices)
        tf.config.experimental_connect_to_cluster(devices)
        tf.tpu.experimental.initialize_tpu_system(devices)

    distribute = NoStrategy() if distribute is None else distribute
    distribute = getattr(tf.distribute, distribute)(devices) \
        if type(distribute) is str else distribute

    return distribute, tpu

def model_cli(parser, model):
    module = importlib.import_module(model + ".train")
    trainer = module.Trainer
    trainer.cli(parser)
    return trainer

def main(model, base, data_format, batch, distribute, epochs, decay, suffix,
         learning_rate, **kwargs):
    model = model(batch, learning_rate)
    model.distribute(*cli_strategy(distribute))
    model.data_format = data_format
    if base is not None:
        model.base = base, suffix

    model.build(**kwargs)
    if decay:
        model.decay()

    model.preprocess()
    model.fit(epochs)

def cli(parser):
    parser.add_argument("model", action=ExtendCLI(model_cli), help=(
            "select the model that you want to train based on the path "
            "relative to the base directory"))
    parser.add_argument(
        "--dir", metavar="DIR", dest="suffix", default="{time}", help=(
            "name template for the training directory, compiled using "
            "python's string formatting; time is the only currently supported "
            "variable"))
    parser.add_argument(
        "--batch", metavar="SIZE", type=int, default=128, help=(
            "batch size per visible device (1 unless distributed)"))
    parser.add_argument("--epochs", metavar="N", type=int, default=1000, help=(
        "how many epochs to run"))
    parser.add_argument(
        "--lr", dest="learning_rate", type=float, default=6.25e-5, help=(
            "model learning rate per example per batch"))
    parser.add_argument("--no-decay", dest="decay", action="store_false",
                        help="don't decay the learning rate")
    parser.add_argument("--name", default=None, help=(
        "name to assign to the model; potentially useful to differentiate "
        "exports"))

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--channels-last", dest="data_format", action="store_const",
        const="channels_last", help=(
            "forces the image data format to be channels last, which is "
            "the default for CPUs and TPUs"))
    group.add_argument(
        "--channels-first", dest="data_format", action="store_const",
        const="channels_first", help=(
            "forces the image data format to be channels first, which is "
            "the default for GPUs"))
    parser.set_defaults(data_format=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--no-base", action="store_const", const=None, dest="base", help=(
            "prevents saving checkpoints or tensorboard logs to disk"))
    group.add_argument(
        "--base", metavar="PATH", default=relpath("runs"), help=(
            "prefix for training directory"))

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--distribute", metavar=("STRATEGY", "DEVICE"), nargs="+",
        default=None, help=(
            "what distribution strategy to use from tf.distribute, and "
            "what devices to distribute to (usually no specified device "
            "implies all visable devices); leaving this unspecified results "
            "in no strategy, and uses tensorflow's default behavior"))
    group.add_argument(
        "--tpu", metavar="DEVICE", nargs="*", dest="distribute",
        action=PresetFlag("TPUStrategy"), help=(
            "use TPUStrategy for distribution; equivalent to --distribute "
            "TPUStrategy [DEVICE...]"))
    group.add_argument(
        "--mirror", metavar="DEVICE", nargs="*", dest="distribute",
        action=PresetFlag("MirroredStrategy"), help=(
            "use MirroredStrategy for distribution; equivalent to "
            "--distribute MirroredStrategy [DEVICE...]"))

    parser.set_defaults(call=main)

if __name__ == "__main__":
    from utils import ArgumentParser
    args = cli(ArgumentParser(fallthrough=True))
    args.parse_args().call()
