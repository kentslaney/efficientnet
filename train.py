import data, os, importlib, argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from utils import relpath, NoStrategy, PresetFlag, HelpFormatter
from preprocessing.augment import (
    PrepStretched, RandAugmentCropped, RandAugmentPadded)

class Trainer:
    _base, _format, checkpoint = None, None, None
    map_wrap = lambda _, f: lambda x, *y: (f(x),) + y

    def __init__(self, batch, learning_rate):
        self.batch, self.learning_rate = batch, learning_rate
        self.opt = tf.keras.optimizers.RMSprop(learning_rate, 0.9, 0.9, 0.001)
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
        self.opt = tf.tpu.CrossShardOptimizer(self.opt) if tpu else self.opt

    def preprocess(self, train, validation=None):
        tune = tf.data.experimental.AUTOTUNE
        self.dataset = self.dataset.shuffle(1000).map(
            self.map_wrap(train), num_parallel_calls=tune).batch(
                self.batch).prefetch(tune)

        if validation is not None and self.validation is not None:
            self.validation = self.validation.map(
                self.map_wrap(validation), num_parallel_calls=tune).batch(
                    self.batch).prefetch(tune)

    def compile(self, *args, **kwargs):
        self.model.compile(self.opt, *args, **kwargs)

    def fit(self, epochs):
        if self.checkpoint is not None:
            self.model.load_weights(self.checkpoint)

        self.model.fit(self.dataset, validation_data=self.validation,
                       callbacks=self.callbacks, batch_size=self.batch,
                       epochs=epochs)

    def cli(self, parser):
        pass

    def build(self, **kwargs):
        pass

class TFDSTrainer(Trainer):
    def build(self, dataset=None, holdout=None, **kwargs):
        splits = tfds.builder(dataset).info.splits
        splits = sorted(splits.keys(), key=lambda x: -splits[x].num_examples)
        data, info = tfds.load(dataset, split=splits, with_info=True,
                               shuffle_files=True, as_supervised=True)
        data = list(data)

        hold = -1 if holdout is False else -2
        for i in data[:hold]:
            data[hold] = data[hold].concatenate(i)
        data = data[hold:] if len(data) > abs(hold) - 1 else (data, None)
        data = (data[0], None) if holdout is True else data
        self.dataset, self.validation = data
        self.outputs = info.features["label"].num_classes
        super().build(**kwargs)

    def cli(self, parser):
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

def main(argv, model, base, data_format, batch, distribute, epochs, decay,
         learning_rate, **kwargs):
    module = importlib.import_module(model + ".train")
    trainer = module.Trainer(batch, learning_rate)
    trainer.distribute(*cli_strategy(distribute))
    trainer.data_format = data_format
    if base is not None:
        trainer.base = base, name

    parser = argparse.ArgumentParser(formatter_class=HelpFormatter)
    trainer.cli(parser)
    trainer.build(**vars(parser.parse_args(argv)))
    if decay:
        trainer.decay()

    trainer.preprocess()
    trainer.fit(epochs)

def cli(parser):
    parser.add_argument("model", metavar="NAME", help=(
        "select the model that you want to train based on the path relative "
        "to the base directory"))
    parser.add_argument(
        "--dir", metavar="DIR", dest="name", default="{time}", help=(
            "name template for the training directory, compiled using "
            "python's string formatting; time is the only currently supported "
            "variable"))
    parser.add_argument(
        "--batch", metavar="SIZE", type=int, default=128, help=(
            "batch size per visible device (1 unless distributed)"))
    parser.add_argument("--size", metavar="N", type=int, default=None, help=(
        "force the input image to be a certain size, will default to the "
        "recommended size for the preset if unset"))
    parser.add_argument("--epochs", metavar="N", type=int, default=1000, help=(
        "how many epochs to run"))
    parser.add_argument("--lr", metavar="FLOAT", dest="learning_rate",
                        type=float, default=0.01, help="model learning rate")
    parser.add_argument("--no-decay", dest="decay", action="store_false",
                        help="don't decay the learning rate")
    parser.add_argument("--name", metavar="NAME", default=None, help=(
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
    return parser

if __name__ == "__main__":
    from utils import ArgumentParser
    cli(ArgumentParser(fallthrough=True)).parse_args().call()
