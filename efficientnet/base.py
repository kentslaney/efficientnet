import os
import tensorflow as tf
import tensorflow_datasets as tfds
from utils import (
    tpu_prep, WarmedExponential, LRTensorBoard, strftime, parse_strategy,
    PresetFlag, cli_builder, relpath, Checkpointer
)
from preprocessing import (
    PrepStretched, RandAugmentCropped, RandAugmentPadded
)

class Trainer:
    path, _format, checkpoint, length = None, None, None, None
    opt = tf.keras.optimizers.Adam

    @classmethod
    def cli(cls, parser):
        parser.add_argument("--id", dest="suffix", help=(
            "name template for the training directory, compiled using "
            "python's string formatting; time is the only currently supported "
            "variable"))
        parser.add_argument(
            "--batch", metavar="SIZE", type=int, help=(
                "batch size per visible device (1 unless distributed)"))
        parser.add_argument("--epochs", metavar="N", type=int, help=(
            "how many epochs to run"))
        parser.add_argument(
            "--lr", dest="learning_rate", type=float, help=(
                "model learning rate per example per batch"))

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument("--no-decay", dest="decay", action="store_false",
                            help="don't decay the learning rate")
        group.add_argument("--decay-warmup", type=int, help=(
            "number of epochs to warm up learning rate"))
        group.add_argument("--decay-factor", type=float, help=(
            "lr decay per epoch after warmup"))

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

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "--no-base", action="store_const", const=None, dest="base", help=(
                "prevents saving checkpoints or tensorboard logs to disk"))
        group.add_argument("--job-dir", dest="base", metavar="PATH", help=(
            "prefix for training directory"))
        group.add_argument("--resume", metavar="PATH", help="load from path")

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "--distribute", metavar=("STRATEGY", "DEVICE"), nargs="+", help=(
                "what distribution strategy to use from tf.distribute, and "
                "what devices to distribute to (usually no specified device "
                "implies all visable devices); leaving this unspecified "
                "results in no strategy, and uses tensorflow's default "
                "behavior"))
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

        parser.set_defaults(call=cls.train)

    @classmethod
    def train(cls, **kw):
        cls(**kw).fit()

    @cli_builder
    def __init__(self, base=relpath("jobs"), data_format=None, batch=64,
                 distribute=None, epochs=1000, decay=True, suffix="{time}",
                 learning_rate=6.25e-5, decay_warmup=5, decay_factor=0.99,
                 resume=None, **kw):
        self.batch, self.learning_rate = batch, learning_rate
        self.data_format, self.epochs = data_format, epochs
        self.decay_warmup, self.decay_factor = decay_warmup, decay_factor
        self.should_decay, self.callbacks = decay, []
        self.step = tf.Variable(0, dtype=tf.int32, name="step")
        self.epoch = tf.Variable(0, dtype=tf.int32, name="epoch")

        self.distribute(*parse_strategy(distribute))
        if resume is not None:
            self.path = os.path.abspath(resume)
        elif base is not None:
            self.path = os.path.join(base, suffix.format(time=strftime()))

        self.build(**kw)

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

    def distribute(self, strategy, tpu=False):
        self.strategy, self.tpu = strategy, tpu
        if tf.distribute.in_cross_replica_context():
            self.batch *= strategy.num_replicas_in_sync
        self.learning_rate *= self.batch
        self.opt = self.opt(self.learning_rate)
        self.opt = tf.tpu.CrossShardOptimizer(self.opt) if tpu else self.opt

    def mapper(self, f):
        return lambda x, y: (f(x), tf.one_hot(y, self.outputs))

    def preprocess(self, train, validation=None):
        tune = tf.data.experimental.AUTOTUNE
        options = tf.data.Options()
        options.experimental_deterministic = False
        if self.tpu:
            train, validation = map(tpu_prep, (train, validation))

        self.dataset = self.dataset.shuffle(self.batch * 4).map(
            self.mapper(train), tune).batch(self.batch, True).prefetch(
                tune).with_options(options)

        if validation is not None and self.validation is not None:
            self.validation = self.validation.map(self.mapper(
                validation), tune).batch(self.batch, True).prefetch(
                    tune).with_options(options)

    def checkpoint(self):
        if self.path is None:
            return

        ckpts, logs = (os.path.join(self.path, i) for i in ("ckpts", "logs"))
        tf.io.gfile.makedirs(ckpts)

        ckptr = Checkpointer(os.path.join(ckpts, "ckpt"), self.epoch,
                             self.model, opt=self.opt, step=self.step)
        if tf.io.gfile.listdir(ckpts):
            path = ckptr.restore(ckpts)
            print(f"Loading model from checkpoint {path}")
        else:
            print(f'Writing to training directory {self.path}')

        self.callbacks += [LRTensorBoard(logs, update_freq=64), ckptr]

    def compile(self, *args, **kw):
        self.checkpoint()
        self.model.compile(self.opt, *args, **kw)

    def fit(self):
        self.model.fit(
            self.dataset, validation_data=self.validation, epochs=self.epochs,
            callbacks=self.callbacks, batch_size=self.batch,
            initial_epoch=self.epoch.numpy().item())

    def build(self):
        if self.should_decay:
            assert self.length is not None
            self.callbacks.append(WarmedExponential(
                self.learning_rate, self.length / self.batch,
                self.decay_warmup, self.decay_factor, self.step))

class TFDSTrainer(Trainer):
    @classmethod
    def cli(cls, parser):
        parser.add_argument("--dataset", help=(
            "choose which TFDS dataset to train on; must be classification "
            "and support as_supervised"))
        parser.add_argument("--data-dir", help=(
            "default directory for TFDS data, supports GCS buckets"))

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument("--train-all", dest="holdout", action="store_false",
                           help="train on all available splits")
        group.add_argument(
            "--no-eval", dest="holdout", action="store_true", help=(
                "don't run evaluation steps, but hold still hold out one "
                "split"))
        super().cli(parser)

    @cli_builder
    def build(self, dataset, holdout=None, data_dir=None, **kw):
        self.builder(dataset, data_dir)
        data, info = tfds.load(dataset, data_dir=data_dir, with_info=True,
                               shuffle_files=True, as_supervised=True)
        sizes, _, data = zip(*sorted([
            (info.splits[k].num_examples, i, v) for i, (k, v) in enumerate(
                data.items())], reverse=True))

        hold = -1 if holdout is False else -2
        for i in data[:hold]:
            data[hold] = data[hold].concatenate(i)
        data = data[hold:] if len(data) >= -hold else (data, None)
        data = (data[0], None) if holdout is True else data
        self.dataset, self.validation = data
        self.outputs = info.features["label"].num_classes
        self.length = sum(sizes[:len(sizes) + hold + 1])
        super().build(**kw)

    @classmethod
    def builder(cls, dataset, data_dir):
        if hasattr(cls, "_tfds_" + dataset):
            getattr(cls, "_tfds_" + dataset)(data_dir)

    @classmethod
    def _tfds_imagenet2012(cls, data_dir):
        data_dir = os.path.expanduser(
            tfds.core.constants.DATA_DIR if data_dir is None else data_dir)
        data_base = os.path.join(data_dir, "downloads", "manual")
        downloader = tfds.download.DownloadManager(download_dir=data_base)
        url_base = f"https://image-net.org/data/ILSVRC/2012/"
        files = ("ILSVRC2012_img_train.tar", "ILSVRC2012_img_val.tar")
        paths = [tfds.download.Resource(path=os.path.join(
            data_base, fname), url=url_base + fname) for fname in files]
        paths = [path for path in paths if not tf.io.gfile.exists(path.path)]
        if paths:
            downloader.download(paths)

class RandAugmentTrainer(Trainer):
    @classmethod
    def cli(self, parser):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument("--pad", action="store_true", help=(
            "pads the augmented images instead of cropping them"))
        group.add_argument(
            "--no-augment", dest="augment", action="store_false",
            help="don't augment the input")
        parser.add_argument("--size", metavar="N", type=int, help=(
            "force the input image to be a certain size, will default to the "
            "recommended size for the preset if unset"))
        super().cli(parser)

    @cli_builder
    def build(self, size=None, augment=True, pad=False, **kw):
        super().build(**kw)
        self.size = (size, size) if type(size) == int else size
        self.augment, self.pad = augment, pad
        train = PrepStretched if not self.augment else \
            RandAugmentPadded if self.pad else RandAugmentCropped
        train = train(self.size, data_format=self.data_format)
        validation = PrepStretched(self.size, data_format=self.data_format)
        super().preprocess(train, validation)
