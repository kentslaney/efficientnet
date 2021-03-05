import tensorflow as tf
import importlib, os
from glob import iglob
from cli.utils import relpath, NoStrategy, PresetFlag, HelpFormatter, helper

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

def main(model, base, data_format, batch, distribute, epochs, decay, suffix,
         learning_rate, **kwargs):
    model = model(batch, learning_rate)
    model.distribute(*cli_strategy(distribute))
    model.data_format = data_format
    if base is not None:
        model.path = base, suffix

    model.build(**kwargs)
    if decay:
        model.decay()

    model.preprocess()
    model.fit(epochs)

def cli(parser):
    subparsers = parser.add_subparsers()
    for model in [os.path.normpath(i).rsplit(os.path.sep, 2)[1] for i in iglob(
            relpath("models", "*", "train.py"))]:
        subparser = subparsers.add_parser(model)
        trainer = importlib.import_module(f"models.{model}.train").Trainer
        subparser.set_defaults(model=trainer, call=main)
        subcli(subparser)
        trainer.cli(subparser)
    parser.set_defaults(call=helper(parser))

def subcli(parser):
    parser.add_argument("--id", dest="suffix", default="{time}", help=(
            "name template for the training directory, compiled using "
            "python's string formatting; time is the only currently supported "
            "variable"))
    parser.add_argument(
        "--batch", metavar="SIZE", type=int, default=64, help=(
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
    group.add_argument("--job-dir", dest="base", metavar="PATH", help=(
        "prefix for training directory"))
    parser.set_defaults(base=relpath("jobs"))

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
