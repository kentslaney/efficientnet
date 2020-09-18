import tensorflow as tf
import tensorflow_datasets as tfds
from efficientnet.trainer import Trainer
from border.efficientnet import Trainer as BorderTrainer
from datetime import datetime
import os, argparse

relpath = lambda *args: os.path.join(
    os.path.dirname(os.path.abspath(__file__)), *args)

class NoStrategy:
    def __init__(self):
        self.num_replicas_in_sync = 1

    def scope(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def main(dataset, border_conv, no_base, base, name, channels_last, batch,
         distribute, augment_only, epochs, **kwargs):
    if augment_only:
        from preprocessing.randaugment import RandAugmentCrop, RandAugmentPad
        data, info = tfds.load(dataset, split="train", with_info=True,
                               shuffle_files=True)
        aug = RandAugmentPad if pad else RandAugmentCrop
        aug = aug((224, 224), data_format="channels_last")
        data = data.map(lambda x: {**x, "image": aug(x["image"]) / 2 + 0.5},
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        fig = tfds.show_examples(data, info)
        fig.show()
        return

    splits = tfds.builder(dataset).info.splits
    splits = sorted(splits.keys(), key=lambda x: -splits[x].num_examples)
    data, info = tfds.load(dataset, split=splits, with_info=True,
                           shuffle_files=True, as_supervised=True)
    data = list(data)

    for i in range(len(data)):
        if info.features[info.supervised_keys[0]].shape[-1] == 1:
            data[i] = data[i].map(
                lambda x, y: (tf.tile(x, (1, 1, 3)), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for i in data[:-2]:
        data[-2] = data[-2].concatenate(i)
    data = data[-2:] if len(data) > 1 else (data, None)

    distribute, *devices = distribute or [None]
    assert not distribute or hasattr(tf.distribute, distribute)
    if channels_last is None:
        channels_last = distribute == "TPUStrategy" or not bool(
            tf.config.experimental.list_physical_devices('GPU'))
    devices = None if len(devices) == 1 else \
        devices[1] if distribute == "OneDeviceStrategy" else devices
    if distribute == "TPUStrategy":
        devices = tf.distribute.cluster_resolver.TPUClusterResolver(*devices)
        tf.config.experimental_connect_to_cluster(devices)
        tf.tpu.experimental.initialize_tpu_system(devices)

    distribute = getattr(tf.distribute, distribute)(devices) \
        if type(distribute) is str else distribute or NoStrategy()
    batch *= distribute.num_replicas_in_sync
    with distribute.scope():
        model = BorderTrainer if border_conv else Trainer
        model = model.from_preset(
            **kwargs, outputs=info.features["label"].num_classes, data_format=(
                "channels_last" if channels_last else "channels_first"))

    if no_base:
        callbacks = []
    else:
        time = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
        formatted = name.format(time=time)
        base = relpath(base, formatted)
        ckpts, logs = os.path.join(base, "ckpts"), os.path.join(base, "logs")
        os.makedirs(ckpts, exist_ok=True)
        prev = ((i.stat().st_ctime, i.path) for i in os.scandir(ckpts))
        prev = max(prev, default=(None, None))[1]
        if prev is not None:
            print(f"Restoring weights from checkpoint {prev}")
            model.load_weights(prev)
        elif formatted != name:
            print(f'Writing to training directory {formatted}')

        callbacks = [
            tf.keras.callbacks.TensorBoard(logs, update_freq=100),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(ckpts, "ckpt_{epoch}")),
        ]

    model.fit(*data, callbacks=callbacks, batch_size=batch, epochs=epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an EfficientNet classifier")
    parser.add_argument(
        "--dataset", metavar="NAME", default="imagenette/320px-v2", help=(
            "choose which TFDS dataset to train on; must be classification "
            "and support as_supervised"))
    parser.add_argument("--preset", metavar="N", type=int, default=0, help=(
        "which preset to use; 0-7 correspond to B0 to B7, and 8 is L2"))
    parser.add_argument("--border-conv", action="store_true", help=(
        "use border corrected convolutions"))
    parser.add_argument("--no-base", action="store_true", help=(
        "prevents saving checkpoints or tensorboard logs to disk"))
    parser.add_argument("--base", metavar="PATH", default="runs", help=(
        "prefix for training directory; relative paths are relative to the "
        "location of this script"))
    parser.add_argument("--name", metavar="DIR", default="{time}", help=(
        "name template for the training directory, compiled using python's "
        "string formatting; time is the only currently supported variable"))
    parser.add_argument("--batch", metavar="SIZE", type=int, default=32, help=(
        "batch size per visible device (1 unless distributed)"))
    parser.add_argument(
        "--distribute", metavar=("STRATEGY", "DEVICE"), nargs="+",
        default=None, help=(
            "what distribution strategy to use from tf.distribute, and "
            "what devices to distribute to (usually no specified device "
            "implies all visable devices); leaving this unspecified results "
            "in no strategy, and uses tensorflow's default behavior"))
    parser.add_argument("--pad", action="store_true", help=(
        "pads the augmented images instead of cropping them"))
    parser.add_argument("--augment-only", action="store_true", help=(
        "only run augmentation for visualization"))
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help= "don't augment the input")
    parser.add_argument("--size", metavar="N", type=int, default=None, help=(
        "force the input image to be a certain size, will default to the "
        "recommended size for the preset if unset"))
    parser.add_argument("--epochs", metavar="N", type=int, default=1000, help=(
        "how many epochs to run"))

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--channels-last", action="store_true", help=(
        "forces the image data format to be channels last"))
    group.add_argument(
        "--channels-first", dest="channels_last", action="store_false", help=(
        "forces the image data format to be channels last"))
    parser.set_defaults(channels_last=None)

    main(**vars(parser.parse_args()))
