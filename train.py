import tensorflow as tf
import tensorflow_datasets as tfds
from efficientnet.border import BorderTrainer
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

def main(dataset, preset, no_base, base, name, channels_last, batch,
         distribute):
    distribute, *devices = distribute or [None]
    assert not distribute or hasattr(tf.distribute, distribute)
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
        model = BorderTrainer.from_preset(preset, data_format=(
            "channels_last" if channels_last else "channels_first"))

    data, info = tfds.load(dataset, split=["train", "test"], with_info=True,
                           shuffle_files=True, as_supervised=True)
    data = list(data)

    # repeat channels so augmentations have 3 input channels as expected
    for i in range(len(data)):
        if info.features[info.supervised_keys[0]].shape[-1] == 1:
            data[i] = data[i].map(
                lambda x, y: (tf.tile(x, (1, 1, 3)), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if no_base:
        callbacks = []
    else:
        time = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
        formatted = name.format(time=time)
        base = relpath(base, name)
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

    model.fit(*data, callbacks=callbacks, batch_size=batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an EfficientNet classifier")
    parser.add_argument("--dataset", metavar="NAME", default="mnist", help=(
        "choose which TFDS dataset to train on; must be classification and "
        "support as_supervised"))
    parser.add_argument("--preset", metavar="N", type=int, default=0, help=(
        "which preset to use; 0-7 correspond to B0 to B7, and 8 is L2"))
    parser.add_argument("--no-base", action="store_true", help=(
        "prevents saving checkpoints or tensorboard logs to disk"))
    parser.add_argument("--base", metavar="PATH", default="runs", help=(
        "prefix for training directory; relative paths are relative to the "
        "location of this script"))
    parser.add_argument("--name", metavar="DIR", default="{time}", help=(
        "name template for the training directory, compiled using python's "
        "string formatting; time is the only currently supported variable"))
    parser.add_argument("--channels-last", action="store_true", help=(
        "sets the image data format to channels last, which is faster for CPU "
        "and TPU"))
    parser.add_argument("--batch", metavar="SIZE", type=int, default=32, help=(
        "batch size per visible device (1 unless distributed)"))
    parser.add_argument(
        "--distribute", metavar=("STRATEGY", "DEVICE"), nargs="+",
        default=None, help=(
            "what distribution strategy to use from tf.distribute, and "
            "what devices to distribute to (usually no specified device "
            "implies all visable devices); leaving this unspecified results "
            "in no strategy, and uses tensorflow's default behavior"))

    main(**vars(parser.parse_args()))
