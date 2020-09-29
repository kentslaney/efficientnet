import tensorflow as tf
import tensorflow_datasets as tfds
from preprocessing.augment import RandAugmentCropped, RandAugmentPadded

def download(dataset, data_dir):
    from train import TFDSTrainer
    TFDSTrainer.builder(dataset, data_dir)
    tfds.load(dataset, split="train", data_dir=data_dir)

def main(dataset, pad, augment, data_dir):
    from train import TFDSTrainer
    TFDSTrainer.builder(dataset, data_dir)
    data, info = tfds.load(dataset, split="train", data_dir=data_dir,
                           with_info=True, shuffle_files=True)

    if augment:
        aug = RandAugmentPadded if pad else RandAugmentCropped
        aug = aug((224, 224), data_format="channels_last")
        data = data.map(lambda x: {**x, "image": tf.clip_by_value(
            aug(x["image"]) / 5 + 0.5, 0., 1.)})

    fig = tfds.show_examples(data, info)
    fig.show()

def preview_cli(parser):
    parser.add_argument(
        "dataset", nargs="?", default="imagenette/320px-v2", help=(
            'choose a TFDS dataset to augment, must have "image" key and '
            'a "train" split'))

    augment_cli(parser)
    parser.set_defaults(call=main)

def download_cli(parser):
    parser.add_argument("dataset", help="choose a TFDS dataset to download")
    parser.add_argument("--job-dir", dest="data_dir")
    data_cli(parser)
    parser.set_defaults(call=download)

def augment_cli(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--pad", action="store_true", help=(
        "pads the augmented images instead of cropping them"))
    group.add_argument("--no-augment", dest="augment", action="store_false",
                        help="don't augment the input")
    data_cli(parser)

def data_cli(parser):
    parser.add_argument("--data-dir", default=None, help=(
        "default directory for TFDS data, supports GCS buckets"))
