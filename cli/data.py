import tensorflow as tf
import tensorflow_datasets as tfds
from preprocessing.augment import RandAugmentCropped, RandAugmentPadded
from cli.utils import strftime
import os.path

def download(dataset, data_dir):
    from models.train import TFDSTrainer
    TFDSTrainer.builder(dataset, data_dir)
    tfds.load(dataset, split="train", data_dir=data_dir)

def main(dataset, pad, augment, data_dir, job_dir, fname, size):
    from models.train import TFDSTrainer
    TFDSTrainer.builder(dataset, data_dir)
    data, info = tfds.load(dataset, split="train", data_dir=data_dir,
                           with_info=True, shuffle_files=True)

    if augment:
        size = size or 224
        aug = RandAugmentPadded if pad else RandAugmentCropped
        aug = aug((size, size), data_format="channels_last")
        data = data.map(lambda x: {**x, "image": tf.clip_by_value(
            aug(x["image"]) / 5 + 0.5, 0., 1.)})

    fig = tfds.show_examples(data, info)
    if job_dir is not None:
        formatted = fname.format(time=strftime())
        path = os.path.join(job_dir, formatted)
        if formatted != fname:
            print(f"saving figure to {path}")
        with tf.io.gfile.GFile(path, "wb") as fp:
            fig.savefig(fp)

def preview_cli(parser):
    parser.add_argument(
        "dataset", nargs="?", default="imagenette/320px-v2", help=(
            'choose a TFDS dataset to augment, must have "image" key and '
            'a "train" split'))
    parser.add_argument("--job-dir", default=None, help=(
        "directory to write the output image to"))
    parser.add_argument("--fname", default="{time}.png", help=(
        "output file name"))

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
    parser.add_argument("--size", metavar="N", type=int, default=None, help=(
        "force the input image to be a certain size, will default to the "
        "recommended size for the preset if unset"))
    data_cli(parser)

def data_cli(parser):
    parser.add_argument("--data-dir", default=None, help=(
        "default directory for TFDS data, supports GCS buckets"))
