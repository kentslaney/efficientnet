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

import tensorflow as tf
import tensorflow_datasets as tfds
from preprocessing.augment import RandAugmentCropped, RandAugmentPadded
from train import TFDSTrainer

def download(dataset, data_dir):
    TFDSTrainer.builder(dataset, data_dir)
    tfds.load(dataset, split="train", data_dir=data_dir)

def main(dataset, pad, augment, data_dir):
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

if __name__ == "__main__":
    from utils import cli_call
    cli_call(preview_cli)
