import tensorflow as tf
import tensorflow_datasets as tfds
from preprocessing.augment import RandAugmentCropped, RandAugmentPadded

def main(dataset, pad, augment):
    data, info = tfds.load(dataset, split="train", with_info=True,
                           shuffle_files=True)

    if augment:
        aug = RandAugmentPadded if pad else RandAugmentCropped
        aug = aug((224, 224), data_format="channels_last")
        data = data.map(lambda x: {**x, "image": aug(x["image"]) / 2 + 0.5})

    fig = tfds.show_examples(data, info)
    fig.show()

def cli(parser):
    parser.add_argument(
        "dataset", nargs="?", default="imagenette/320px-v2", help=(
            'choose a TFDS dataset to augment, must have "image" key and '
            'a "train" split'))

    augment_cli(parser)
    parser.set_defaults(call=main)
    return parser

def augment_cli(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--pad", action="store_true", help=(
        "pads the augmented images instead of cropping them"))
    group.add_argument("--no-augment", dest="augment", action="store_false",
                        help="don't augment the input")
    return parser

if __name__ == "__main__":
    from utils import cli_call
    cli_call(cli)
