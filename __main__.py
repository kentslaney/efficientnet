import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from utils import cli_call
import data
import train

def cli(parser):
    subparsers = parser.add_subparsers()
    train.cli(subparsers.add_parser("train", help="Train a network"))
    data.preview_cli(subparsers.add_parser("preview", help=(
        "Preview an augmented dataset")))
    data.download_cli(subparsers.add_parser("download", help=(
        "Download a dataset")))
    return parser

cli_call(cli)
