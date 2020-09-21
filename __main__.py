import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from utils import cli_call
import data
import train

def cli(parser):
    subparsers = parser.add_subparsers()
    train.cli(subparsers.add_parser("train", fallthrough=True, help=(
        "Train a network")))
    data.cli(subparsers.add_parser("preview", help=(
        "Preview an augmented dataset")))
    return parser

cli_call(cli)
