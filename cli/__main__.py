import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# hack to get gcloud deployment to line up with normal CLI calls
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir))

from cli.utils import cli_call
from cli import data, train

def cli(parser):
    subparsers = parser.add_subparsers()
    train.cli(subparsers.add_parser("train", help="Train a network"))
    data.preview_cli(subparsers.add_parser("preview", help=(
        "Preview an augmented dataset")))
    data.download_cli(subparsers.add_parser("download", help=(
        "Download a dataset")))
    parser.set_defaults(call=lambda: parser.parse_args(["-h"]))

if __name__ == "__main__":
    cli_call(cli)
