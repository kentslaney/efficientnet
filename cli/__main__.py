import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# hack to get gcloud deployment to line up with normal CLI calls
sys.path[0] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir)

from cli import data, train, remote
from cli.utils import helper

def cli(parser):
    subparsers = parser.add_subparsers()
    train.cli(subparsers.add_parser("train", help="Train a network"))
    data.preview_cli(subparsers.add_parser("preview", help=(
        "Preview an augmented dataset")))
    data.download_cli(subparsers.add_parser("download", help=(
        "Download a dataset")))
    remote.cli(subparsers.add_parser("remote", help=(
        "Run commands remotely")))
    parser.set_defaults(call=helper(parser))

if __name__ == "__main__":
    from cli.utils import cli_call
    cli_call(cli)
