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
