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
