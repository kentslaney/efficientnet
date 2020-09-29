from cli.utils import helper, ExtendCLI, relpath
import argparse, importlib, os

actions = {"choices": {"train", "download"}, "nargs": "..."}
platforms = {}

def cli(parser):
    for platform in os.listdir(relpath("cli", "platforms")):
        if platform.endswith(".py"):
            importlib.import_module("cli.platforms." + platform[:-3])

    parser.set_defaults(call=helper(parser))
    subparsers = parser.add_subparsers()
    for k, v in platforms.items():
        v(subparsers.add_parser(k))
