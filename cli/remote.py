from cli.utils import helper, ExtendCLI, relpath
import argparse, importlib, os

action_clis = ("train", "download")
platforms = {}

@ExtendCLI
def remote_action(parser, action):
    if action not in action_clis:
        raise argparse.ArgumentTypeError(
            'argument "action" must be one of ' + ", ".join(action_clis))
    parser.add_argument("argv", nargs="...")
    return action

def cli(parser):
    for platform in os.listdir(relpath("cli", "platforms")):
        if platform.endswith(".py"):
            importlib.import_module("cli.platforms." + platform[:-3])

    parser.set_defaults(call=helper(parser))
    subparsers = parser.add_subparsers()
    for k, v in platforms.items():
        v(subparsers.add_parser(k))
