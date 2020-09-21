import argparse, os
from functools import partial
from datetime import datetime

class HelpFormatter(argparse.HelpFormatter):
    def _format_args(self, action, default_metavar):
        if hasattr(action, "format_meta"):
            return action.format_meta(self._metavar_formatter(
                action, default_metavar))
        else:
            return super()._format_args(action, default_metavar)

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, fallthrough=False, formatter_class=HelpFormatter,
                 **kwargs):
        super().__init__(*args, formatter_class=formatter_class, **kwargs)
        self.fallthrough = fallthrough

    def parse_known_args(self, args=None, namespace=None):
        args, argv = super().parse_known_args(args, namespace)
        if hasattr(args, "call"):
            args.caller = args.call
            if self.fallthrough:
                args.argv, argv = argv, []
            args.caller = partial(args.caller, **{i: j for i, j in vars(
                args).items() if i not in ("call", "caller")})
        return args, argv

relpath = lambda *args: os.path.join(
    os.path.dirname(os.path.abspath(__file__)), *args)

class NoStrategy:
    def __init__(self):
        self.num_replicas_in_sync = 1

    def scope(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def RequiredLength(minimum, maximum):
    class RequiredLength(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            super().__init__(option_strings, dest, nargs="*", **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            if not minimum <= len(values) <= maximum:
                raise argparse.ArgumentTypeError(
                    f'argument "{self.dest}" requires between {minimum} and '
                    f'{maximum} arguments')
            setattr(namespace, self.dest, values)

        def format_meta(self, metavars):
            metavars = metavars(maximum)
            formatted = ' %s' * minimum + ' [%s' * (maximum - minimum) \
                + ']' * maximum
            return formatted[1:] % metavars

    return RequiredLength

def PresetFlag(*preset):
    class PresetFlag(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, list(preset) + values)
    return PresetFlag

strftime = lambda: datetime.today().strftime("%Y_%m_%d_%H_%M_%S")

def cli_call(*f):
    parser = ArgumentParser()
    for i in f:
        i(parser)

    return parser.parse_args().caller()
