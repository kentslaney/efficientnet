import argparse, os
from functools import partial, wraps
from datetime import datetime

def HelpFormatter(parser=None):
    class HelpFormatter(argparse.HelpFormatter):
        def __init__(self, *args, **kwargs):
            if hasattr(parser, "_group"):
                group, parser._group = parser._group, None
                while group is not None:
                    parser._add_container_actions(group)
                    group = group._group
            super().__init__(*args, **kwargs)

        def _format_args(self, action, default_metavar):
            if hasattr(action, "format_meta"):
                return action.format_meta(self._metavar_formatter(
                    action, default_metavar))
            else:
                return super()._format_args(action, default_metavar)
    return HelpFormatter

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs = {"formatter_class": HelpFormatter(self), **kwargs}
        super().__init__(*args, **kwargs)
        kwargs["add_help"] = False
        self._group, self.copy = None, lambda: self.__class__(*args, **kwargs)

    @property
    def group(self):
        if self._group is None:
            self._group = self.copy()
        return self._group

    def parse_known_args(self, args=None, namespace=None):
        res, args = super().parse_known_args(args, namespace)
        if self._group is not None:
            res, args = self._group.parse_known_args(args, res)
        self._group = None

        if hasattr(res, "call"):
            res.caller = partial(res.call, **{i: j for i, j in vars(
                res).items() if i not in ("call", "caller")})
        return res, args

relpath = lambda *args: os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, *args))

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

def ExtendCLI(f):
    class ExtendCLI(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            setattr(namespace, self.dest, f(parser.group, value))
    return ExtendCLI

strftime = lambda: datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
helper = lambda parser: lambda: parser.parse_args(["-h"])

def cli_call(*f):
    parser = ArgumentParser()
    for i in f:
        i(parser)

    return parser.parse_args().caller()
