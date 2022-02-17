from cli import main
from efficientnet.utils import CallParser
from absl.flags import argparse_flags

class ArgumentParser(CallParser, argparse_flags.ArgumentParser):
    pass

if __name__ == "__main__":
    main(ArgumentParser())
