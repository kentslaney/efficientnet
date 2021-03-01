import unittest
from models.border.test import *

def cli(parser):
    parser.add_argument("tests", nargs="*", help=(
        "which tests should be run (defaults to all)"))
    parser.set_defaults(call=main)

def main(tests, **kwargs):
    tests = ["cli test"] + ["test." + i for i in tests] if tests else None
    unittest.TestProgram(argv=tests)
