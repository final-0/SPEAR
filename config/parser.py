import argparse
import json


def parse_args(parser):
    """Config JSON values, overridden by every command-line argument that is not None."""
    entry = parser.parse_args()
    with open(entry.cfg) as f:
        args = argparse.Namespace(**json.load(f))
    for key, value in vars(entry).items():
        if value is not None:
            setattr(args, key, value)
    return args
