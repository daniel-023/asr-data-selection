from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asr_data_selection.config import add_config_arguments, load_config, print_config_summary


def configured_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_config_arguments(parser)
    return parser


def config_from_args(args: argparse.Namespace) -> dict:
    config = load_config(args.config, args.overrides)
    print_config_summary(config)
    return config
