#!/usr/bin/env python3
from _common import config_from_args, configured_parser


if __name__ == "__main__":
    args = configured_parser("Plot accuracy for the pooled MLP embedding comparison.").parse_args()
    from asr_data_selection.plotting import run

    run(config_from_args(args))
