#!/usr/bin/env python3
from _common import config_from_args, configured_parser


if __name__ == "__main__":
    args = configured_parser("Select the domain-balanced reference/candidate subset.").parse_args()
    from asr_data_selection.subset import run

    run(config_from_args(args))
