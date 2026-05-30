#!/usr/bin/env python3
from _common import config_from_args, configured_parser


if __name__ == "__main__":
    args = configured_parser("Extract projected pooled embeddings and configured fusion matrices.").parse_args()
    from asr_data_selection.features import run

    run(config_from_args(args))
