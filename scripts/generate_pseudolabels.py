#!/usr/bin/env python3
from _common import config_from_args, configured_parser


if __name__ == "__main__":
    parser = configured_parser("Generate Whisper pseudolabels for the selected subset.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore the resumable checkpoint.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    args = parser.parse_args()
    from asr_data_selection.pseudolabels import run

    run(config_from_args(args), overwrite=args.overwrite, limit=args.limit)
