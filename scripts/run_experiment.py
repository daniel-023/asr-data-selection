#!/usr/bin/env python3
import importlib

from _common import config_from_args, configured_parser
from asr_data_selection.config import STAGES, write_resolved_config


MODULES = {
    "subset": "asr_data_selection.subset",
    "pseudolabels": "asr_data_selection.pseudolabels",
    "embeddings": "asr_data_selection.features",
    "classify": "asr_data_selection.classification",
    "plot": "asr_data_selection.plotting",
}


if __name__ == "__main__":
    parser = configured_parser("Run the pooled embedding comparison workflow.")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES), help="Ordered stages to run.")
    args = parser.parse_args()
    config = config_from_args(args)
    snapshot = write_resolved_config(config)
    print(f"[config] wrote {snapshot}")
    selected = set(args.stages)
    for stage in STAGES:
        if stage in selected:
            print(f"[stage] {stage}")
            importlib.import_module(MODULES[stage]).run(config)
