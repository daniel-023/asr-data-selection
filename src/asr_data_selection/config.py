from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "embedding_comparison_13200.yaml"
STAGES = ("subset", "pseudolabels", "embeddings", "classify", "plot")


def parse_scalar(value: str) -> Any:
    return yaml.safe_load(value)


def apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must use dotted.key=value syntax: {override}")
    dotted_key, raw_value = override.split("=", 1)
    keys = dotted_key.split(".")
    if not all(keys):
        raise ValueError(f"Override contains an empty key: {override}")
    current: dict[str, Any] = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            raise KeyError(f"Unknown configuration key: {dotted_key}")
        current = current[key]
    if keys[-1] not in current:
        raise KeyError(f"Unknown configuration key: {dotted_key}")
    current[keys[-1]] = parse_scalar(raw_value)


def load_config(path: Path | str = DEFAULT_CONFIG, overrides: Sequence[str] = ()) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    config = copy.deepcopy(config)
    for override in overrides:
        apply_override(config, override)
    config["_meta"] = {"config_path": str(config_path), "repo_root": str(REPO_ROOT)}
    return config


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiment YAML config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML setting using dotted.key=value syntax. Repeat as needed.",
    )


def artifact_root(config: dict[str, Any]) -> Path:
    return resolve_repo_path(config["artifacts"]["root"])


def artifact_path(config: dict[str, Any], key: str) -> Path:
    return artifact_root(config) / config["artifacts"][key]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def write_resolved_config(config: dict[str, Any]) -> Path:
    root = artifact_root(config)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "resolved_config.yaml"
    serializable = {key: value for key, value in config.items() if key != "_meta"}
    output.write_text(yaml.safe_dump(serializable, sort_keys=False), encoding="utf-8")
    return output


def print_config_summary(config: dict[str, Any]) -> None:
    print(f"[config] experiment={config['experiment']['name']}")
    print(f"[config] artifacts={artifact_root(config)}")
    print(f"[config] source={config['_meta']['config_path']}")


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))
