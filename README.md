# ASR Data Selection

Domain-aware ASR data selection with pooled semantic, acoustic, and fused embeddings.

This repository reproduces the 13,200-sample comparison between SBERT embeddings from ground-truth transcripts vs Whisper large-v3 pseudolabels. It extracts SBERT, WavLM, and MFA-Conformer vectors, projects them to 256 dimensions, concatenates embeddings, trains MLP domain classifiers, and plots accuracy by embedding family.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install `ffmpeg` separately so Whisper can load audio files.

## MFA-Conformer Setup

MFA-based extraction requires an external source checkout and pretrained checkpoint:

```bash
git clone https://github.com/zyzisyz/mfa_conformer mfa_conformer_repo
git -C mfa_conformer_repo checkout 1b9c229948f8dbdbe9370937813ec75d4b06b097

mkdir -p checkpoints
# Download the checkpoint linked from:
# https://github.com/ductuantruong/mfa_conformer_sv
mv /path/to/downloaded.ckpt checkpoints/MFA_conformer.ckpt
```

See [Environment and external artifacts](docs/environment.md) for the direct checkpoint download link and implementation note.

## Run

Run the complete experiment:

```bash
python scripts/run_experiment.py \
  --config configs/embedding_comparison_13200.yaml
```

Run selected dependent stages after subset generation:

```bash
python scripts/run_experiment.py \
  --config configs/embedding_comparison_13200.yaml \
  --stages pseudolabels embeddings classify plot
```

Every stage also has a thin script under `scripts/` for debugging or reruns. Use `--set dotted.key=value` to override a YAML setting without editing the committed config.

## Documentation

- [Reproduction guide](docs/reproduction.md)
- [Environment and external artifacts](docs/environment.md)
- [13,200-sample experiment report](docs/embedding_comparison_13200.md)

Generated audio, manifests, embeddings, checkpoints, and classifier outputs are written under `artifacts/embedding_comparison_13200/` and excluded from version control. Compact plotting CSVs and report figures are committed under `docs/`.
