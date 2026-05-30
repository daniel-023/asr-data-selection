# ASR Data Selection

Embedding-based analysis pipeline for domain-aware ASR data selection using semantic, acoustic, and fused speech representations.

## Report

The current 13,200-sample embedding comparison, including the data split, Whisper pseudo-label generation, text normalization, embedding configurations, MLP settings, results, and report figures, is documented in:

- [Embedding Comparison for Domain-Aware ASR Data Selection](docs/embedding_comparison_13200.md)

## Primary Pipeline

```text
extract_subset.py
generate_pseudolabels.py
extract_embeddings.py
extract_sbert_embeddings.py
classify_embeddings.py
plot_mlp_comparison.py
```

NSCC job files and running instructions are documented in [nscc_workflow.md](nscc_workflow.md).

## Setup

Create a virtual environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For local test execution, install `requirements-dev.txt` instead.

## External Artifacts

Audio, generated manifests containing sample-level metadata, embeddings, classifier outputs, virtual environments, and model checkpoints are intentionally excluded from version control.

MFA-Conformer extraction requires:

- a local `MFA_conformer.ckpt` checkpoint; and
- the external [MFA-Conformer source repository](https://github.com/zyzisyz/mfa_conformer) checked out locally as `mfa_conformer_repo/`.
