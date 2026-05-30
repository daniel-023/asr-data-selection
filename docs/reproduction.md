# Reproducing the 13,200-Sample Experiment

## Configuration

The canonical settings live in:

```text
configs/embedding_comparison_13200.yaml
```

Relative paths resolve from the repository root. Each runner accepts repeated overrides:

```bash
python scripts/classify_mlp.py \
  --config configs/embedding_comparison_13200.yaml \
  --set classification.epochs=1
```

The complete runner saves its resolved configuration to:

```text
artifacts/embedding_comparison_13200/resolved_config.yaml
```

## Stage Order

Run the complete workflow:

```bash
python scripts/run_experiment.py \
  --config configs/embedding_comparison_13200.yaml
```

The stages execute in this order:

```text
subset -> pseudolabels -> embeddings -> classify -> plot
```

To rerun specific dependent stages:

```bash
python scripts/run_experiment.py \
  --config configs/embedding_comparison_13200.yaml \
  --stages pseudolabels embeddings classify plot
```

Each stage can also run independently:

```bash
python scripts/select_subset.py --config configs/embedding_comparison_13200.yaml
python scripts/generate_pseudolabels.py --config configs/embedding_comparison_13200.yaml
python scripts/extract_embeddings.py --config configs/embedding_comparison_13200.yaml
python scripts/classify_mlp.py --config configs/embedding_comparison_13200.yaml
python scripts/plot_mlp_comparison.py --config configs/embedding_comparison_13200.yaml
```

## Generated Artifacts

Outputs are ignored by Git and written below:

```text
artifacts/embedding_comparison_13200/
  audio/
  metadata/
    selected_manifest.json
    selected_manifest.whisper_large_v3.json
    selected_manifest.whisper_large_v3.partial.jsonl
  embeddings/
    manifests/train.jsonl
    manifests/test.jsonl
    projections/
    <feature_set>/train.npy
    <feature_set>/test.npy
  results/classification/
    summary.csv
    <feature_set>/mlp/
    mlp_bar_accuracy_by_family_13200.png
    mlp_bar_accuracy_by_family_13200.csv
```

The default extractor writes projected and fused vectors only. Set `embeddings.save_raw_embeddings=true` to retain raw encoder matrices for audits or alternate projection experiments.

## Feature Sets

```text
sbert_gt
sbert_pseudo
wavlm_256
mfa_conformer_256
wavlm_mfa
sbert_gt_mfa
sbert_pseudo_mfa
sbert_gt_wavlm
sbert_pseudo_wavlm
sbert_gt_wavlm_mfa
sbert_pseudo_wavlm_mfa
```

Legacy GT aliases are intentionally not generated.
