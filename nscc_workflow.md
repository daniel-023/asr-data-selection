# NSCC Workflow Runbook

This runbook collects the commands for syncing the embedding analysis pipeline to NSCC, pulling outputs back to OneDrive, and submitting the PBS jobs.

## Current PBS Defaults

| PBS                         |                     Job | Resources                             |     Walltime | Key Defaults                                                                                                                                                                                     |
| --------------------------- | ----------------------: | ------------------------------------- | -----------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `extract_subset.pbs`      |      `extract_subset` | `select=1:ncpus=4:mem=32gb`         |  `4:00:00` | `WORKDIR=${HOME}/scratch/embedding_analysis`, `OUTPUT_ROOT=${WORKDIR}`, env via `micromamba activate speech_lab`            |
| `generate_pseudolabels.pbs` | `generate_pseudolabels` | `select=1:ncpus=4:mem=64gb:ngpus=1` | `8:00:00` | `MANIFEST=${WORKDIR}/metadata/selected_manifest.json`, `OUT_JSON=${WORKDIR}/metadata/selected_manifest.whisper_large_v3.json`, `WHISPER_MODEL="openai/whisper-large-v3"`, `WHISPER_LANGUAGE="english"`, `CHUNK_LENGTH_S=30`, `STRIDE_LENGTH_S=5` |
| `extract_embeddings.pbs`  |  `extract_embeddings` | `select=1:ncpus=4:mem=64gb:ngpus=1` | `12:00:00` | `OUT_ROOT=${WORKDIR}/embeddings`, `MANIFEST=${WORKDIR}/metadata/selected_manifest.whisper_large_v3.json`, `ENCODERS="wavlm mfa_conformer sbert"`, `SEMANTIC_SOURCES="gt pseudo"`                 |
| `cluster_embeddings.pbs`  |  `cluster_embeddings` | `select=1:ncpus=4:mem=32gb`         |  `2:00:00` | `OUT=${WORKDIR}/embeddings`, `FEATURE_SETS="sbert wavlm_mfa sbert_mfa sbert_wavlm sbert_wavlm_mfa"`                                                                                          |
| `classify_embeddings.pbs` | `classify_embeddings` | `select=1:ncpus=4:mem=32gb`         |  `4:00:00` | `OUT=${WORKDIR}/embeddings`, expanded GT/pseudo pooled `FEATURE_SETS`, `MODELS="mlp"` |

## Pooled Experiment Configs

Default embedding extraction with `ENCODERS="wavlm mfa_conformer sbert"` and `SEMANTIC_SOURCES="gt pseudo"` generates these pooled feature sets:

| Experiment config | Feature set         | Components                                | Expected dim |
| ----------------- | ------------------- | ----------------------------------------- | -----------: |
| Semantic GT | `sbert_gt` | SBERT on `transcript_norm` | 256 |
| Semantic Pseudo | `sbert_pseudo` | SBERT on `pseudo_transcript_norm` | 256 |
| WavLM | `wavlm_256` | projected WavLM | 256 |
| MFA | `mfa_conformer_256` | projected MFA-Conformer | 256 |
| WavLM + MFA | `wavlm_mfa` | `wavlm_256 + mfa_conformer_256` | 512 |
| Semantic GT + MFA | `sbert_gt_mfa` | `sbert_gt + mfa_conformer_256` | 512 |
| Semantic Pseudo + MFA | `sbert_pseudo_mfa` | `sbert_pseudo + mfa_conformer_256` | 512 |
| Semantic GT + WavLM | `sbert_gt_wavlm` | `sbert_gt + wavlm_256` | 512 |
| Semantic Pseudo + WavLM | `sbert_pseudo_wavlm` | `sbert_pseudo + wavlm_256` | 512 |
| Full Fusion GT | `sbert_gt_wavlm_mfa` | `sbert_gt + wavlm_256 + mfa_conformer_256` | 768 |
| Full Fusion Pseudo | `sbert_pseudo_wavlm_mfa` | `sbert_pseudo + wavlm_256 + mfa_conformer_256` | 768 |

## Reference/Candidate Subset

`extract_subset.py` now generates 13,200 rows:

```text
300 reference samples per domain   = 1,200 reference rows
3000 candidate samples per domain  = 12,000 candidate rows
```

The split labels are role-based:

```text
reference = selected from the Hugging Face test/eval split
candidate = selected from the Hugging Face train split
```

Svarah only has a `test` split, so both Svarah roles are sampled from `test` with no selected-sample overlap.

The current global quality filter is:

```text
min_words = 3
min_duration_sec = 3.0
```

Subset outputs:

```text
audio/data_selection_13200/<domain>_reference/
audio/data_selection_13200/<domain>_candidate/
metadata/selected_manifest.json
metadata/selected_manifest.jsonl
metadata/reference.jsonl
metadata/candidate.jsonl
```

## Shared Rsync Variables

Run these in your local terminal before the rsync commands below.

```bash
JUMP="<jump-user>@<jump-host>"
REMOTE="<nscc-user>@aspire2antu.nscc.sg"
REMOTE_ROOT="scratch/embedding_analysis"
REMOTE_AUDIO_SELECTION="${REMOTE_ROOT}/audio/data_selection_13200"
LOCAL_CODE="/path/to/asr-data-selection"
LOCAL_EXP="/path/to/llm_domain_experiment"
SSH_CMD="ssh -J ${JUMP}"
```

Replace the placeholder values before running the commands.

## Rsync To NSCC

Create the expected remote directories:

```bash
ssh -J "${JUMP}" "${REMOTE}" \
  "mkdir -p '${REMOTE_ROOT}' '${REMOTE_AUDIO_SELECTION}' '${REMOTE_ROOT}/metadata' '${REMOTE_ROOT}/checkpoints'"
```

Push the main pipeline scripts and PBS files:

```bash
rsync -avP -e "${SSH_CMD}" \
  "${LOCAL_CODE}/extract_embeddings.py" \
  "${LOCAL_CODE}/extract_embeddings.pbs" \
  "${LOCAL_CODE}/generate_pseudolabels.py" \
  "${LOCAL_CODE}/generate_pseudolabels.pbs" \
  "${LOCAL_CODE}/cluster_embeddings.py" \
  "${LOCAL_CODE}/cluster_embeddings.pbs" \
  "${LOCAL_CODE}/classify_embeddings.py" \
  "${LOCAL_CODE}/classify_embeddings.pbs" \
  "${LOCAL_CODE}/plot_mlp_comparison.py" \
  "${REMOTE}:${REMOTE_ROOT}/"
```

Push optional subset scripts:

```bash
rsync -avP -e "${SSH_CMD}" \
  "${LOCAL_CODE}/extract_subset.py" \
  "${LOCAL_CODE}/extract_subset.pbs" \
  "${REMOTE}:${REMOTE_ROOT}/"
```

Push local experiment inputs:

```bash
rsync -avP -e "${SSH_CMD}" \
  "${LOCAL_EXP}/audio/data_selection_13200/" \
  "${REMOTE}:${REMOTE_AUDIO_SELECTION}/"

rsync -avP -e "${SSH_CMD}" \
  "${LOCAL_EXP}/metadata/" \
  "${REMOTE}:${REMOTE_ROOT}/metadata/"
```

Push MFA-Conformer code and checkpoint:

```bash
rsync -avP -e "${SSH_CMD}" \
  "${LOCAL_CODE}/mfa_conformer_repo/" \
  "${REMOTE}:${REMOTE_ROOT}/mfa_conformer_repo/"

rsync -avP -e "${SSH_CMD}" \
  "${LOCAL_CODE}/checkpoints/MFA_conformer.ckpt" \
  "${REMOTE}:${REMOTE_ROOT}/checkpoints/MFA_conformer.ckpt"
```

## Rsync From NSCC

Pull the full embeddings output:

```bash
rsync -avP -e "${SSH_CMD}" \
  "${REMOTE}:${REMOTE_ROOT}/embeddings/" \
  "${LOCAL_EXP}/embeddings/"
```

Pull results only:

```bash
rsync -avP -e "${SSH_CMD}" \
  "${REMOTE}:${REMOTE_ROOT}/embeddings/results/" \
  "${LOCAL_EXP}/results/"
```

Pull frame embeddings only:

```bash
rsync -avP -e "${SSH_CMD}" \
  "${REMOTE}:${REMOTE_ROOT}/embeddings/frame_embeddings/" \
  "${LOCAL_EXP}/embeddings/frame_embeddings/"
```

## Qsub Commands

Run these from the NSCC working directory:

```bash
PBS_PROJECT="<pbs-project-code>"
cd "${HOME}/scratch/embedding_analysis"
```

## Current Experiment Run Order

Use this order for the GT vs Whisper pseudolabel MLP comparison. Wait for each submitted job to finish before starting the next dependent step.

Verify synced inputs:

```bash
ls metadata/selected_manifest.json
ls audio/data_selection_13200 | head
ls checkpoints/MFA_conformer.ckpt
```

Generate Whisper large-v3 pseudolabels:

```bash
qsub -P "${PBS_PROJECT}" generate_pseudolabels.pbs
```

After the pseudolabel job finishes, verify the Whisper manifest exists:

```bash
ls metadata/selected_manifest.whisper_large_v3.json
```

Default embedding extraction from the pseudolabel manifest:

```bash
qsub -P "${PBS_PROJECT}" extract_embeddings.pbs
```

This maps `reference` rows to downstream `train.npy` files and `candidate` rows to downstream `test.npy` files.

After the embedding job finishes, verify representative outputs:

```bash
ls embeddings/sbert_gt/train.npy embeddings/sbert_pseudo/test.npy
ls embeddings/wavlm_256/train.npy embeddings/mfa_conformer_256/test.npy
```

This generates the expanded pooled feature sets when WavLM, MFA-Conformer, GT SBERT, and pseudo SBERT all complete:

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

Embedding encoder choices:

```text
wavlm                  pooled WavLM raw + wavlm_256
mfa_conformer          pooled MFA-Conformer raw + mfa_conformer_256
sbert                  SBERT text embeddings projected to 256-d
wavlm_frame            WavLM frame files only
mfa_conformer_frame    MFA-Conformer frame files only
```

Classification:

```bash
qsub -P "${PBS_PROJECT}" classify_embeddings.pbs
```

This trains MLP only by default on the expanded pooled feature sets.

After classification finishes, verify the summary exists:

```bash
ls embeddings/results/classification/summary.csv
```

Plot the accuracy-only MLP comparison charts after classification:

```bash
"${PYTHON:-${HOME}/scratch/micromamba/envs/speech_lab/bin/python}" \
  plot_mlp_comparison.py --out-root "${HOME}/scratch/embedding_analysis/embeddings"
```

The plot step writes:

```text
embeddings/results/classification/mlp_bar_accuracy_by_family_13200.png
embeddings/results/classification/mlp_bar_accuracy_by_family_13200.csv
embeddings/results/classification/mlp_accuracy_key_findings.csv
embeddings/results/classification/current_mlp_results_manifest.txt
```

The manifest lists current 13,200-row MLP outputs to share and legacy/non-headline result folders to exclude unless explicitly discussed.

## Optional Commands

Frame-only WavLM and MFA extraction:

```bash
qsub -P "${PBS_PROJECT}" -v ENCODERS="wavlm_frame mfa_conformer_frame" extract_embeddings.pbs
```

Acoustic-only embedding extraction with frame-level outputs:

```bash
qsub -P "${PBS_PROJECT}" -v ENCODERS="wavlm mfa_conformer wavlm_frame mfa_conformer_frame" extract_embeddings.pbs
```

Clustering:

```bash
qsub -P "${PBS_PROJECT}" cluster_embeddings.pbs
```

Frame-level CNN only:

```bash
qsub -P "${PBS_PROJECT}" -v MODELS="cnn",FEATURE_SETS="wavlm_frame mfa_conformer_frame" classify_embeddings.pbs
```

## Job Status And Logs

```bash
qstat -u "${USER}"
qdel <job_id>
ls -lh "${HOME}/scratch/embedding_analysis"/*.o*
```
