# Environment and External Artifacts

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Whisper audio loading also requires `ffmpeg` on `PATH`.

Some Hugging Face datasets may require authentication:

```bash
export HF_TOKEN="<your-token>"
```

## Model Identifiers

```text
Whisper: openai/whisper-large-v3
SBERT:   all-mpnet-base-v2
WavLM:   microsoft/wavlm-base-plus
```

## MFA-Conformer

The MFA-Conformer implementation and checkpoint are external artifacts and are not committed.

Clone the tested source revision:

```bash
git clone https://github.com/zyzisyz/mfa_conformer mfa_conformer_repo
git -C mfa_conformer_repo checkout 1b9c229948f8dbdbe9370937813ec75d4b06b097
```

Place the checkpoint at:

```text
checkpoints/MFA_conformer.ckpt
```

Download the pretrained checkpoint from the [MFA-Conformer speaker-verification fork](https://github.com/ductuantruong/mfa_conformer_sv):

```text
https://entuedu-my.sharepoint.com/:u:/g/personal/truongdu001_e_ntu_edu_sg/EfeIgwS89qpGpp8oZFyDuHcBQh2w0NwH2cABV6uKvMwLdA?e=kNNe2E
```

Expected SHA-256:

```text
b40a1bdf78762808fc3f93069d95fc54b09740afe428e69b959b1652430485a2
```

Verify the downloaded file:

```bash
shasum -a 256 checkpoints/MFA_conformer.ckpt
```

The checkpoint source includes speaker-verification inference code. This experiment uses the tested source revision above and extracts the 3,072-dimensional penultimate pooled representation before projecting it to 256 dimensions.

## Validated Local Versions

```text
Python                3.9.6
torch                 2.8.0
torchaudio            2.8.0
transformers          4.57.6
sentence-transformers 5.1.2
datasets              4.5.0
scikit-learn          1.6.1
numpy                 2.0.2
pandas                2.3.3
scipy                 1.13.1
soundfile             0.13.1
joblib                1.5.3
matplotlib            3.9.4
```
