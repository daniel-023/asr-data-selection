#!/usr/bin/env python3
"""Tests for extract_sbert_embeddings.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_sbert_embeddings import (
    EXPECTED_DOMAINS,
    SBERTAdapter,
    load_manifest,
    resolve_device,
    run_pipeline,
    save_embedding,
    save_jsonl,
    validate_embedding,
)

# ── helpers ────────────────────────────────────────────────────────────────

DOMAINS = sorted(EXPECTED_DOMAINS)


def _rows(split_counts: dict, pseudo: bool = True) -> List[dict]:
    rows = []
    for domain in DOMAINS:
        for split, n in split_counts.items():
            for i in range(n):
                row = dict(
                    utt_id=f"{domain}_{split}_{i}",
                    orig_id=f"orig_{i}",
                    domain=domain,
                    split=split,
                    audio_path=f"audio/{domain}/{i}.wav",
                    transcript=f"raw {i}",
                    transcript_norm=f"norm {i}",
                    duration_sec=1.5,
                )
                if pseudo:
                    row["pseudo_transcript_norm"] = f"pseudo {i}"
                rows.append(row)
    return rows


def _manifest(path: Path, rows: List[dict]) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _args(manifest: Path, out: Path, sources: List[str], seed: int = 0) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=manifest,
        out_root=out,
        device="cpu",
        sbert_model="all-mpnet-base-v2",
        sbert_batch_size=64,
        semantic_sources=sources,
        seed=seed,
    )


def _mock_enc(dim: int = 768) -> MagicMock:
    rng = np.random.RandomState(99)
    enc = MagicMock()
    enc.encode.side_effect = lambda texts: rng.randn(len(texts), dim).astype(np.float32)
    return enc


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def legacy(tmp_path):
    return _manifest(tmp_path / "m.json", _rows({"train": 300, "test": 700}))


@pytest.fixture
def legacy_no_pseudo(tmp_path):
    return _manifest(tmp_path / "m.json", _rows({"train": 300, "test": 700}, pseudo=False))


@pytest.fixture
def refcand(tmp_path):
    return _manifest(tmp_path / "m.json", _rows({"reference": 300, "candidate": 3000}))


# ── load_manifest ──────────────────────────────────────────────────────────

class TestLoadManifest:

    def test_valid_legacy(self, legacy):
        df, plan = load_manifest(legacy)
        assert len(df) == 4000
        assert set(plan["manifest_splits"]) == {"train", "test"}
        assert plan["output_splits"] == {"train": "train", "test": "test"}

    def test_valid_refcand(self, refcand):
        df, plan = load_manifest(refcand)
        assert len(df) == 13200
        assert plan["output_splits"] == {"reference": "train", "candidate": "test"}

    def test_per_domain_split_counts(self, legacy):
        df, _ = load_manifest(legacy)
        counts = df.groupby(["domain", "split"]).size()
        for d in DOMAINS:
            assert counts[(d, "train")] == 300
            assert counts[(d, "test")] == 700

    def test_not_a_list_raises(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text('{"a":1}')
        with pytest.raises(ValueError, match="list of rows"):
            load_manifest(p)

    def test_missing_column_raises(self, tmp_path, legacy):
        rows = json.loads(legacy.read_text())
        for r in rows:
            del r["transcript_norm"]
        p = _manifest(tmp_path / "bad.json", rows)
        with pytest.raises(ValueError, match="missing required columns"):
            load_manifest(p)

    def test_extra_domain_raises(self, tmp_path, legacy):
        rows = json.loads(legacy.read_text())
        rows[0]["domain"] = "unknown_corpus"
        p = _manifest(tmp_path / "bad.json", rows)
        with pytest.raises(ValueError, match="Expected domains"):
            load_manifest(p)

    def test_missing_domain_raises(self, tmp_path, legacy):
        rows = [r for r in json.loads(legacy.read_text()) if r["domain"] != "gigaspeech"]
        p = _manifest(tmp_path / "bad.json", rows)
        with pytest.raises(ValueError, match="Expected domains"):
            load_manifest(p)

    def test_unsupported_split_names_raises(self, tmp_path, legacy):
        rows = json.loads(legacy.read_text())
        for r in rows:
            if r["split"] == "test":
                r["split"] = "val"
        p = _manifest(tmp_path / "bad.json", rows)
        with pytest.raises(ValueError, match="Unsupported manifest splits"):
            load_manifest(p)

    def test_wrong_total_count_raises(self, tmp_path, legacy):
        rows = json.loads(legacy.read_text())
        rows.append(rows[0].copy())
        p = _manifest(tmp_path / "bad.json", rows)
        with pytest.raises(ValueError, match="Expected 4000 rows"):
            load_manifest(p)

    def test_wrong_per_domain_count_raises(self, tmp_path, legacy):
        rows = json.loads(legacy.read_text())
        # Move one gigaspeech/train row to test — keeps total at 4000
        for r in rows:
            if r["domain"] == "gigaspeech" and r["split"] == "train":
                r["split"] = "test"
                break
        p = _manifest(tmp_path / "bad.json", rows)
        with pytest.raises(ValueError, match="gigaspeech/train"):
            load_manifest(p)


# ── validate_embedding ─────────────────────────────────────────────────────

class TestValidateEmbedding:

    def test_valid_float32_passthrough(self):
        emb = np.ones((10, 64), dtype=np.float32)
        out = validate_embedding("f", "train", emb, expected_rows=10)
        assert out.shape == (10, 64) and out.dtype == np.float32

    def test_float64_cast_to_float32(self):
        out = validate_embedding("f", "train", np.ones((5, 32), dtype=np.float64), expected_rows=5)
        assert out.dtype == np.float32

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            validate_embedding("f", "train", np.ones(10), expected_rows=10)

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            validate_embedding("f", "train", np.ones((2, 5, 3)), expected_rows=2)

    def test_wrong_row_count_raises(self):
        with pytest.raises(ValueError, match="row mismatch"):
            validate_embedding("f", "train", np.ones((10, 8), dtype=np.float32), expected_rows=5)

    def test_wrong_dim_raises(self):
        with pytest.raises(ValueError, match="dim mismatch"):
            validate_embedding("f", "train", np.ones((10, 64), dtype=np.float32), expected_rows=10, expected_dim=128)

    def test_none_expected_dim_skips_check(self):
        out = validate_embedding("f", "train", np.ones((10, 999), dtype=np.float32), expected_rows=10)
        assert out.shape[1] == 999

    def test_nan_raises(self):
        emb = np.ones((5, 8), dtype=np.float32)
        emb[2, 3] = np.nan
        with pytest.raises(ValueError, match="NaN/Inf"):
            validate_embedding("f", "train", emb, expected_rows=5)

    def test_pos_inf_raises(self):
        emb = np.ones((5, 8), dtype=np.float32)
        emb[0, 0] = np.inf
        with pytest.raises(ValueError, match="NaN/Inf"):
            validate_embedding("f", "train", emb, expected_rows=5)

    def test_neg_inf_raises(self):
        emb = np.ones((5, 8), dtype=np.float32)
        emb[0, 0] = -np.inf
        with pytest.raises(ValueError, match="NaN/Inf"):
            validate_embedding("f", "train", emb, expected_rows=5)


# ── save_jsonl ─────────────────────────────────────────────────────────────

class TestSaveJsonl:

    def test_one_line_per_record(self, tmp_path):
        save_jsonl([{"a": 1}, {"b": 2}], tmp_path / "out.jsonl")
        lines = (tmp_path / "out.jsonl").read_text().splitlines()
        assert len(lines) == 2

    def test_round_trip_values(self, tmp_path):
        records = [{"x": 42, "y": "hello"}, {"x": -1}]
        save_jsonl(records, tmp_path / "out.jsonl")
        loaded = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]
        assert loaded == records

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "out.jsonl"
        save_jsonl([{"k": 1}], out)
        assert out.exists()

    def test_unicode_preserved(self, tmp_path):
        save_jsonl([{"t": "日本語"}], tmp_path / "out.jsonl")
        assert json.loads((tmp_path / "out.jsonl").read_text(encoding="utf-8"))["t"] == "日本語"

    def test_empty_list_writes_empty_file(self, tmp_path):
        out = tmp_path / "out.jsonl"
        save_jsonl([], out)
        assert out.read_text() == ""


# ── save_embedding ─────────────────────────────────────────────────────────

class TestSaveEmbedding:

    def test_shape_dtype_values(self, tmp_path):
        emb = np.arange(20, dtype=np.float32).reshape(4, 5)
        save_embedding(tmp_path, "feat", "train", emb)
        loaded = np.load(tmp_path / "feat" / "train.npy")
        assert loaded.shape == (4, 5)
        assert loaded.dtype == np.float32
        np.testing.assert_array_equal(loaded, emb)

    def test_float64_saved_as_float32(self, tmp_path):
        save_embedding(tmp_path, "f", "train", np.ones((3, 8), dtype=np.float64))
        assert np.load(tmp_path / "f" / "train.npy").dtype == np.float32

    def test_creates_subdirectory(self, tmp_path):
        save_embedding(tmp_path, "new_feat", "test", np.zeros((2, 4), dtype=np.float32))
        assert (tmp_path / "new_feat" / "test.npy").exists()

    def test_returns_output_path(self, tmp_path):
        result = save_embedding(tmp_path, "feat", "train", np.zeros((2, 4), dtype=np.float32))
        assert result == tmp_path / "feat" / "train.npy"


# ── resolve_device ─────────────────────────────────────────────────────────

class TestResolveDevice:

    def test_cpu_passthrough(self):
        assert resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self):
        assert resolve_device("cuda") == "cuda"

    def test_auto_no_torch_returns_cpu(self):
        with patch.dict("sys.modules", {"torch": None}):
            assert resolve_device("auto") == "cpu"

    def test_auto_no_cuda_returns_cpu(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert resolve_device("auto") == "cpu"

    def test_auto_with_cuda_returns_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert resolve_device("auto") == "cuda"


# ── SBERTAdapter ───────────────────────────────────────────────────────────

class TestSBERTAdapter:

    def _patch_st(self, return_value=None):
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model
        if return_value is not None:
            mock_model.encode.return_value = return_value
        return mock_st, mock_model

    def test_missing_library_raises(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(RuntimeError, match="sentence-transformers"):
                SBERTAdapter("model", "cpu", 32)

    def test_encode_casts_to_float32(self):
        mock_st, _ = self._patch_st(return_value=np.ones((3, 768), dtype=np.float64))
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            adapter = SBERTAdapter("model", "cpu", 32)
            result = adapter.encode(["a", "b", "c"])
        assert result.dtype == np.float32

    def test_none_inputs_replaced_with_empty_string(self):
        mock_st, mock_model = self._patch_st(return_value=np.zeros((3, 8), dtype=np.float32))
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            adapter = SBERTAdapter("model", "cpu", 32)
            adapter.encode([None, "hello", None])
        passed = mock_model.encode.call_args[0][0]
        assert passed == ["", "hello", ""]


# ── run_pipeline (integration) ─────────────────────────────────────────────

class TestRunPipeline:

    @patch("extract_sbert_embeddings.SBERTAdapter")
    def test_gt_only_output_files(self, MockSBERT, tmp_path, legacy_no_pseudo):
        MockSBERT.return_value = _mock_enc()
        out = tmp_path / "out"
        run_pipeline(_args(legacy_no_pseudo, out, ["gt"]))

        for name in ("all", "train", "test"):
            assert (out / "manifests" / f"{name}.jsonl").exists()

        assert np.load(out / "sbert_gt_raw" / "train.npy").shape == (1200, 768)
        assert np.load(out / "sbert_gt_raw" / "test.npy").shape == (2800, 768)

        # sbert_raw is an alias for sbert_gt_raw
        np.testing.assert_array_equal(
            np.load(out / "sbert_raw" / "train.npy"),
            np.load(out / "sbert_gt_raw" / "train.npy"),
        )

        assert np.load(out / "sbert_gt" / "train.npy").shape == (1200, 256)
        assert np.load(out / "sbert_gt" / "test.npy").shape == (2800, 256)

        # sbert is an alias for sbert_gt
        np.testing.assert_array_equal(
            np.load(out / "sbert" / "train.npy"),
            np.load(out / "sbert_gt" / "train.npy"),
        )

        assert (out / "projections" / "sbert_gt_grp_768_to_256.joblib").exists()
        assert (out / "projections" / "sbert_grp_768_to_256.joblib").exists()

        assert not (out / "sbert_pseudo_raw").exists()
        assert not (out / "sbert_pseudo").exists()

    @patch("extract_sbert_embeddings.SBERTAdapter")
    def test_gt_pseudo_output_files(self, MockSBERT, tmp_path, legacy):
        MockSBERT.return_value = _mock_enc()
        out = tmp_path / "out"
        run_pipeline(_args(legacy, out, ["gt", "pseudo"]))

        for split, n in [("train", 1200), ("test", 2800)]:
            assert np.load(out / "sbert_gt_raw" / f"{split}.npy").shape == (n, 768)
            assert np.load(out / "sbert_pseudo_raw" / f"{split}.npy").shape == (n, 768)
            assert np.load(out / "sbert_gt" / f"{split}.npy").shape == (n, 256)
            assert np.load(out / "sbert_pseudo" / f"{split}.npy").shape == (n, 256)

        assert (out / "projections" / "sbert_gt_pseudo_grp_768_to_256.joblib").exists()
        assert (out / "projections" / "sbert_grp_768_to_256.joblib").exists()

    @patch("extract_sbert_embeddings.SBERTAdapter")
    def test_refcand_split_sizes(self, MockSBERT, tmp_path, refcand):
        MockSBERT.return_value = _mock_enc()
        out = tmp_path / "out"
        run_pipeline(_args(refcand, out, ["gt"]))

        # reference→train (1200), candidate→test (12000)
        assert np.load(out / "sbert_gt_raw" / "train.npy").shape == (1200, 768)
        assert np.load(out / "sbert_gt_raw" / "test.npy").shape == (12000, 768)

    @patch("extract_sbert_embeddings.SBERTAdapter")
    def test_encode_call_count_and_sizes(self, MockSBERT, tmp_path, legacy_no_pseudo):
        enc = _mock_enc()
        MockSBERT.return_value = enc
        run_pipeline(_args(legacy_no_pseudo, tmp_path / "out", ["gt"]))

        assert enc.encode.call_count == 2
        sizes = [len(call[0][0]) for call in enc.encode.call_args_list]
        assert sizes == [1200, 2800]  # train then test, per OUTPUT_SPLIT_ORDER

    @patch("extract_sbert_embeddings.SBERTAdapter")
    def test_projected_dim_is_256(self, MockSBERT, tmp_path, legacy_no_pseudo):
        MockSBERT.return_value = _mock_enc()
        out = tmp_path / "out"
        run_pipeline(_args(legacy_no_pseudo, out, ["gt"]))
        assert np.load(out / "sbert_gt" / "train.npy").shape[1] == 256
        assert np.load(out / "sbert_gt" / "test.npy").shape[1] == 256

    @patch("extract_sbert_embeddings.SBERTAdapter")
    def test_all_outputs_float32_and_finite(self, MockSBERT, tmp_path, legacy_no_pseudo):
        MockSBERT.return_value = _mock_enc()
        out = tmp_path / "out"
        run_pipeline(_args(legacy_no_pseudo, out, ["gt"]))

        for fname in [
            "sbert_gt_raw/train.npy", "sbert_gt_raw/test.npy",
            "sbert_gt/train.npy", "sbert_gt/test.npy",
        ]:
            arr = np.load(out / fname)
            assert arr.dtype == np.float32, f"{fname} not float32"
            assert np.isfinite(arr).all(), f"{fname} contains non-finite values"

    @patch("extract_sbert_embeddings.SBERTAdapter")
    def test_manifest_row_counts(self, MockSBERT, tmp_path, legacy_no_pseudo):
        MockSBERT.return_value = _mock_enc()
        out = tmp_path / "out"
        run_pipeline(_args(legacy_no_pseudo, out, ["gt"]))

        def nlines(p): return len(p.read_text().strip().splitlines())
        assert nlines(out / "manifests" / "all.jsonl") == 4000
        assert nlines(out / "manifests" / "train.jsonl") == 1200
        assert nlines(out / "manifests" / "test.jsonl") == 2800

    @patch("extract_sbert_embeddings.SBERTAdapter")
    def test_seed_reproducibility(self, MockSBERT, tmp_path, legacy_no_pseudo):
        fixed = {
            1200: np.random.RandomState(7).randn(1200, 768).astype(np.float32),
            2800: np.random.RandomState(8).randn(2800, 768).astype(np.float32),
        }
        enc = MagicMock()
        enc.encode.side_effect = lambda texts: fixed[len(texts)]
        MockSBERT.return_value = enc

        for i in range(2):
            run_pipeline(_args(legacy_no_pseudo, tmp_path / f"r{i}", ["gt"], seed=42))

        np.testing.assert_array_equal(
            np.load(tmp_path / "r0" / "sbert_gt" / "train.npy"),
            np.load(tmp_path / "r1" / "sbert_gt" / "train.npy"),
        )

    def test_missing_pseudo_column_raises(self, tmp_path, legacy_no_pseudo):
        with patch("extract_sbert_embeddings.SBERTAdapter"):
            with pytest.raises(ValueError, match="pseudo_transcript_norm"):
                run_pipeline(_args(legacy_no_pseudo, tmp_path / "out", ["pseudo"]))

    def test_empty_pseudo_text_raises(self, tmp_path):
        rows = _rows({"train": 300, "test": 700}, pseudo=True)
        rows[0]["pseudo_transcript_norm"] = "   "  # whitespace-only
        p = _manifest(tmp_path / "m.json", rows)
        with patch("extract_sbert_embeddings.SBERTAdapter"):
            with pytest.raises(ValueError, match="empty texts"):
                run_pipeline(_args(p, tmp_path / "out", ["pseudo"]))

    @patch("extract_sbert_embeddings.SBERTAdapter")
    def test_failed_pseudo_status_skips_empty_check(self, MockSBERT, tmp_path):
        rows = _rows({"train": 300, "test": 700}, pseudo=True)
        rows[0]["pseudo_transcript_norm"] = ""
        rows[0]["pseudo_status"] = "failed"
        p = _manifest(tmp_path / "m.json", rows)
        MockSBERT.return_value = _mock_enc()
        # should not raise despite one empty pseudo text
        run_pipeline(_args(p, tmp_path / "out", ["gt", "pseudo"]))
