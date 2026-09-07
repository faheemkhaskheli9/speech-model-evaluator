import json

import pytest

from sme.dataset import (
    AudioError,
    DatasetError,
    TranscriptError,
    assemble_dataset,
    load_manifest,
    probe_duration,
    write_manifest,
)


def test_assemble_happy_path(sample_dataset_dir):
    entries = assemble_dataset(sample_dataset_dir, language="en")
    assert [e.transcript for e in entries] == ["hello world", "the quick brown fox"]
    assert all(e.language == "en" for e in entries)
    assert entries[0].duration_seconds == pytest.approx(0.5, abs=0.01)
    assert entries[1].duration_seconds == pytest.approx(1.0, abs=0.01)
    assert entries[0].audio_path.endswith("clip1.wav")


def test_manifest_roundtrip(sample_dataset_dir, tmp_path):
    entries = assemble_dataset(sample_dataset_dir)
    out = write_manifest(entries, tmp_path / "ds")
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["count"] == 2
    assert data["total_duration_seconds"] == pytest.approx(1.5, abs=0.02)
    assert [e.transcript for e in load_manifest(out)] == [
        "hello world",
        "the quick brown fox",
    ]
    assert not list((tmp_path / "ds").glob("*.tmp"))


def test_missing_transcript_is_hard_error(sample_dataset_dir):
    (sample_dataset_dir / "clip2.txt").unlink()
    with pytest.raises(TranscriptError):
        assemble_dataset(sample_dataset_dir)


def test_empty_transcript_rejected(sample_dataset_dir):
    (sample_dataset_dir / "clip1.txt").write_text("   \n", encoding="utf-8")
    with pytest.raises(TranscriptError):
        assemble_dataset(sample_dataset_dir)


def test_unsupported_audio_extension(tmp_path):
    (tmp_path / "a.flac").write_bytes(b"not really flac")
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    with pytest.raises(DatasetError):  # no .wav/.mp3 -> DatasetError
        assemble_dataset(tmp_path)


def test_corrupt_wav_raises_audio_error(tmp_path):
    (tmp_path / "bad.wav").write_bytes(b"RIFFxxxxWAVEjunk")
    (tmp_path / "bad.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(AudioError):
        assemble_dataset(tmp_path)


def test_missing_audio_dir_is_hard_error(tmp_path):
    with pytest.raises(DatasetError):
        assemble_dataset(tmp_path / "nope")


def test_probe_duration_unsupported_ext(tmp_path):
    p = tmp_path / "x.ogg"
    p.write_bytes(b"junk")
    with pytest.raises(AudioError):
        probe_duration(p)
