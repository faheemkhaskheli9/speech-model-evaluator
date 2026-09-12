import tarfile

import pytest

from sme.librispeech import (
    LibriSpeechConfig,
    LibriSpeechError,
    extract,
    extract_and_convert,
    load_config,
)


def _make_librispeech_tree(root, make_wav):
    """Build a tiny tree with LibriSpeech's real layout:
    <speaker>/<chapter>/<speaker>-<chapter>-<utt>.wav (real layout uses
    .flac; .wav here avoids a mutagen/real-FLAC dependency in tests, and
    extract_and_convert accepts either) plus one .trans.txt per chapter.
    """
    chapter_dir = root / "19" / "198"
    chapter_dir.mkdir(parents=True)
    make_wav(chapter_dir / "19-198-0000.wav", seconds=0.3)
    make_wav(chapter_dir / "19-198-0001.wav", seconds=0.6)
    (chapter_dir / "19-198.trans.txt").write_text(
        "19-198-0000 THIS IS THE FIRST UTTERANCE\n"
        "19-198-0001 THIS IS THE SECOND UTTERANCE\n",
        encoding="utf-8",
    )
    return chapter_dir


# --- config ---------------------------------------------------------------


def test_load_config_defaults_when_no_path():
    cfg = load_config()
    assert cfg.url.endswith("dev-clean.tar.gz")
    assert cfg.language == "en"
    assert cfg.max_utterances is None


def test_load_config_explicit_missing_path_is_hard_error():
    with pytest.raises(FileNotFoundError):
        load_config("nope/missing.yaml")


def test_load_config_reads_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("language: fr\nmax_utterances: 5\ncache_dir: /tmp/x\n")
    cfg = load_config(p)
    assert cfg.language == "fr"
    assert cfg.max_utterances == 5
    assert cfg.cache_dir == "/tmp/x"


def test_non_positive_max_utterances_rejected():
    with pytest.raises(LibriSpeechError):
        LibriSpeechConfig(max_utterances=0)


# --- extract_and_convert ---------------------------------------------------


def test_extract_and_convert_parses_transcripts_and_probes_duration(tmp_path, make_wav):
    _make_librispeech_tree(tmp_path, make_wav)
    entries = extract_and_convert(tmp_path)

    assert len(entries) == 2
    assert entries[0].transcript == "THIS IS THE FIRST UTTERANCE"
    assert entries[0].language == "en"
    assert entries[0].duration_seconds == pytest.approx(0.3, abs=0.05)
    assert entries[1].transcript == "THIS IS THE SECOND UTTERANCE"


def test_extract_and_convert_respects_max_utterances(tmp_path, make_wav):
    _make_librispeech_tree(tmp_path, make_wav)
    entries = extract_and_convert(tmp_path, max_utterances=1)
    assert len(entries) == 1


def test_extract_and_convert_custom_language(tmp_path, make_wav):
    _make_librispeech_tree(tmp_path, make_wav)
    entries = extract_and_convert(tmp_path, language="fr")
    assert all(e.language == "fr" for e in entries)


def test_extract_and_convert_missing_directory_raises(tmp_path):
    with pytest.raises(LibriSpeechError):
        extract_and_convert(tmp_path / "does-not-exist")


def test_extract_and_convert_no_transcripts_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(LibriSpeechError):
        extract_and_convert(tmp_path / "empty")


def test_extract_and_convert_missing_audio_for_utterance_raises(tmp_path, make_wav):
    chapter_dir = _make_librispeech_tree(tmp_path, make_wav)
    (chapter_dir / "19-198-0000.wav").unlink()
    with pytest.raises(LibriSpeechError):
        extract_and_convert(tmp_path)


# --- extract (tarball, not a real network fetch) --------------------------


def test_extract_unpacks_tarball_into_dest(tmp_path, make_wav):
    src = tmp_path / "src"
    _make_librispeech_tree(src, make_wav)
    tarball = tmp_path / "dev-clean.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(src, arcname="LibriSpeech/dev-clean")

    dest = tmp_path / "extracted"
    extract(tarball, dest)

    entries = extract_and_convert(dest)
    assert len(entries) == 2


def test_extract_is_idempotent_via_completion_marker(tmp_path, make_wav):
    src = tmp_path / "src"
    _make_librispeech_tree(src, make_wav)
    tarball = tmp_path / "dev-clean.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(src, arcname="LibriSpeech/dev-clean")

    dest = tmp_path / "extracted"
    extract(tarball, dest)
    marker = dest / ".extracted.ok"
    assert marker.is_file()

    # Simulate a partial/interrupted second extraction attempt: delete the
    # tarball so a real re-extract would fail, then confirm the marker
    # short-circuits it instead of trying.
    tarball.unlink()
    extract(tarball, dest)  # should not raise: marker says already done
