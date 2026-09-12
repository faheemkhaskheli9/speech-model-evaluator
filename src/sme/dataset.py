"""Reference dataset assembly.

Ingests audio files (``.wav`` / ``.mp3``) plus matching transcript ``.txt``
files into a unified list of :class:`DatasetEntry` and writes an atomic
``manifest.json``. This is the consistent basis every later phase runs
providers against.

Design choices:
* WAV duration is read with the stdlib ``wave`` module (no dependency); MP3
  duration needs ``mutagen`` and raises a clear error if it is not installed
  rather than guessing.
* A missing or empty transcript is a hard error — a silent skip would quietly
  shrink the benchmark set and make two runs incomparable.
* The manifest is written to a temp file and ``os.replace``-d into place, so an
  interrupted run never leaves a half-written manifest (portfolio rule 1).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac"}


class DatasetError(Exception):
    pass


class AudioError(DatasetError):
    pass


class TranscriptError(DatasetError):
    pass


@dataclass(frozen=True)
class DatasetEntry:
    audio_path: str
    transcript: str
    language: str
    duration_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def probe_duration(path: Path) -> float:
    ext = path.suffix.lower()
    if ext == ".wav":
        try:
            with contextlib.closing(wave.open(str(path), "rb")) as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
        except (wave.Error, EOFError, OSError) as exc:
            raise AudioError(f"cannot read WAV {path}: {exc}") from exc
        if rate <= 0:
            raise AudioError(f"WAV {path} reports a non-positive sample rate")
        return round(frames / rate, 3)
    if ext == ".mp3":
        try:
            from mutagen.mp3 import MP3
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise AudioError(
                "MP3 duration requires the 'mutagen' package; install it or "
                "convert the clip to WAV"
            ) from exc
        try:
            audio = MP3(str(path))
        except Exception as exc:  # mutagen raises a variety of types
            raise AudioError(f"cannot read MP3 {path}: {exc}") from exc
        return round(float(audio.info.length), 3)
    if ext == ".flac":
        try:
            from mutagen.flac import FLAC
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise AudioError(
                "FLAC duration requires the 'mutagen' package; install it or "
                "convert the clip to WAV"
            ) from exc
        try:
            audio = FLAC(str(path))
        except Exception as exc:  # mutagen raises a variety of types
            raise AudioError(f"cannot read FLAC {path}: {exc}") from exc
        return round(float(audio.info.length), 3)
    raise AudioError(f"unsupported audio extension {ext!r} for {path}")


def _read_transcript(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise TranscriptError(f"cannot read transcript {path}: {exc}") from exc
    if not text:
        raise TranscriptError(f"transcript {path} is empty")
    return text


def assemble_dataset(
    audio_dir: str | os.PathLike[str],
    transcript_dir: str | os.PathLike[str] | None = None,
    *,
    language: str = "en",
) -> list[DatasetEntry]:
    """Pair each audio file under ``audio_dir`` with ``<stem>.txt``.

    ``transcript_dir`` defaults to ``audio_dir``. A user-supplied directory
    that does not exist is a hard error (strict on explicit input).
    """

    audio_root = Path(audio_dir)
    if not audio_root.is_dir():
        raise DatasetError(f"audio directory not found: {audio_root}")
    transcript_root = Path(transcript_dir) if transcript_dir is not None else audio_root
    if not transcript_root.is_dir():
        raise DatasetError(f"transcript directory not found: {transcript_root}")

    audio_files = sorted(
        p for p in audio_root.iterdir() if p.suffix.lower() in AUDIO_EXTS
    )
    if not audio_files:
        raise DatasetError(f"no .wav/.mp3 files under {audio_root}")

    entries: list[DatasetEntry] = []
    for audio_path in audio_files:
        transcript_path = transcript_root / f"{audio_path.stem}.txt"
        if not transcript_path.is_file():
            raise TranscriptError(
                f"no transcript for {audio_path.name} (expected {transcript_path})"
            )
        entries.append(
            DatasetEntry(
                audio_path=str(audio_path.resolve()),
                transcript=_read_transcript(transcript_path),
                language=language,
                duration_seconds=probe_duration(audio_path),
            )
        )
    return entries


def write_manifest(entries: list[DatasetEntry], out_dir: str | os.PathLike[str]) -> Path:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / "manifest.json"
    payload = {
        "count": len(entries),
        "total_duration_seconds": round(sum(e.duration_seconds for e in entries), 3),
        "entries": [e.to_dict() for e in entries],
    }
    data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")

    fd, tmp = tempfile.mkstemp(dir=out_root, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return target


def load_manifest(path: str | os.PathLike[str]) -> list[DatasetEntry]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DatasetEntry(**entry) for entry in raw["entries"]]
