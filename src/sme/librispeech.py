"""Load a public STT benchmark dataset (LibriSpeech) into the project's
manifest format.

LibriSpeech ships as a per-split ``.tar.gz`` (OpenSLR resource 12); each
archive extracts to ``LibriSpeech/<split>/<speaker>/<chapter>/`` directories
holding one ``.flac`` per utterance plus one ``<speaker>-<chapter>.trans.txt``
listing "<utterance_id> TRANSCRIPT" for every utterance in that chapter —
unlike ``dataset.assemble_dataset``'s one-``.txt``-per-audio-file layout, so
this module parses it directly into the same :class:`~sme.dataset.DatasetEntry`
/ manifest.json format the rest of the pipeline already consumes.

The smallest real split (``dev-clean``) is still ~337 MB, more than this
CPU-only sweep's bandwidth/time budget can spend on one issue. ``download``
and ``extract_and_convert`` are real, correct code for that path (used as
documented in docs/evaluation.md to actually reproduce a run); the test
suite instead points ``extract_and_convert`` at a small locally-built
fixture directory with the same layout, never a real network fetch or the
full archive -- the CPU/network boundary this rule asks to mock, done
explicitly rather than hidden behind a silent skip.
"""

from __future__ import annotations

import os
import re
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

from .dataset import DatasetEntry, DatasetError, probe_duration

DEFAULT_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
_TRANS_LINE = re.compile(r"^(\S+)\s+(.+)$")
_AUDIO_EXTS = (".flac", ".wav", ".mp3")
_EXTRACT_MARKER = ".extracted.ok"


class LibriSpeechError(DatasetError):
    pass


@dataclass(frozen=True)
class LibriSpeechConfig:
    url: str = DEFAULT_URL
    language: str = "en"
    max_utterances: int | None = None
    cache_dir: str = "data/librispeech"

    def __post_init__(self) -> None:
        if self.max_utterances is not None and self.max_utterances <= 0:
            raise LibriSpeechError("max_utterances must be positive or null")


def load_config(config_path: str | os.PathLike[str] | None = None) -> LibriSpeechConfig:
    """Rule 7: lenient with no path (built-in defaults); strict with an
    explicit path that doesn't exist (hard error, never a silent
    fall-through)."""

    if config_path is None:
        return LibriSpeechConfig()
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise LibriSpeechError("config file must contain a mapping at the top level")
    return LibriSpeechConfig(
        url=str(raw.get("url", DEFAULT_URL)),
        language=str(raw.get("language", "en")),
        max_utterances=raw.get("max_utterances"),
        cache_dir=str(raw.get("cache_dir", "data/librispeech")),
    )


def _atomic_download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh, urllib.request.urlopen(url) as resp:  # noqa: S310
            while chunk := resp.read(1024 * 1024):
                fh.write(chunk)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def download(url: str, dest_dir: str | os.PathLike[str]) -> Path:
    """Download ``url`` (a LibriSpeech split tarball) into ``dest_dir``,
    skipping the fetch if it was already downloaded completely.

    Existence alone doesn't mean "already downloaded" for a multi-hundred-MB
    streamed write, so completion is the atomic rename landing the final
    name — a killed download leaves only the ``.tmp`` sibling behind, never
    a truncated file under the real name (portfolio rules 1/2).
    """

    dest = Path(dest_dir) / Path(url).name
    if dest.is_file():
        return dest
    _atomic_download(url, dest)
    return dest


def extract(tarball_path: str | os.PathLike[str], dest_dir: str | os.PathLike[str]) -> Path:
    """Extract ``tarball_path`` into ``dest_dir``, skipping if a previous
    run already completed (marker file, not just "the directory exists" —
    an interrupted extraction leaves partial content behind)."""

    dest = Path(dest_dir)
    marker = dest / _EXTRACT_MARKER
    if marker.is_file():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    # `filter="data"` (PEP 706) needs Python 3.12+; fall back to the
    # unfiltered extract on 3.10/3.11 rather than a TypeError, since the
    # source (OpenSLR) is trusted content, not an untrusted upload.
    extract_kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    with tarfile.open(tarball_path) as tar:
        tar.extractall(dest, **extract_kwargs)  # noqa: S202 - trusted OpenSLR source
    marker.write_text("ok", encoding="utf-8")
    return dest


def _iter_trans_files(root: Path):
    yield from sorted(root.rglob("*.trans.txt"))


def extract_and_convert(
    root_dir: str | os.PathLike[str],
    language: str = "en",
    max_utterances: int | None = None,
) -> list[DatasetEntry]:
    """Walk an already-extracted LibriSpeech-layout tree and build
    :class:`DatasetEntry` records: one per utterance, transcript parsed from
    its chapter's ``*.trans.txt``, audio located by utterance id in the same
    directory, duration probed the same way the rest of the pipeline does.
    """

    root = Path(root_dir)
    if not root.is_dir():
        raise LibriSpeechError(f"LibriSpeech directory not found: {root}")

    entries: list[DatasetEntry] = []
    for trans_path in _iter_trans_files(root):
        chapter_dir = trans_path.parent
        for line in trans_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            match = _TRANS_LINE.match(line)
            if not match:
                continue
            utt_id, text = match.group(1), match.group(2)

            audio_path = next(
                (
                    chapter_dir / f"{utt_id}{ext}"
                    for ext in _AUDIO_EXTS
                    if (chapter_dir / f"{utt_id}{ext}").is_file()
                ),
                None,
            )
            if audio_path is None:
                raise LibriSpeechError(f"no audio file for utterance {utt_id!r} in {chapter_dir}")

            entries.append(
                DatasetEntry(
                    audio_path=str(audio_path.resolve()),
                    transcript=text,
                    language=language,
                    duration_seconds=probe_duration(audio_path),
                )
            )
            if max_utterances is not None and len(entries) >= max_utterances:
                return entries
    if not entries:
        raise LibriSpeechError(f"no transcribed utterances found under {root}")
    return entries


def load_librispeech_subset(config: LibriSpeechConfig) -> list[DatasetEntry]:
    """End-to-end: download (if needed) + extract (if needed) + convert."""

    cache_dir = Path(config.cache_dir)
    tarball = download(config.url, cache_dir)
    extracted_dir = cache_dir / "extracted"
    extract(tarball, extracted_dir)
    return extract_and_convert(
        extracted_dir, language=config.language, max_utterances=config.max_utterances
    )
