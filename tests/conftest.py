import struct
import wave

import pytest


def _write_silent_wav(path, seconds=0.5, rate=8000):
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


@pytest.fixture
def make_wav():
    return _write_silent_wav


@pytest.fixture
def sample_dataset_dir(tmp_path, make_wav):
    """A tiny valid dataset: two wavs, each with a matching transcript."""
    audio = tmp_path / "audio"
    audio.mkdir()
    make_wav(audio / "clip1.wav", seconds=0.5)
    make_wav(audio / "clip2.wav", seconds=1.0)
    (audio / "clip1.txt").write_text("hello world", encoding="utf-8")
    (audio / "clip2.txt").write_text("the quick brown fox", encoding="utf-8")
    return audio
