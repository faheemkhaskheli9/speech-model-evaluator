# Evaluation Notes: Speech-to-Text / TTS Evaluation Toolkit

## Metrics

Document the specific metrics used for this project (e.g., Dice/IoU for
segmentation, WER for speech, precision/recall/F1 for classification,
MAE/RMSE for regression, or LLM-as-Judge scores for generative tasks).

## Reference dataset

STT baselines use a subset of **LibriSpeech** (`dev-clean`, OpenSLR resource
12) — a public-domain corpus of read English audiobook speech, no
employer/client data involved. `src/sme/librispeech.py` downloads the split
tarball, extracts it, and converts its `<speaker>/<chapter>/*.trans.txt` +
`.flac` layout into this project's standard `DatasetEntry`/`manifest.json`
format (the same one `sme.dataset` uses for hand-supplied audio/transcript
pairs).

To reproduce a run:

```bash
python -c "
from sme.librispeech import load_config, load_librispeech_subset
from sme.dataset import write_manifest
cfg = load_config('configs/librispeech.yaml')
entries = load_librispeech_subset(cfg)
write_manifest(entries, 'data/librispeech/manifest')
"
```

`configs/librispeech.yaml` controls the subset size (`max_utterances`),
`language`, and `cache_dir` the download/extraction is cached under. The
full `dev-clean` split is ~337 MB — expect the first run to take a while;
re-runs skip both the download and the extraction once each has completed.

## Reproducing Results

```bash
python -m src.evaluate --config configs/eval.yaml
```

## Result Log

| Date | Config | Metric | Value | Notes |
|------|--------|--------|-------|-------|
|      |        |        |       |       |
