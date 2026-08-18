# Architecture Notes: Speech-to-Text / TTS Evaluation Toolkit

## Pipeline

```text
Reference Dataset -> Run Across Providers -> Compute WER/Latency/Cost -> Comparison Report
```

## Components

- Whisper model size/version comparison
- TTS provider comparison
- Voice comparison
- Transcription accuracy (WER) measurement
- Latency benchmarking
- Audio duration handling
- Generation cost tracking

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
