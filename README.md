# Polylinguist

Polylinguist is a desktop-only local Stremio subtitle addon. It runs as a
FastAPI sidecar on `127.0.0.1`, fetches a primary subtitle candidate, translates
it with a user-selected local model backend, and serves translated or dual
subtitles back to Stremio.

This project currently targets desktop use first. TV and mobile deployment are
out of scope for the initial release.

## What it does

- exposes a local Stremio addon manifest
- lets you choose source language, target language, model, and CPU or GPU mode
- installs translation backends locally
- fetches source subtitles from the OpenSubtitles Stremio subtitle endpoint
- generates translated subtitles with shared timestamps
- shows live install and subtitle generation activity in the configurator

## Current limitations

- it does not translate embedded subtitles directly from the Stremio player
- it does not translate "whatever subtitle track the user selected" inside the player
- source subtitle discovery is currently built around the OpenSubtitles Stremio endpoint
- Stremio may require re-selecting a subtitle after background translation finishes

## Requirements

- Python 3.11 or newer
- desktop Stremio client
- Windows is the most tested target so far

Optional:

- NVIDIA CUDA-capable GPU for MarianMT GPU runs
- extra free disk space for model downloads and subtitle cache

## Install

Create a virtual environment and install the app:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

If you want to preinstall model runtimes up front instead of letting Polylinguist
bootstrap them on demand:

```powershell
pip install -e .[models,system]
```

That optional install includes:

- `argostranslate`
- `huggingface-hub`
- `sentencepiece`
- `transformers`
- `torch`
- `psutil`

## Run

The simplest way to run the sidecar is:

```powershell
polylinguist
```

That starts the local addon on:

- `http://127.0.0.1:8000/configure`

For development with explicit uvicorn control:

```powershell
uvicorn polylinguist.app:create_app --factory --host 127.0.0.1 --port 8000 --reload
```

## First-time setup

1. Open `http://127.0.0.1:8000/configure`.
2. Choose:
   - primary subtitle language
   - target translation language
   - subtitle mode
   - processing target: `auto`, `cpu`, or `cuda`
3. Click `Evaluate models`.
4. Install the model backend you want to use.
5. Click `Copy manifest URL`.
6. Add that manifest URL to Stremio as an addon.

Notes:

- Polylinguist can reuse an existing Python runtime on your machine if it
  already has the needed packages installed.
- If required packages are missing, Polylinguist can install them on demand.
- The configurator shows install progress and recent subtitle generation jobs.

## Using it in Stremio

1. Start playback for a movie or episode.
2. Open the subtitle picker.
3. Choose one of the Polylinguist subtitle entries.
4. If the subtitle is still being generated, Stremio may temporarily show a
   status subtitle telling you to check the configurator.
5. Watch `Translation Activity` in the configurator until the job is completed.
6. Re-select the Polylinguist subtitle if Stremio does not automatically refresh.

## Configurator behavior

The local configurator currently supports:

- source subtitle language
- target translation language
- subtitle output mode
- target model selection
- processing target selection
- local default settings save
- manifest URL copy
- live model install activity
- live subtitle generation activity

The generated manifest URL encodes the selected configuration, so model choice,
language pair, output mode, and CPU or GPU preference travel with the addon URL.

## Model backends

Polylinguist currently supports these local backends:

### Argos Translate

- CPU oriented
- smallest install footprint
- good low-end or offline default

### MarianMT / OPUS-MT

- default path for many common language pairs
- works on CPU
- significantly better speed on CUDA when available

### NLLB-200 distilled 600M

- larger multilingual fallback
- better suited to stronger machines
- GPU is preferred where practical

## Benchmark snapshot

Benchmarks were run on:

- sample file: TV episode subtitle file
- workload: `585` subtitle cues, `19,505` subtitle characters
- GPU: `NVIDIA GeForce GTX 1660 Ti` with `6 GB` VRAM

Results:

| Backend | Model | Device | Translation time | Throughput | Notes |
| --- | --- | --- | ---: | ---: | --- |
| Argos Translate | `argos:en-pl` | CPU | `111.14s` | `5.264 cues/s` | Fastest CPU result in this environment |
| MarianMT | `Helsinki-NLP/opus-mt-en-ine` | CPU | `177.148s` | `3.302 cues/s` | Used Polish target token `>>pol<<` |
| MarianMT | `Helsinki-NLP/opus-mt-en-ine` | GPU | `41.298s` | `14.165 cues/s` | Peak GPU allocation about `386 MB` |

The benchmark runner lives in
[`scripts/benchmark_translation.py`](scripts/benchmark_translation.py).

## Development

Run the test suite:

```powershell
python -m pytest -q
```

Useful local paths:

- app state: `%USERPROFILE%\\.polylinguist`
- generated subtitle cache: `%USERPROFILE%\\.polylinguist\\cache\\subtitles`

You can override the default state directory with:

```powershell
$env:POLYLINGUIST_HOME = "C:\\path\\to\\polylinguist-data"
```

If you want Polylinguist to prefer a specific existing Python interpreter for
model work:

```powershell
$env:POLYLINGUIST_PYTHON = "C:\\Path\\To\\python.exe"
```
