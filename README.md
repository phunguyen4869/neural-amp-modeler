# NAM: Neural Amp Modeler

[![Build](https://github.com/sdatkinson/neural-amp-modeler/actions/workflows/python-package.yml/badge.svg)](https://github.com/sdatkinson/neural-amp-modeler/actions/workflows/python-package.yml)

This repository handles training models and exporting them to .nam files.
For playing trained models in real time in a standalone application or plugin, see the partner repo,
[NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

For more information about the NAM ecosystem please check out https://www.neuralampmodeler.com/.

## Training with Docker
Training can be run in a container (CLI only; the `nam` GUI is not supported).

1. Put `input.wav` and `output.wav` in `data/`. The example `data/data.json` expects
   them at `/data/input.wav` and `/data/output.wav`; adjust it to match your recording.
   `data/wavenet_a2.json` is the A2 model (packed WaveNet) used by the simplified
   trainers; for the plain WaveNet use `nam_full_configs/models/wavenet.json` instead.
2. Build the image (CPU-only PyTorch by default):
   ```bash
   docker compose build
   ```
   This builds both the `train` and `test` services. For GPU training, build with the
   CUDA-enabled PyTorch instead:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml build train
   ```
3. Train on a Linux host with an NVIDIA GPU (requires `nvidia-container-toolkit`):
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm train data.json wavenet_a2.json default.json /outputs
   ```
   Or on CPU only (very slow; Docker on macOS cannot use the GPU), using
   `default_cpu.json`, which sets `"accelerator": "cpu"`:
   ```bash
   docker compose run --rm train data.json wavenet_a2.json default_cpu.json /outputs
   ```
   For a quick smoke test, lower `max_epochs` in `default_cpu.json`.
4. The exported `.nam` model is written to `outputs/`.

### Web UI
A small browser UI trains an A2 model on a headless host (it uses the same trainer
as the GUI and Colab, with automatic latency detection and data checks):
```bash
docker compose up -d web
```
Open `http://<host>:8080`, upload `input.wav` and `output.wav`, set the epochs and
start. It shows the validation ESR per epoch and offers the `.nam` for download;
results are kept in `outputs/web/`. There is no authentication, so only expose it
on a trusted network. "Stop and export best so far" ends training early and still
exports a model. On CPU, 100 epochs takes hours.

To run the tests: `docker compose run --rm test`.

## Documentation
Online documentation can be found here: 
https://neural-amp-modeler.readthedocs.io

To build the documentation locally on a Linux system:
```bash
cd docs
make html
```

Or on Windows,
```
cd docs
make.bat html
```
