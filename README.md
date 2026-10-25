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
2. Build the image:
   ```bash
   docker compose build train
   ```
3. Train on a Linux host with an NVIDIA GPU (requires `nvidia-container-toolkit`):
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm train data.json wavenet.json default.json /outputs
   ```
   Or on CPU only (slow; Docker on macOS cannot use the GPU):
   ```bash
   docker compose run --rm train data.json wavenet.json default.json /outputs
   ```
4. The exported `.nam` model is written to `outputs/`.

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
