FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# .git is not copied, so setuptools_scm needs an explicit version.
ARG NAM_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${NAM_VERSION}

# CPU-only PyTorch by default (small download). For GPU, build with
# TORCH_INDEX_URL= (empty) to use the default PyPI wheels, which bundle CUDA.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install --no-cache-dir torch --index-url "$TORCH_INDEX_URL"; \
    fi
RUN pip install --no-cache-dir . pytest

# Mount training data at /data (audio + configs) and collect results in /outputs.
VOLUME ["/data", "/outputs"]
WORKDIR /data
ENTRYPOINT ["nam-full"]
