FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# .git is not copied, so setuptools_scm needs an explicit version.
ARG NAM_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${NAM_VERSION}

# The Linux PyTorch wheels bundle CUDA, so the same image runs on CPU or GPU.
RUN pip install --no-cache-dir . pytest

# Mount training data at /data (audio + configs) and collect results in /outputs.
VOLUME ["/data", "/outputs"]
WORKDIR /data
ENTRYPOINT ["nam-full"]
