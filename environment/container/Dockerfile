# Container specification for the STH-SepNet course project.
# Builds a CUDA-ready environment so the training pipeline runs the same way
# on the team GPU host. For a pure CPU smoke test you can build with
# `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu`.
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

# Install the pinned dependencies (override torch index for CPU-only smoke).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
COPY environment/lockfile/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir --index-url ${TORCH_INDEX} torch==2.6.0 \
    && pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/
COPY tests/ /workspace/tests/
COPY configs/ /workspace/configs/

# Placeholder: datasets + weights are mounted at runtime (never baked in).
CMD ["bash", "/workspace/scripts/smoke_test.sh"]
