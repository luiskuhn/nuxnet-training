FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

LABEL description="NuMorph training image with CUDA 12.4, cuDNN, and Python 3.12"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git procps \
    && rm -rf /var/lib/apt/lists/*

# The CUDA Ubuntu base does not provide Python 3.12. Use the repository's
# pinned Miniforge installer so clean GitHub Actions checkouts can build the
# image without relying on a locally downloaded, gitignored installer.
ARG MINIFORGE_VERSION=24.7.1-2
RUN curl --fail --location --show-error --silent \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
        --output /tmp/Miniforge3.sh \
    && sh /tmp/Miniforge3.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3.sh \
    && /opt/conda/bin/conda install --yes python=3.12 pip=24.2 \
    && /opt/conda/bin/conda clean --all --yes
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app
COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir --upgrade pip==24.2 setuptools==75.1.0 wheel==0.44.0 \
    && python -m pip install --no-cache-dir -r requirements.txt
COPY . /app
RUN chmod 755 /app/docker-entrypoint.sh
ENV MLF_CORE_DOCKER_RUN=TRUE CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
VOLUME ["/data", "/mlruns", "/output"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import matplotlib, mlflow, numpy, pytorch_lightning, tensorboard, tifffile, torch" || exit 1
ENTRYPOINT ["/app/docker-entrypoint.sh"]
