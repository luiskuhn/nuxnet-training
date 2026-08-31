FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends procps git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir --upgrade pip==24.2 setuptools==75.1.0 wheel==0.44.0 \
    && python -m pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV MLF_CORE_DOCKER_RUN=TRUE CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
VOLUME ["/data", "/mlruns"]
ENTRYPOINT ["python", "-m", "numorph_nuclei_segmentation.numorph_nuclei_segmentation"]
