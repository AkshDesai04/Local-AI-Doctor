# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12.10-slim-bookworm
ARG NODE_IMAGE=node:22.14.0-bookworm-slim
ARG NGINX_IMAGE=nginx:1.28.0-alpine

FROM ${NODE_IMAGE} AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM ${NGINX_IMAGE} AS frontend
COPY --from=frontend-builder --chown=10001:10001 /build/frontend/dist/ /usr/share/nginx/html/
COPY --chown=10001:10001 docker/nginx.conf /etc/nginx/nginx.conf
COPY --chown=10001:10001 docker/frontend-entrypoint.sh /usr/local/bin/local-ai-doctor-frontend
RUN chmod 0555 /usr/local/bin/local-ai-doctor-frontend
USER 10001:10001
ENTRYPOINT ["/usr/local/bin/local-ai-doctor-frontend"]
EXPOSE 6969
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=5s --timeout=3s --start-period=20s --retries=6 \
    CMD ["wget", "--quiet", "--tries=1", "--output-document=/dev/null", "http://127.0.0.1:6969/healthz"]

FROM ${PYTHON_IMAGE} AS dependency-wheels
WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip wheel --wheel-dir /wheels --requirement requirements.lock

FROM ${PYTHON_IMAGE} AS application-wheel
WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml README.md ./
COPY backend/ ./backend/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip wheel --no-deps --wheel-dir /app-wheel .

FROM dependency-wheels AS cpu-framework-wheels
ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip wheel \
        --wheel-dir /wheels \
        --index-url "${TORCH_CPU_INDEX_URL}" \
        --extra-index-url "${PYPI_INDEX_URL}" \
        "torch==${TORCH_VERSION}+cpu" \
        "torchvision==${TORCHVISION_VERSION}+cpu"

FROM cpu-framework-wheels AS cpu-wheels
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
COPY requirements-ml.lock ./
RUN sed -e '/^[[:space:]]*torch==/d' -e '/^[[:space:]]*torchvision==/d' \
        requirements-ml.lock > /tmp/requirements-ml-no-framework.lock \
    && printf 'torch==%s+cpu\ntorchvision==%s+cpu\n' \
        "${TORCH_VERSION}" "${TORCHVISION_VERSION}" > /tmp/framework-constraints.lock \
    && python -m pip wheel \
        --wheel-dir /wheels \
        --find-links /wheels \
        --constraint /tmp/framework-constraints.lock \
        --requirement /tmp/requirements-ml-no-framework.lock

FROM dependency-wheels AS nvidia-framework-wheels
ARG PYPI_INDEX_URL=https://pypi.org/simple
COPY requirements-cuda.lock ./
RUN python -m pip wheel \
        --wheel-dir /wheels \
        --find-links /wheels \
        --extra-index-url "${PYPI_INDEX_URL}" \
        --requirement requirements-cuda.lock

FROM nvidia-framework-wheels AS nvidia-wheels
COPY requirements-ml.lock ./
RUN sed -e '/^[[:space:]]*torch==/d' -e '/^[[:space:]]*torchvision==/d' \
        requirements-ml.lock > /tmp/requirements-ml-no-framework.lock \
    && sed -n \
        -e '/^[[:space:]]*torch==/p' \
        -e '/^[[:space:]]*torchvision==/p' \
        requirements-cuda.lock > /tmp/framework-constraints.lock \
    && python -m pip wheel \
        --wheel-dir /wheels \
        --find-links /wheels \
        --constraint /tmp/framework-constraints.lock \
        --requirement /tmp/requirements-ml-no-framework.lock

FROM ${PYTHON_IMAGE} AS cpu-python
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY --from=cpu-wheels /wheels/ /wheels/
COPY --from=application-wheel /app-wheel/ /wheels/
COPY requirements.lock requirements-ml.lock /requirements/
RUN python -m pip install \
        --no-index \
        --find-links /wheels \
        --requirement /requirements/requirements.lock \
        --requirement /requirements/requirements-ml.lock \
        "local-ai-doctor" \
    && python -c "import importlib.metadata as m; t=m.version('torch'); v=m.version('torchvision'); assert '+cpu' in t and '+cpu' in v, (t,v); assert not any(d.metadata['Name'].lower().startswith('nvidia-') or d.metadata['Name'].lower() == 'bitsandbytes' for d in m.distributions())"

FROM ${PYTHON_IMAGE} AS nvidia-python
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY --from=nvidia-wheels /wheels/ /wheels/
COPY --from=application-wheel /app-wheel/ /wheels/
COPY requirements.lock requirements-ml.lock requirements-cuda.lock /requirements/
RUN python -m pip install \
        --no-index \
        --find-links /wheels \
        --requirement /requirements/requirements.lock \
        --requirement /requirements/requirements-ml.lock \
        --requirement /requirements/requirements-cuda.lock \
        "local-ai-doctor" \
    && python -c "import importlib.metadata as m; t=m.version('torch'); v=m.version('torchvision'); assert '+cu' in t and t.partition('+')[2] == v.partition('+')[2], (t,v); assert m.version('bitsandbytes') == '0.50.2'"

FROM ${PYTHON_IMAGE} AS runtime-base
ARG APP_UID=10001
ARG APP_GID=10001
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    CONTAINER_LISTEN_HOST=0.0.0.0 \
    CONTAINER_LISTEN_PORT=6767 \
    UVICORN_WORKERS=1 \
    SHUTDOWN_GRACE_SECONDS=20
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --home-dir /home/app app \
    && install -d -o app -g app -m 0750 \
        /app \
        /app/config \
        /data/database \
        /data/uploads \
        /data/cache \
        /data/exports \
        /data/backups \
    && install -d -o root -g root -m 0555 /models
WORKDIR /app
COPY --chown=app:app config/default.yaml /app/config/default.yaml
COPY --chown=app:app config/local.example.yaml /app/config/local.yaml
COPY --chown=app:app \
    docker/entrypoint.sh \
    docker/preflight.py \
    docker/sqlite_backup.py \
    /app/docker/
COPY requirements.lock requirements-ml.lock requirements-cuda.lock /app/
RUN chmod 0555 /app/docker/entrypoint.sh \
    && chmod 0444 /app/docker/*.py /app/config/*.yaml

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["serve"]
EXPOSE 6767
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import json,os,urllib.request; u='http://127.0.0.1:'+os.environ.get('CONTAINER_LISTEN_PORT','6767')+'/api/v1/health'; r=urllib.request.urlopen(u,timeout=3); p=json.load(r); raise SystemExit(0 if r.status==200 and p.get('status')=='ok' else 1)"]

FROM runtime-base AS cpu
COPY --from=cpu-python /usr/local/ /usr/local/
USER app:app

FROM runtime-base AS nvidia
COPY --from=nvidia-python /usr/local/ /usr/local/
USER app:app

# A plain `docker build .` is intentionally CPU-only. Select `--target nvidia`
# only when the host has WSL GPU passthrough and NVIDIA Container Toolkit.
FROM cpu AS production
