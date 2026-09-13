#!/bin/sh
set -eu

umask 027

if [ "${1:-serve}" != "serve" ]; then
    exec "$@"
fi

case "${CONTAINER_LISTEN_PORT:-6767}" in
    ''|*[!0-9]*)
        echo "CONTAINER_LISTEN_PORT must be an integer" >&2
        exit 64
        ;;
esac

case "${UVICORN_WORKERS:-1}" in
    ''|*[!0-9]*)
        echo "UVICORN_WORKERS must be a positive integer" >&2
        exit 64
        ;;
    0)
        echo "UVICORN_WORKERS must be at least 1" >&2
        exit 64
        ;;
esac

exec python -m uvicorn local_ai_doctor.main:create_app \
    --factory \
    --host "${CONTAINER_LISTEN_HOST:-0.0.0.0}" \
    --port "${CONTAINER_LISTEN_PORT:-6767}" \
    --workers "${UVICORN_WORKERS:-1}" \
    --timeout-graceful-shutdown "${SHUTDOWN_GRACE_SECONDS:-20}" \
    --no-access-log
