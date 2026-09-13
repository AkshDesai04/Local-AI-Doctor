#!/bin/sh
set -eu

backend_health_url="${BACKEND_HEALTH_URL:-http://backend:6767/api/v1/health}"
backend_wait_timeout="${BACKEND_WAIT_TIMEOUT_SECONDS:-180}"

case "$backend_wait_timeout" in
    ''|*[!0-9]*)
        echo "BACKEND_WAIT_TIMEOUT_SECONDS must be a positive integer" >&2
        exit 64
        ;;
    0)
        echo "BACKEND_WAIT_TIMEOUT_SECONDS must be at least 1" >&2
        exit 64
        ;;
esac

backend_is_healthy() {
    response="$(
        wget --quiet --timeout=3 --tries=1 --output-document=- \
            --header='Host: 127.0.0.1:6767' \
            "$backend_health_url" 2>/dev/null || true
    )"
    printf '%s\n' "$response" | grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"'
}

wait_for_backend() {
    started_at="$(date +%s)"
    deadline=$((started_at + backend_wait_timeout))
    while ! backend_is_healthy; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "Backend did not become healthy within ${backend_wait_timeout} seconds" >&2
            exit 1
        fi
        sleep 1
    done
}

while :; do
    wait_for_backend
    echo "Backend healthy; holding frontend startup for 16 seconds"
    sleep 16
    if backend_is_healthy; then
        break
    fi
    echo "Backend lost health during the startup hold; restarting the health gate" >&2
done

echo "Frontend startup hold complete; starting nginx"
exec nginx -g 'daemon off;'
