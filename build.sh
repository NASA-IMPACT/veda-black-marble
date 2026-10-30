#!/usr/bin/env bash
set -euo pipefail

# Thin OGC/DPS build: install packaging deps + shallow-clone upstream science code.
# Science lives in NASA-IMPACT/veda-black-marble — this repo only wraps it.

basedir=$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")" && pwd)
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/NASA-IMPACT/veda-black-marble.git}"
UPSTREAM_REF="${UPSTREAM_REF:-main}"
VENDOR_DIR="${basedir}/vendor/veda-black-marble"

echo "Ensuring uv is available"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || {
  echo "ERROR: uv not found after install" >&2
  exit 1
}

echo "Ensuring maap-py is available (MAAP Secrets)"
python3 -m pip install --quiet --no-cache-dir 'maap-py' || \
  pip install --quiet --no-cache-dir 'maap-py'

echo "Shallow-cloning upstream Black Marble (${UPSTREAM_URL}@${UPSTREAM_REF})"
rm -rf "${VENDOR_DIR}"
mkdir -p "$(dirname "${VENDOR_DIR}")"
git clone --depth 1 --single-branch --branch "${UPSTREAM_REF}" \
  "${UPSTREAM_URL}" "${VENDOR_DIR}"

echo "Syncing upstream dependencies with uv"
(
  cd "${VENDOR_DIR}"
  uv sync --no-dev
  uv run blackmarble --help >/dev/null
)

chmod +x "${basedir}/run.sh" "${basedir}/build.sh" "${basedir}/resolve_earthdata_token.py"
echo "Build complete → ${VENDOR_DIR}"
