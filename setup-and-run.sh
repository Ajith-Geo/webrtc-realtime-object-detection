#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Unified setup & run script (Linux/macOS) for the WebRTC demo.
# Mirrors the PowerShell script logic:
#  1. Verify python, node, npm, ngrok availability
#  2. Prompt for ngrok authtoken (if not already configured)
#  3. Create/upgrade virtual environment (.venv)
#  4. Install npm deps & Python deps
#  5. Ask for mode: wasm | server
#     - wasm   -> npm run start:wasm
#     - server -> npm run start:server + python-receiver/server.py
#  6. Start ngrok http 3000 and print public HTTPS URL
#  7. Trap Ctrl+C to cleanly stop background processes
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# ---------- stylistic helpers ----------
bold() { printf '\033[1m%s\033[0m' "$*"; }
cyan() { printf '\033[36m%s\033[0m' "$*"; }
yellow() { printf '\033[33m%s\033[0m' "$*"; }
red() { printf '\033[31m%s\033[0m' "$*"; }
green() { printf '\033[32m%s\033[0m' "$*"; }
section() { echo; echo "==== $(cyan "$1") ===="; echo; }
info() { echo "[INFO] $*"; }
warn() { echo "$(yellow "[WARN]") $*"; }
err()  { echo "$(red "[ERR ]") $*" >&2; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

# ---- network helpers ----
is_port_in_use() {
  local port=$1
  # Try nc
  if command_exists nc; then
    nc -z 127.0.0.1 "$port" >/dev/null 2>&1 && return 0 || return 1
  fi
  # Try lsof
  if command_exists lsof; then
    lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 && return 0 || return 1
  fi
  # Try netstat
  if command_exists netstat; then
    netstat -an 2>/dev/null | grep -E "[:.]$port +LISTEN" >/dev/null 2>&1 && return 0 || return 1
  fi
  return 1
}

find_free_port() {
  local start=$1 max=$2 p
  p=$start
  while (( p <= max )); do
    if ! is_port_in_use "$p"; then echo "$p"; return 0; fi
    ((p++))
  done
  return 1
}

section "Prerequisite checks"

# Track detailed status
MISSING_ANY=0
MISSING_CRITICAL=()

report_tool() {
  local tool=$1 alias_list=$2
  local displayed=$tool
  if [[ -n "$alias_list" ]]; then displayed="$tool ($alias_list)"; fi
  if command_exists "$tool"; then
    local ver; ver=$("$tool" --version 2>&1 | head -n1 || true)
    info "$displayed present ($ver)"
  else
    warn "$displayed missing"
    ((MISSING_ANY++)) || true
  fi
}

# Raw checks (some have alternatives handled below)
report_tool node ""
report_tool npm ""
report_tool ngrok ""

# Robust Python selection (avoid Windows store alias stub)
select_python() {
  local candidates=(python3 python)
  for exe in "${candidates[@]}"; do
    if command_exists "$exe"; then
      local out; out=$("$exe" --version 2>&1 || true)
      if grep -qi 'was not found' <<<"$out"; then continue; fi
      echo "$exe"; return 0
    fi
  done
  return 1
}

if PY=$(select_python); then
  info "Using Python interpreter candidate: $PY ($("$PY" --version 2>&1 | head -n1))"
else
  MISSING_CRITICAL+=(Python)
fi

# Determine critical missing components (Python, node, ngrok)
command_exists node || MISSING_CRITICAL+=(Node.js)
command_exists ngrok || MISSING_CRITICAL+=(ngrok)

if ((${#MISSING_CRITICAL[@]} > 0)); then
  err "Missing required components: ${MISSING_CRITICAL[*]}"
  echo
  echo "Installation hints:" >&2
  for comp in "${MISSING_CRITICAL[@]}"; do
    case $comp in
      Python)
        echo " - Python: https://www.python.org/downloads/ (check 'Add to PATH')" >&2 ;;
      Node.js)
        echo " - Node.js: https://nodejs.org/en/download (LTS recommended)" >&2 ;;
      ngrok)
        echo " - ngrok: https://ngrok.com/download (ensure 'ngrok' in PATH)" >&2 ;;
    esac
  done
  echo >&2
  err "Terminate: install the missing dependencies then re-run this script.";
  exit 1
fi

# ---------- ngrok authtoken ----------
if command_exists ngrok; then
  section "ngrok auth token"
  info "Forcing ngrok auth token configuration"
  if ngrok config add-authtoken 31TCv0ZqBRxpzDjug3bWCcOmlTV_42qafSCBiHT9WR28tgF5H >/dev/null 2>&1; then
    info "ngrok auth token set/overwritten"
  else
    warn "Failed to set ngrok auth token"
  fi
fi

# ---------- Python virtual environment ----------
section "Python virtual environment"
VENV_DIR="${SCRIPT_DIR}/.venv"

# If a previous broken attempt left only partial files, ensure python executable won't mislead us
create_venv=0
if [[ ! -d "$VENV_DIR" ]]; then
  create_venv=1
else
  if [[ -f "$VENV_DIR/pyvenv.cfg" ]]; then
    # Determine expected interpreter path depending on platform
    if [[ -f "$VENV_DIR/bin/python" ]]; then
      :
    elif [[ -f "$VENV_DIR/Scripts/python.exe" ]]; then
      :
    else
      warn "Existing .venv appears incomplete; recreating"
      rm -rf "$VENV_DIR"; create_venv=1
    fi
  else
    warn ".venv directory exists but not a venv; recreating"
    rm -rf "$VENV_DIR"; create_venv=1
  fi
fi

if (( create_venv == 1 )); then
  info "Creating venv at .venv using $PY"
  "$PY" -m venv "$VENV_DIR"
fi

# Activate (handle Windows vs POSIX layout)
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
else
  err "Failed to locate venv activate script"; exit 1
fi

PYTHON_VENV_BIN="$(command -v python)"
info "Using venv python: $PYTHON_VENV_BIN ($(python --version 2>&1))"
info "Upgrading pip"
python -m pip install --upgrade pip >/dev/null || warn "pip upgrade failed"

# ---------- Install JS deps ----------
section "JavaScript dependencies"
if [[ -f package.json ]]; then
  info "Running npm install"
  npm install
else
  warn "package.json missing; skipping npm install"
fi

# ---------- Install Python deps ----------
section "Python dependencies"
REQ_FILE="python-receiver/requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
  info "Installing requirements from $REQ_FILE"
  python -m pip install -r "$REQ_FILE"
else
  warn "$REQ_FILE not found; skipping"
fi

# ---------- Select mode ----------
section "Select mode"
MODE=""
while [[ -z "$MODE" ]]; do
  read -r -p "Enter mode (wasm/server): " ANSWER || true
  case "${ANSWER,,}" in
    wasm) MODE=wasm ;;
    server) MODE=server ;;
    *) warn "Invalid choice. Type 'wasm' or 'server'" ;;
  esac
done
info "Mode selected: $MODE"

# Select application port (default 3000 unless busy)
DEFAULT_PORT=3000
if is_port_in_use "$DEFAULT_PORT"; then
  warn "Port $DEFAULT_PORT already in use. Searching for a free port..."
  ALT_PORT=$(find_free_port 3001 3020 || true)
  if [[ -n "$ALT_PORT" ]]; then
    APP_PORT=$ALT_PORT
    info "Using alternative free port: $APP_PORT"
  else
    err "No free port found in 3001-3020. Free port 3000 or extend range."; exit 1
  fi
else
  APP_PORT=$DEFAULT_PORT
fi
export PORT=$APP_PORT
info "Server will listen on port $PORT"

# ---------- Launch processes ----------
section "Launching processes"
PIDS=()
log_bg() { # usage: log_bg <cmd> [args...]
  # Start command directly in background (no subshell wrapper) so $! is reliable
  "$@" &
  local pid=$!
  if [[ -z "${pid:-}" ]]; then
    warn "Could not obtain PID for: $*"
  else
    PIDS+=("$pid")
    info "Started $1 (pid=$pid)"
  fi
}

log_bg_in_dir() { # usage: log_bg_in_dir <dir> <cmd> [args...]
  local dir=$1; shift
  ( cd "$dir" && "$@" & echo $! > .__tmp_pid )
  if [[ -f "$dir/.__tmp_pid" ]]; then
    local pid
    pid=$(<"$dir/.__tmp_pid")
    rm -f "$dir/.__tmp_pid"
    if [[ -n "$pid" ]]; then
      PIDS+=("$pid")
      info "Started $1 in $dir (pid=$pid)"
    else
      warn "Failed to capture PID for $1 in $dir"
    fi
  else
    warn "PID file not created for process in $dir"
  fi
}

if [[ "$MODE" == wasm ]]; then
  log_bg env PORT=$PORT npm run start:wasm
else
  log_bg env PORT=$PORT npm run start:server
  # Python receiver (ensure it uses venv python) - fixed port 8080 internal
  if command -v python >/dev/null 2>&1; then
    log_bg_in_dir python-receiver python server.py
  else
    warn "python command not found inside venv when starting receiver"
  fi
fi

# ---------- Start ngrok ----------
NGROK_URL=""
if command_exists ngrok; then
  log_bg ngrok http --url=tomcat-beloved-feline.ngrok-free.app "$PORT"
  section "Waiting for ngrok tunnel"
  for i in {1..30}; do
    sleep 1
    # Query local API; require curl
    if command_exists curl; then
      JSON=$(curl -fsS http://127.0.0.1:4040/api/tunnels 2>/dev/null || true)
      if [[ -n "$JSON" ]]; then
        # Try jq first
        if command_exists jq; then
          NGROK_URL=$(echo "$JSON" | jq -r '.tunnels[]?.public_url' | grep '^https://' | head -n1 || true)
        else
          # crude grep/sed fallback
          NGROK_URL=$(echo "$JSON" | grep -Eo 'https://[a-zA-Z0-9.-]+\.ngrok-[a-zA-Z0-9.-]+\.app' | head -n1 || true)
          [[ -z "$NGROK_URL" ]] && NGROK_URL=$(echo "$JSON" | grep -Eo 'https://[a-zA-Z0-9.-]+\.ngrok\.io' | head -n1 || true)
        fi
        if [[ -n "$NGROK_URL" ]]; then break; fi
      fi
    fi
  done
  if [[ -n "$NGROK_URL" ]]; then
    echo
    echo "$(green 'NGROK PUBLIC URL:') $NGROK_URL"
    echo "Open ${NGROK_URL}/laptop.html on your laptop and scan QR with phone.";
  else
    warn "Failed to detect ngrok https URL (check http://127.0.0.1:4040)"
  fi
else
  warn "ngrok not installed; skipping tunnel"
fi

# ---------- Trap & wait ----------
cleanup() {
  echo
  warn "Stopping background processes..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 0.2
      if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid" 2>/dev/null || true; fi
    fi
  done
  info "All processes terminated. Bye."
}
trap cleanup INT TERM EXIT

section "Running"
echo "Background PIDs: ${PIDS[*]}"
echo "Press Ctrl+C to stop.";

# Wait (tail -f on /dev/null keeps script in foreground while trap works)
tail -f /dev/null & WAITER=$!
wait $WAITER
