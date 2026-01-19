#!/bin/bash
# CCMemory wrapper - ensures binary exists then runs it
# This wrapper is called by hooks and MCP server configs

set -e

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(dirname "$(dirname "$(realpath "$0")")")}"
BIN_DIR="$PLUGIN_ROOT/bin"
BINARY="$BIN_DIR/ccmemory"
VERSION_FILE="$BIN_DIR/.ccmemory-version"
UPDATE_CHECK_FILE="$BIN_DIR/.last-update-check"
REPO="JoeyEamigh/ccmemory"

# Update check interval: 24 hours in seconds
UPDATE_CHECK_INTERVAL=86400

should_check_updates() {
    if [ ! -f "$UPDATE_CHECK_FILE" ]; then
        return 0
    fi
    local last_check now
    last_check=$(cat "$UPDATE_CHECK_FILE" 2>/dev/null || echo 0)
    now=$(date +%s)
    [ $((now - last_check)) -gt $UPDATE_CHECK_INTERVAL ]
}

detect_platform() {
    local os arch
    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    arch=$(uname -m)

    case "$os" in
        linux)  os="linux" ;;
        darwin) os="darwin" ;;
        mingw*|msys*|cygwin*) os="windows" ;;
        *) echo "Unsupported OS: $os" >&2; exit 1 ;;
    esac

    case "$arch" in
        x86_64|amd64) arch="x64" ;;
        arm64|aarch64) arch="arm64" ;;
        *) echo "Unsupported architecture: $arch" >&2; exit 1 ;;
    esac

    echo "${os}-${arch}"
}

get_latest_version() {
    local url="https://api.github.com/repos/$REPO/releases/latest"
    if command -v curl &>/dev/null; then
        curl -fsSL --connect-timeout 5 "$url" 2>/dev/null | grep -o '"tag_name": *"[^"]*"' | head -1 | cut -d'"' -f4
    elif command -v wget &>/dev/null; then
        wget -qO- --timeout=5 "$url" 2>/dev/null | grep -o '"tag_name": *"[^"]*"' | head -1 | cut -d'"' -f4
    else
        echo ""
    fi
}

download_binary() {
    local version="$1"
    local platform="$2"
    local filename="ccmemory-${platform}"

    if [[ "$platform" == windows-* ]]; then
        filename="${filename}.exe"
    fi

    local url="https://github.com/$REPO/releases/download/${version}/${filename}"

    echo "CCMemory: Downloading ${version} for ${platform}..." >&2

    mkdir -p "$BIN_DIR"

    if command -v curl &>/dev/null; then
        curl -fsSL "$url" -o "$BINARY.tmp"
    elif command -v wget &>/dev/null; then
        wget -q "$url" -O "$BINARY.tmp"
    else
        echo "Error: curl or wget required" >&2
        exit 1
    fi

    mv "$BINARY.tmp" "$BINARY"
    chmod +x "$BINARY"
    echo "$version" > "$VERSION_FILE"
    date +%s > "$UPDATE_CHECK_FILE"

    echo "CCMemory: Installed ${version}" >&2
}

check_for_updates_background() {
    # Run update check in background without blocking
    (
        local platform installed_version latest_version
        platform=$(detect_platform)
        installed_version=$(cat "$VERSION_FILE" 2>/dev/null || echo "")

        latest_version=$(get_latest_version)
        date +%s > "$UPDATE_CHECK_FILE"

        if [ -n "$latest_version" ] && [ -n "$installed_version" ] && [ "$latest_version" != "$installed_version" ]; then
            download_binary "$latest_version" "$platform"
        fi
    ) &>/dev/null &
    disown 2>/dev/null || true
}

ensure_binary() {
    local platform

    # If binary exists and is executable, use it
    if [ -x "$BINARY" ]; then
        # Periodically check for updates in background (once per day)
        if should_check_updates; then
            check_for_updates_background
        fi
        return 0
    fi

    # Binary doesn't exist - try to download
    platform=$(detect_platform)
    local latest_version
    latest_version=$(get_latest_version)

    if [ -z "$latest_version" ]; then
        echo "CCMemory Error: Binary not found at $BINARY" >&2
        echo "Install with: curl -fsSL https://raw.githubusercontent.com/$REPO/main/scripts/install.sh | bash" >&2
        exit 1
    fi

    download_binary "$latest_version" "$platform"
}

# Ensure binary exists
ensure_binary

# Run the binary with all passed arguments
exec "$BINARY" "$@"
