#!/bin/bash
# CCEngram Installation Script
# Usage: curl -fsSL https://raw.githubusercontent.com/JoeyEamigh/ccengram/main/scripts/install.sh | bash

set -e

REPO="JoeyEamigh/ccengram"
INSTALL_DIR="${CCENGRAM_INSTALL_DIR:-$HOME/.local/bin}"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/ccengram"
VERSION_FILE="$DATA_DIR/.version"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

detect_platform() {
    local os arch

    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    arch=$(uname -m)

    case "$os" in
        linux)  os="linux" ;;
        darwin) os="darwin" ;;
        mingw*|msys*|cygwin*) os="windows" ;;
        *) error "Unsupported OS: $os" ;;
    esac

    case "$arch" in
        x86_64|amd64) arch="x64" ;;
        arm64|aarch64) arch="arm64" ;;
        *) error "Unsupported architecture: $arch" ;;
    esac

    echo "${os}-${arch}"
}

get_latest_version() {
    local url="https://api.github.com/repos/$REPO/releases/latest"
    local version

    if command -v curl &>/dev/null; then
        version=$(curl -fsSL "$url" 2>/dev/null | grep -o '"tag_name": *"[^"]*"' | head -1 | cut -d'"' -f4)
    elif command -v wget &>/dev/null; then
        version=$(wget -qO- "$url" 2>/dev/null | grep -o '"tag_name": *"[^"]*"' | head -1 | cut -d'"' -f4)
    else
        error "curl or wget is required"
    fi

    if [ -z "$version" ]; then
        error "Failed to get latest version. Check https://github.com/$REPO/releases"
    fi

    echo "$version"
}

download_binary() {
    local version="$1"
    local platform="$2"
    local filename="ccengram-${platform}"
    local binary_name="ccengram"

    if [[ "$platform" == windows-* ]]; then
        filename="${filename}.exe"
        binary_name="ccengram.exe"
    fi

    local url="https://github.com/$REPO/releases/download/${version}/${filename}"

    info "Downloading CCEngram ${version} for ${platform}..."

    mkdir -p "$INSTALL_DIR"

    if command -v curl &>/dev/null; then
        curl -fsSL "$url" -o "$INSTALL_DIR/$binary_name.tmp" || error "Download failed. URL: $url"
    elif command -v wget &>/dev/null; then
        wget -q "$url" -O "$INSTALL_DIR/$binary_name.tmp" || error "Download failed. URL: $url"
    fi

    mv "$INSTALL_DIR/$binary_name.tmp" "$INSTALL_DIR/$binary_name"
    chmod +x "$INSTALL_DIR/$binary_name"
    mkdir -p "$DATA_DIR"
    echo "$version" > "$VERSION_FILE"

    info "Installed CCEngram ${version} to $INSTALL_DIR/$binary_name"
}

check_path() {
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        warn "$INSTALL_DIR is not in your PATH"
        echo ""
        echo "Add it to your shell profile:"
        echo ""

        if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
            echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc"
            echo "  source ~/.zshrc"
        elif [ -n "$BASH_VERSION" ] || [ -f "$HOME/.bashrc" ]; then
            echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
            echo "  source ~/.bashrc"
        else
            echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        fi
        echo ""
    fi
}

install_plugin() {
    info "Installing Claude Code plugin..."

    # Check if claude command exists
    if ! command -v claude &>/dev/null; then
        warn "Claude Code CLI not found. Install manually with:"
        echo "  /plugin marketplace add $REPO"
        echo "  /plugin install ccengram@ccengram-marketplace"
        return
    fi

    # The plugin can be installed via marketplace
    echo ""
    info "To install the CCEngram plugin in Claude Code, run:"
    echo "  /plugin marketplace add $REPO"
    echo "  /plugin install ccengram@ccengram-marketplace"
    echo ""
}

main() {
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║       CCEngram Installation          ║"
    echo "╚══════════════════════════════════════╝"
    echo ""

    local platform version

    platform=$(detect_platform)
    info "Detected platform: $platform"

    version=$(get_latest_version)
    info "Latest version: $version"

    # Check if already installed
    if [ -f "$VERSION_FILE" ]; then
        local installed_version
        installed_version=$(cat "$VERSION_FILE")
        if [ "$installed_version" = "$version" ]; then
            info "CCEngram $version is already installed"
            echo ""
            return
        fi
        info "Upgrading from $installed_version to $version"
    fi

    download_binary "$version" "$platform"
    check_path
    install_plugin

    echo ""
    info "Installation complete!"
    echo ""
    echo "Quick start:"
    echo "  ccengram health       # Check system health"
    echo "  ccengram daemon       # Start daemon"
    echo "  ccengram tui          # Launch TUI"
    echo "  ccengram --help       # Show all commands"
    echo ""
}

main "$@"
