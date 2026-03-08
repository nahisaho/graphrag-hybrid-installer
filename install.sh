#!/usr/bin/env bash
# =============================================================================
# GraphRAG Hybrid Installer
# =============================================================================
# scispaCy + GiNZA + ドメイン辞書によるハイブリッドGraphRAG環境を
# 対話式でセットアップするインストーラー
#
# Usage:
#   bash install.sh [--target-dir /path/to/project]
# =============================================================================

set -euo pipefail

# --- Color output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}ℹ️  ${NC}$1"; }
ok()    { echo -e "${GREEN}✅ ${NC}$1"; }
warn()  { echo -e "${YELLOW}⚠️  ${NC}$1"; }
err()   { echo -e "${RED}❌ ${NC}$1"; }
header(){ echo -e "\n${BOLD}${CYAN}━━━ $1 ━━━${NC}\n"; }

INSTALLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Parse arguments
# =============================================================================
TARGET_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dir) TARGET_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash install.sh [--target-dir /path/to/project]"
      echo ""
      echo "Interactive installer for GraphRAG Hybrid (scispaCy + GiNZA + Domain Dictionary)"
      exit 0
      ;;
    *) err "Unknown option: $1"; exit 1 ;;
  esac
done

# =============================================================================
# Banner
# =============================================================================
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║        GraphRAG Hybrid Installer v1.0                       ║${NC}"
echo -e "${BOLD}${CYAN}║   scispaCy + GiNZA + Domain Dictionary for GraphRAG         ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# =============================================================================
# 1. Prerequisites check
# =============================================================================
header "1/9 Prerequisites Check"

if ! command -v python3 &>/dev/null; then
  err "Python 3 is required. Install it first."
  exit 1
fi

PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Python: $PYTHON_VER"

if ! command -v pip3 &>/dev/null && ! python3 -m pip --version &>/dev/null 2>&1; then
  err "pip is required. Install it with: python3 -m ensurepip"
  exit 1
fi
ok "Prerequisites OK"

# Detect pip install flags
PIP_FLAGS=""
if python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))" 2>/dev/null | grep -q "/usr/lib"; then
  PIP_FLAGS="--break-system-packages --user"
  info "Detected system Python, using: pip install $PIP_FLAGS"
fi

# =============================================================================
# 2. Target directory
# =============================================================================
header "2/9 Project Directory"

if [[ -z "$TARGET_DIR" ]]; then
  read -rp "$(echo -e "${CYAN}Project directory [default: ./graphrag-project]: ${NC}")" TARGET_DIR
  TARGET_DIR="${TARGET_DIR:-./graphrag-project}"
fi

TARGET_DIR="$(realpath -m "$TARGET_DIR")"
info "Target: $TARGET_DIR"

mkdir -p "$TARGET_DIR"
ok "Directory ready"

# =============================================================================
# 3. Input files location
# =============================================================================
header "3/9 Input Files"

echo "入力ファイル（Markdown/テキスト）の場所を指定してください。"
echo "  1) 既存のディレクトリを指定"
echo "  2) プロジェクト内の input/ にあとで配置"
read -rp "$(echo -e "${CYAN}選択 [1/2, default: 2]: ${NC}")" INPUT_CHOICE
INPUT_CHOICE="${INPUT_CHOICE:-2}"

if [[ "$INPUT_CHOICE" == "1" ]]; then
  read -rp "$(echo -e "${CYAN}入力ディレクトリのパス: ${NC}")" INPUT_DIR
  INPUT_DIR="$(realpath -m "$INPUT_DIR")"
  if [[ ! -d "$INPUT_DIR" ]]; then
    warn "Directory not found: $INPUT_DIR (will be created)"
    mkdir -p "$INPUT_DIR"
  fi
  # Create symlink
  ln -sfn "$INPUT_DIR" "$TARGET_DIR/input"
  ok "Input: $INPUT_DIR → $TARGET_DIR/input"
else
  INPUT_DIR="$TARGET_DIR/input"
  mkdir -p "$INPUT_DIR"
  ok "Input: $INPUT_DIR (place your files here)"
fi

INPUT_FILE_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name "*.md" -o -name "*.txt" 2>/dev/null | wc -l)
info "Current files: $INPUT_FILE_COUNT"

# =============================================================================
# 4. LLM Provider
# =============================================================================
header "4/9 LLM Provider (Completion)"

echo "LLMプロバイダーを選択してください:"
echo "  1) OpenAI   (gpt-4o-mini, gpt-4o, etc.)"
echo "  2) Azure OpenAI"
echo "  3) Ollama   (ローカル実行)"
read -rp "$(echo -e "${CYAN}選択 [1/2/3, default: 1]: ${NC}")" LLM_CHOICE
LLM_CHOICE="${LLM_CHOICE:-1}"

LLM_PROVIDER=""
LLM_MODEL=""
API_KEY=""
AZURE_ENDPOINT=""
OLLAMA_URL=""

case "$LLM_CHOICE" in
  1)
    LLM_PROVIDER="openai"
    read -rp "$(echo -e "${CYAN}Model [default: gpt-4o-mini]: ${NC}")" LLM_MODEL
    LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
    read -rp "$(echo -e "${CYAN}OpenAI API Key: ${NC}")" API_KEY
    if [[ -z "$API_KEY" ]]; then
      # Try to find from environment or .bashrc
      API_KEY="${OPENAI_API_KEY:-}"
      if [[ -z "$API_KEY" ]] && [[ -f ~/.bashrc ]]; then
        API_KEY=$(grep 'OPENAI_API_KEY' ~/.bashrc 2>/dev/null | head -1 | sed "s/.*export OPENAI_API_KEY=//;s/['\"]//g" || true)
      fi
      if [[ -n "$API_KEY" ]]; then
        info "Found API key from environment"
      else
        warn "API key not set. Edit .env later."
      fi
    fi
    ;;
  2)
    LLM_PROVIDER="azure"
    read -rp "$(echo -e "${CYAN}Model [default: gpt-4o-mini]: ${NC}")" LLM_MODEL
    LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
    read -rp "$(echo -e "${CYAN}Azure OpenAI Endpoint URL: ${NC}")" AZURE_ENDPOINT
    read -rp "$(echo -e "${CYAN}Azure OpenAI API Key: ${NC}")" API_KEY
    ;;
  3)
    LLM_PROVIDER="ollama"
    read -rp "$(echo -e "${CYAN}Ollama URL [default: http://localhost:11434/v1]: ${NC}")" OLLAMA_URL
    OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/v1}"

    # List available models
    OLLAMA_BASE="${OLLAMA_URL%/v1}"
    OLLAMA_BASE="${OLLAMA_BASE%/}"
    info "Checking Ollama at $OLLAMA_BASE ..."
    MODELS=$(curl -s "$OLLAMA_BASE/api/tags" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for m in d.get('models', []):
        print(f\"  - {m['name']}\")
except: print('  (could not list models)')
" 2>/dev/null || echo "  (connection failed)")
    echo "Available models:"
    echo "$MODELS"

    read -rp "$(echo -e "${CYAN}Model [default: qwen2.5:7b]: ${NC}")" LLM_MODEL
    LLM_MODEL="${LLM_MODEL:-qwen2.5:7b}"
    API_KEY="ollama"
    ;;
esac
ok "LLM: $LLM_PROVIDER / $LLM_MODEL"

# =============================================================================
# 5. Embedding Provider
# =============================================================================
header "5/9 Embedding Provider"

echo "Embeddingプロバイダーを選択してください:"
echo "  1) Ollama bge-m3        (ローカル, 1024dim, 多言語対応, 無料)"
echo "  2) OpenAI small         (text-embedding-3-small, 1536dim)"
echo "  3) OpenAI large         (text-embedding-3-large, 3072dim)"
echo "  4) Azure OpenAI"
read -rp "$(echo -e "${CYAN}選択 [1/2/3/4, default: 1]: ${NC}")" EMB_CHOICE
EMB_CHOICE="${EMB_CHOICE:-1}"

EMB_PROVIDER=""
EMB_MODEL=""
case "$EMB_CHOICE" in
  1)
    EMB_PROVIDER="ollama_bge"
    EMB_MODEL="bge-m3"
    if [[ -z "$OLLAMA_URL" ]]; then
      read -rp "$(echo -e "${CYAN}Ollama URL [default: http://localhost:11434/v1]: ${NC}")" OLLAMA_URL
      OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/v1}"
    fi
    # Check if bge-m3 is pulled
    OLLAMA_BASE="${OLLAMA_URL%/v1}"
    OLLAMA_BASE="${OLLAMA_BASE%/}"
    if curl -s "$OLLAMA_BASE/api/tags" 2>/dev/null | grep -q "bge-m3"; then
      ok "bge-m3 found in Ollama"
    else
      warn "bge-m3 not found. Run: ollama pull bge-m3"
    fi
    ;;
  2) EMB_PROVIDER="openai_small"; EMB_MODEL="text-embedding-3-small" ;;
  3) EMB_PROVIDER="openai_large"; EMB_MODEL="text-embedding-3-large" ;;
  4)
    EMB_PROVIDER="azure"
    read -rp "$(echo -e "${CYAN}Embedding model [default: text-embedding-3-large]: ${NC}")" EMB_MODEL
    EMB_MODEL="${EMB_MODEL:-text-embedding-3-large}"
    if [[ -z "$AZURE_ENDPOINT" ]]; then
      read -rp "$(echo -e "${CYAN}Azure OpenAI Endpoint URL: ${NC}")" AZURE_ENDPOINT
    fi
    ;;
esac
ok "Embedding: $EMB_PROVIDER / $EMB_MODEL"

# =============================================================================
# 6. NLP Mode
# =============================================================================
header "6/9 NLP Mode"

echo "NLP抽出モードを選択してください:"
echo "  1) hybrid    — scispaCy + GiNZA + ドメイン辞書 (推奨: 多言語科学論文)"
echo "  2) scispacy  — scispaCy のみ (英語科学論文)"
echo "  3) ginza     — GiNZA のみ (日本語テキスト)"
read -rp "$(echo -e "${CYAN}選択 [1/2/3, default: 1]: ${NC}")" NLP_CHOICE
NLP_CHOICE="${NLP_CHOICE:-1}"

NLP_MODE=""
case "$NLP_CHOICE" in
  1) NLP_MODE="hybrid" ;;
  2) NLP_MODE="scispacy" ;;
  3) NLP_MODE="ginza" ;;
esac
ok "NLP mode: $NLP_MODE"

# =============================================================================
# 7. Install Python packages
# =============================================================================
header "7/9 Installing Dependencies"

info "Installing Python packages..."
# shellcheck disable=SC2086
pip3 install $PIP_FLAGS -q graphrag ginza ja-ginza scispacy tiktoken 2>&1 | tail -3
ok "Base packages installed"

# MCP SDK
info "Installing MCP SDK..."
# shellcheck disable=SC2086
pip3 install $PIP_FLAGS -q "mcp[cli]" 2>&1 | tail -2
if python3 -c "from mcp.server.fastmcp import FastMCP; print('OK')" 2>/dev/null | grep -q "OK"; then
  ok "MCP SDK installed"
else
  warn "MCP SDK installation failed. Install manually: pip install 'mcp[cli]'"
fi

if [[ "$NLP_MODE" == "hybrid" ]] || [[ "$NLP_MODE" == "scispacy" ]]; then
  info "Installing scispaCy model (en_core_sci_lg)..."
  # Try versioned URLs
  for VER in "0.5.4" "0.5.3" "0.5.1"; do
    URL="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v${VER}/en_core_sci_lg-${VER}.tar.gz"
    if curl -sI "$URL" 2>/dev/null | head -1 | grep -q "200"; then
      # shellcheck disable=SC2086
      pip3 install $PIP_FLAGS -q "$URL" 2>&1 | tail -2
      ok "en_core_sci_lg installed (v${VER})"
      break
    fi
  done

  # Verify
  if ! python3 -c "import spacy; spacy.load('en_core_sci_lg')" 2>/dev/null; then
    warn "en_core_sci_lg could not be loaded. Install manually."
  fi
fi

# Verify GiNZA
if [[ "$NLP_MODE" == "hybrid" ]] || [[ "$NLP_MODE" == "ginza" ]]; then
  if python3 -c "import spacy; spacy.load('ja_ginza')" 2>/dev/null; then
    ok "ja_ginza model OK"
  else
    warn "ja_ginza could not be loaded. Try: pip install ja-ginza"
  fi
fi

# =============================================================================
# 8. Setup project
# =============================================================================
header "8/9 Project Setup"

# Copy source files
info "Copying source files..."
mkdir -p "$TARGET_DIR/src"
cp "$INSTALLER_DIR/src/hybrid_extractor.py"       "$TARGET_DIR/src/"
cp "$INSTALLER_DIR/src/run_graphrag_hybrid.py"     "$TARGET_DIR/src/"
cp "$INSTALLER_DIR/src/build_domain_dictionary.py" "$TARGET_DIR/src/"
cp "$INSTALLER_DIR/src/build_bilingual_thesaurus.py" "$TARGET_DIR/src/"
cp "$INSTALLER_DIR/src/generate_settings.py"       "$TARGET_DIR/src/"
cp "$INSTALLER_DIR/src/graphrag_mcp_server.py"     "$TARGET_DIR/src/"
cp "$INSTALLER_DIR/src/patch_noun_graph.py"        "$TARGET_DIR/src/"
ok "Source files copied"

# Apply NLP edge extraction optimization patch
info "Applying NLP edge extraction optimization (Top-K=17, co-occurrence≥2)..."
if python3 "$INSTALLER_DIR/src/patch_noun_graph.py" --max-k 17 --min-cooccurrence 2 2>&1; then
  ok "NLP optimization patch applied"
else
  warn "NLP optimization patch failed (non-critical). You can apply manually:"
  warn "  python3 $TARGET_DIR/src/patch_noun_graph.py --max-k 17 --min-cooccurrence 2"
fi

# Initialize GraphRAG (prompts)
info "Initializing GraphRAG prompts..."
cd "$TARGET_DIR"
python3 -m graphrag init --root "$TARGET_DIR" --force 2>/dev/null || true
ok "Prompts initialized"

# Generate config JSON for settings generator
CONFIG_JSON="$TARGET_DIR/.installer_config.json"
python3 -c "
import json
config = {
    'llm_provider': '$LLM_PROVIDER',
    'llm_model': '$LLM_MODEL',
    'api_key': '$API_KEY',
    'azure_endpoint': '$AZURE_ENDPOINT',
    'ollama_url': '$OLLAMA_URL',
    'embedding_provider': '$EMB_PROVIDER',
    'embedding_model': '$EMB_MODEL',
    'nlp_mode': '$NLP_MODE',
    'input_dir': 'input',
    'domain_dictionary': '$TARGET_DIR/domain_dictionary.json',
}
with open('$CONFIG_JSON', 'w') as f:
    json.dump(config, f, indent=2)
"
ok "Config saved"

# Generate settings.yaml and .env
python3 "$TARGET_DIR/src/generate_settings.py" \
  --config "$CONFIG_JSON" \
  --output-dir "$TARGET_DIR"

# Create convenience scripts
cat > "$TARGET_DIR/build_dictionary.sh" << 'DICTEOF'
#!/usr/bin/env bash
# Build domain dictionary from input files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/src/build_domain_dictionary.py" \
  --input-dir "$SCRIPT_DIR/input" \
  --output "$SCRIPT_DIR/domain_dictionary.json" \
  --model en_core_sci_lg \
  --ja-model ja_ginza \
  "$@"
DICTEOF
chmod +x "$TARGET_DIR/build_dictionary.sh"

cat > "$TARGET_DIR/build_thesaurus.sh" << 'THEOF'
#!/usr/bin/env bash
# Build bilingual thesaurus from domain dictionary
# Requires: domain_dictionary.json (run build_dictionary.sh first)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -a
source "$SCRIPT_DIR/.env" 2>/dev/null || true
set +a

API_KEY="${OPENAI_API_KEY:-${GRAPHRAG_API_KEY:-}}"
if [[ -z "$API_KEY" ]] || [[ "$API_KEY" == "ollama" ]]; then
  echo "⚠️  OpenAI API key required for thesaurus generation."
  echo "   Set OPENAI_API_KEY environment variable."
  exit 1
fi

if [[ ! -f "$SCRIPT_DIR/domain_dictionary.json" ]]; then
  echo "⚠️  domain_dictionary.json not found. Run ./build_dictionary.sh first."
  exit 1
fi

python3 "$SCRIPT_DIR/src/build_bilingual_thesaurus.py" \
  --input-dict "$SCRIPT_DIR/domain_dictionary.json" \
  --output "$SCRIPT_DIR/bilingual_thesaurus.json" \
  --api-key "$API_KEY" \
  "$@"

echo ""
echo "💡 To use the thesaurus, update GRAPHRAG_DOMAIN_DICTIONARY in .env:"
echo "   GRAPHRAG_DOMAIN_DICTIONARY=$SCRIPT_DIR/bilingual_thesaurus.json"
THEOF
chmod +x "$TARGET_DIR/build_thesaurus.sh"

cat > "$TARGET_DIR/run_index.sh" << 'IDXEOF'
#!/usr/bin/env bash
# Run GraphRAG indexing with hybrid extractor
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env
set -a
source "$SCRIPT_DIR/.env"
set +a
export GRAPHRAG_ROOT="$SCRIPT_DIR"

METHOD="${1:-fast}"
echo "🚀 Starting GraphRAG index (method: $METHOD)..."
python3 "$SCRIPT_DIR/src/run_graphrag_hybrid.py" index \
  --root "$SCRIPT_DIR" \
  --method "$METHOD" \
  --skip-validation \
  --verbose
IDXEOF
chmod +x "$TARGET_DIR/run_index.sh"

cat > "$TARGET_DIR/run_query.sh" << 'QEOF'
#!/usr/bin/env bash
# Query the GraphRAG index
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

set -a
source "$SCRIPT_DIR/.env"
set +a
export GRAPHRAG_ROOT="$SCRIPT_DIR"

METHOD="${1:-local}"
shift
QUERY="$*"

if [[ -z "$QUERY" ]]; then
  echo "Usage: ./run_query.sh [local|global|drift|basic] \"your question\""
  exit 1
fi

python3 "$SCRIPT_DIR/src/run_graphrag_hybrid.py" query \
  --root "$SCRIPT_DIR" \
  --method "$METHOD" \
  "$QUERY"
QEOF
chmod +x "$TARGET_DIR/run_query.sh"

ok "Convenience scripts created"

# =============================================================================
# 9. MCP Server Setup
# =============================================================================
header "9/9 MCP Server Setup"

cat > "$TARGET_DIR/run_mcp_server.sh" << 'MCPEOF'
#!/usr/bin/env bash
# Start GraphRAG MCP Server
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env
set -a
source "$SCRIPT_DIR/.env" 2>/dev/null || true
set +a
export GRAPHRAG_ROOT="$SCRIPT_DIR"

TRANSPORT="${1:-stdio}"
PORT="${2:-8765}"

case "$TRANSPORT" in
  stdio)
    echo "🔌 Starting GraphRAG MCP Server (stdio mode)..."
    echo "   Use this mode for Claude Desktop / VS Code Copilot"
    python3 "$SCRIPT_DIR/src/graphrag_mcp_server.py" --transport stdio
    ;;
  http)
    echo "🌐 Starting GraphRAG MCP Server (HTTP mode on port $PORT)..."
    python3 "$SCRIPT_DIR/src/graphrag_mcp_server.py" --transport http --port "$PORT"
    ;;
  *)
    echo "Usage: ./run_mcp_server.sh [stdio|http] [port]"
    echo ""
    echo "  stdio  — Standard I/O (Claude Desktop, VS Code)"
    echo "  http   — Streamable HTTP (remote clients)"
    exit 1
    ;;
esac
MCPEOF
chmod +x "$TARGET_DIR/run_mcp_server.sh"
ok "MCP Server script created"

# Generate MCP config for Claude Desktop / VS Code
MCP_CONFIG="$TARGET_DIR/mcp_config.json"
python3 -c "
import json
config = {
    'mcpServers': {
        'graphrag': {
            'command': 'python3',
            'args': ['$TARGET_DIR/src/graphrag_mcp_server.py'],
            'env': {
                'GRAPHRAG_ROOT': '$TARGET_DIR'
            }
        }
    }
}
with open('$MCP_CONFIG', 'w') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
"
ok "MCP config generated: $MCP_CONFIG"
info "Copy to Claude Desktop: ~/.config/claude/claude_desktop_config.json"
info "Or VS Code settings: .vscode/mcp.json"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║              Installation Complete! 🎉                      ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Project:${NC}    $TARGET_DIR"
echo -e "${BOLD}LLM:${NC}        $LLM_PROVIDER / $LLM_MODEL"
echo -e "${BOLD}Embedding:${NC}  $EMB_PROVIDER / $EMB_MODEL"
echo -e "${BOLD}NLP Mode:${NC}   $NLP_MODE"
echo -e "${BOLD}Input:${NC}      $TARGET_DIR/input/"
echo ""
echo -e "${BOLD}${YELLOW}Next Steps:${NC}"
echo "  1. Place your Markdown/text files in: $TARGET_DIR/input/"
echo "  2. Build domain dictionary (optional):"
echo "     cd $TARGET_DIR && ./build_dictionary.sh"
echo "  3. Build bilingual thesaurus (optional, improves JP-EN linking):"
echo "     cd $TARGET_DIR && ./build_thesaurus.sh"
echo "  4. Build index:"
echo "     cd $TARGET_DIR && ./run_index.sh fast"
echo "  5. Query:"
echo "     cd $TARGET_DIR && ./run_query.sh local \"your question\""
echo "  6. Start MCP Server (after index build):"
echo "     cd $TARGET_DIR && ./run_mcp_server.sh stdio"
echo ""
echo -e "${BOLD}Files:${NC}"
echo "  settings.yaml              — GraphRAG configuration"
echo "  .env                       — API keys and endpoints"
echo "  mcp_config.json            — MCP client configuration (Claude/VS Code)"
echo "  src/hybrid_extractor.py    — Hybrid NLP extractor"
echo "  src/run_graphrag_hybrid.py — CLI wrapper with monkey-patch"
echo "  src/graphrag_mcp_server.py — MCP server for GraphRAG"
echo "  src/build_domain_dictionary.py — Dictionary builder"
echo "  src/build_bilingual_thesaurus.py — Bilingual thesaurus builder"
echo "  build_dictionary.sh        — Dictionary build shortcut"
echo "  build_thesaurus.sh         — Thesaurus build shortcut"
echo "  run_index.sh               — Index build shortcut"
echo "  run_query.sh               — Query shortcut"
echo "  run_mcp_server.sh          — MCP server shortcut"
echo ""
