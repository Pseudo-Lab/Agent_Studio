#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Kiosk Agent - Vision-Language-Action Agent for Kiosk Automation
#  
#  🏛️  PseudoLab (가짜연구소) 11기 Agent Studio
#  📦  https://github.com/Pseudo-Lab/Agent_Studio
# ═══════════════════════════════════════════════════════════════════════════════

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# PseudoLab 브랜드 컬러
ORANGE='\033[38;5;208m'
BLUE='\033[38;5;33m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# 커서 제어
hide_cursor() { printf '\033[?25l'; }
show_cursor() { printf '\033[?25h'; }
trap 'show_cursor; cleanup' EXIT INT TERM

clear

# ═══════════════════════════════════════════════════════════════════════════════
# KIOSK AGENT 로고
# ═══════════════════════════════════════════════════════════════════════════════
print_logo() {
    echo ""
    echo -e "  ${ORANGE}██╗  ██╗██╗ ██████╗ ███████╗██╗  ██╗${NC}"
    echo -e "  ${ORANGE}██║ ██╔╝██║██╔═══██╗██╔════╝██║ ██╔╝${NC}"
    echo -e "  ${ORANGE}█████╔╝ ██║██║   ██║███████╗█████╔╝${NC} "
    echo -e "  ${ORANGE}██╔═██╗ ██║██║   ██║╚════██║██╔═██╗${NC} "
    echo -e "  ${ORANGE}██║  ██╗██║╚██████╔╝███████║██║  ██╗${NC}"
    echo -e "  ${ORANGE}╚═╝  ╚═╝╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝${NC}"
    echo ""
    echo -e "   ${BLUE} █████╗  ██████╗ ███████╗███╗   ██╗████████╗${NC}"
    echo -e "   ${BLUE}██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝${NC}"
    echo -e "   ${BLUE}███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ${NC}"
    echo -e "   ${BLUE}██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ${NC}"
    echo -e "   ${BLUE}██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ${NC}"
    echo -e "   ${BLUE}╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ${NC}"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# 프로젝트 정보
# ═══════════════════════════════════════════════════════════════════════════════
print_info_box() {
    echo -e "  ${GRAY}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "  ${GRAY}║${NC}                                                               ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}   ${BOLD}${WHITE}Vision-Language-Action Agent for Kiosk Automation${NC}         ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}                                                               ${GRAY}║${NC}"
    echo -e "  ${GRAY}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "  ${GRAY}║${NC}                                                               ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}   ${ORANGE}🏛️  PseudoLab${NC} ${GRAY}(가짜연구소)${NC}                                  ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}   ${BLUE}📚  11기 Agent Studio${NC}                                      ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}                                                               ${GRAY}║${NC}"
    echo -e "  ${GRAY}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "  ${GRAY}║${NC}                                                               ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}   ${BOLD}${WHITE}👥 Team Members${NC}                                            ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}                                                               ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}      ${ORANGE}●${NC} ${WHITE}김승혁${NC} ${GRAY}(@SeungHyeokKim)${NC}   - ${CYAN}namu${NC}                  ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}      ${BLUE}●${NC} ${WHITE}김재현${NC} ${GRAY}(@jh941213)${NC}        - ${CYAN}KTDS${NC}                  ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}      ${ORANGE}●${NC} ${WHITE}이규민${NC} ${GRAY}(@qmin2)${NC}           - ${CYAN}KT${NC}                    ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}      ${BLUE}●${NC} ${WHITE}전민정${NC} ${GRAY}(@ummjevel)${NC}        - ${CYAN}AICESS${NC}                ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}                                                               ${GRAY}║${NC}"
    echo -e "  ${GRAY}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "  ${GRAY}║${NC}                                                               ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}   ${WHITE}📦 GitHub${NC}  ${CYAN}https://github.com/Pseudo-Lab/Agent_Studio${NC}     ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}   ${WHITE}🏠 PseudoLab${NC}  ${CYAN}https://pseudo-lab.com${NC}                      ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}   ${WHITE}📜 License${NC}  ${GRAY}Apache License 2.0${NC}                           ${GRAY}║${NC}"
    echo -e "  ${GRAY}║${NC}                                                               ${GRAY}║${NC}"
    echo -e "  ${GRAY}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# 유틸리티 함수
# ═══════════════════════════════════════════════════════════════════════════════
spinner() {
    local pid=$1
    local msg=$2
    local spinstr='⣾⣽⣻⢿⡿⣟⣯⣷'
    local i=0
    hide_cursor
    while kill -0 $pid 2>/dev/null; do
        local char="${spinstr:$i:1}"
        printf "\r  ${ORANGE}${char}${NC} ${msg}"
        i=$(( (i+1) % 8 ))
        sleep 0.1
    done
    printf "\r"
    show_cursor
}

print_check() { printf "  ${GREEN}✓${NC} ${1}\n"; }
print_error() { printf "  ${RED}✗${NC} ${1}\n"; }
print_info() { printf "  ${BLUE}ℹ${NC} ${1}\n"; }
print_warn() { printf "  ${YELLOW}⚠${NC} ${1}\n"; }

print_section() {
    echo ""
    echo -e "  ${BOLD}${ORANGE}▸${NC} ${BOLD}${WHITE}${1}${NC}"
    echo -e "  ${GRAY}─────────────────────────────────────────────────${NC}"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 프로세스 정리
# ═══════════════════════════════════════════════════════════════════════════════
cleanup() {
    echo ""
    print_section "Shutting Down"
    
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null && print_check "Backend stopped" || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null && print_check "Frontend stopped" || true
    fi
    
    echo ""
    echo -e "  ${ORANGE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${BOLD}${WHITE}👋 Thanks for using Kiosk Agent!${NC}"
    echo -e "  ${GRAY}   Made with ❤️  by PseudoLab Agent Studio${NC}"
    echo -e "  ${ORANGE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    exit 0
}

# ═══════════════════════════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════════════════════════
main() {
    print_logo
    sleep 0.3
    print_info_box
    sleep 0.5
    
    # ───────────────────────────────────────────────────────────────────────────
    print_section "Environment Setup"
    
    # Python 캐시 정리
    find backend -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find backend -name "*.pyc" -delete 2>/dev/null || true
    print_check "Python cache cleared"
    
    # .env 로드
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs) 2>/dev/null
        print_check "Environment variables loaded ${GRAY}(.env)${NC}"
    else
        print_warn "No .env file found"
    fi
    
    # 상대경로를 절대경로로 변환 (Google SDK는 절대경로 필요)
    if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ] && [[ "$GOOGLE_APPLICATION_CREDENTIALS" == ./* ]]; then
        export GOOGLE_APPLICATION_CREDENTIALS="$PROJECT_ROOT/${GOOGLE_APPLICATION_CREDENTIALS#./}"
    fi
    
    # Hugging Face 관련 경고 비활성화 (로깅 일관성)
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    export TQDM_DISABLE=1
    export TOKENIZERS_PARALLELISM=false
    sleep 0.1
    
    # 가상환경 활성화
    VENV_PATH="$PROJECT_ROOT/.venv_mac"
    if [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate"
        print_check "Virtual environment activated ${GRAY}(.venv_mac)${NC}"
    else
        print_error "Virtual environment not found"
        print_info "Run: ${CYAN}uv venv .venv_mac && source .venv_mac/bin/activate && uv pip install -e backend/${NC}"
        exit 1
    fi
    sleep 0.1
    
    # Python 버전
    PYTHON_VER=$(python --version 2>&1)
    print_check "Python ready ${GRAY}(${PYTHON_VER})${NC}"
    sleep 0.1
    
    # ───────────────────────────────────────────────────────────────────────────
    print_section "Dependencies"
    
    # Frontend
    cd "$PROJECT_ROOT/web"
    if [ ! -d "node_modules" ]; then
        npm install --silent &
        spinner $! "Installing npm packages..."
        print_check "npm packages installed"
    else
        print_check "npm packages ready ${GRAY}(node_modules)${NC}"
    fi
    sleep 0.1
    
    # ADB
    if command -v adb &> /dev/null; then
        ADB_DEVICES=$(adb devices | grep -v "List" | grep -v "^$" | wc -l | tr -d ' ')
        if [ "$ADB_DEVICES" -gt "0" ]; then
            print_check "ADB connected ${GRAY}(${ADB_DEVICES} device)${NC}"
        else
            print_warn "ADB ready, no devices ${GRAY}(adb connect <IP>:5555)${NC}"
        fi
    else
        print_warn "ADB not installed"
    fi
    sleep 0.1
    
    # ───────────────────────────────────────────────────────────────────────────
    print_section "Starting Services"
    
    echo ""
    echo -e "  ${GRAY}┌─────────────────────────────────────────────────────┐${NC}"
    echo -e "  ${GRAY}│${NC}                                                     ${GRAY}│${NC}"
    echo -e "  ${GRAY}│${NC}   ${ORANGE}▶${NC} ${BOLD}Backend${NC}   ${WHITE}http://localhost:8080${NC}              ${GRAY}│${NC}"
    echo -e "  ${GRAY}│${NC}   ${BLUE}▶${NC} ${BOLD}Frontend${NC}  ${WHITE}http://localhost:3000${NC}              ${GRAY}│${NC}"
    echo -e "  ${GRAY}│${NC}                                                     ${GRAY}│${NC}"
    echo -e "  ${GRAY}└─────────────────────────────────────────────────────┘${NC}"
    echo ""
    
    # Backend 실행
    cd "$PROJECT_ROOT/backend"
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 2>&1 | while IFS= read -r line; do
        echo -e "  ${ORANGE}│${NC} ${GRAY}[API]${NC} $line"
    done &
    BACKEND_PID=$!
    sleep 1.5
    
    # Frontend 실행
    cd "$PROJECT_ROOT/web"
    npm run dev 2>&1 | while IFS= read -r line; do
        echo -e "  ${BLUE}│${NC} ${GRAY}[WEB]${NC} $line"
    done &
    FRONTEND_PID=$!
    
    echo ""
    echo -e "  ${YELLOW}⌨️  Press ${BOLD}Ctrl+C${NC}${YELLOW} to stop all services${NC}"
    echo ""
    echo -e "  ${GRAY}═══════════════════════════════════════════════════════${NC}"
    echo ""
    
    wait
}

# ═══════════════════════════════════════════════════════════════════════════════
# 실행
# ═══════════════════════════════════════════════════════════════════════════════
main
