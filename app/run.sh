#!/bin/bash

# Mobile-Selection-Eval One-click Startup Script
# Start web applications and backend services

set -e  # Exit immediately on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}$1${NC}"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Service configuration
BACKEND_DIR="$PROJECT_ROOT/backend"
WEB_APP_DIR="$PROJECT_ROOT/web_app"
SHOPPING_STORE_DIR="$WEB_APP_DIR/shopping-store-vue"
MEITUAN_DIR="$WEB_APP_DIR/vue-meituan"

# Port configuration
BACKEND_PORT=8000
SHOPPING_STORE_PORT=5173
MEITUAN_PORT=8080

# PID file directory
PID_DIR="$PROJECT_ROOT/.pids"
mkdir -p "$PID_DIR"

# Cleanup function
cleanup() {
    log_header "=== Cleaning Processes ==="
    
    # Kill all child processes
    if [[ -f "$PID_DIR/backend.pid" ]]; then
        BACKEND_PID=$(cat "$PID_DIR/backend.pid")
        if kill -0 "$BACKEND_PID" 2>/dev/null; then
            log_info "Stopping backend service (PID: $BACKEND_PID)"
            kill "$BACKEND_PID"
        fi
        rm -f "$PID_DIR/backend.pid"
    fi
    
    if [[ -f "$PID_DIR/shopping-store.pid" ]]; then
        STORE_PID=$(cat "$PID_DIR/shopping-store.pid")
        if kill -0 "$STORE_PID" 2>/dev/null; then
            log_info "Stopping Shopping Store service (PID: $STORE_PID)"
            kill "$STORE_PID"
        fi
        rm -f "$PID_DIR/shopping-store.pid"
    fi
    
    if [[ -f "$PID_DIR/meituan.pid" ]]; then
        MEITUAN_PID=$(cat "$PID_DIR/meituan.pid")
        if kill -0 "$MEITUAN_PID" 2>/dev/null; then
            log_info "Stopping Meituan service (PID: $MEITUAN_PID)"
            kill "$MEITUAN_PID"
        fi
        rm -f "$PID_DIR/meituan.pid"
    fi
    
    # Clean up port usage
    for port in $BACKEND_PORT $SHOPPING_STORE_PORT $MEITUAN_PORT; do
        PID=$(lsof -ti:$port 2>/dev/null || true)
        if [[ -n "$PID" ]]; then
            log_info "Cleaning port $port usage (PID: $PID)"
            kill -9 "$PID" 2>/dev/null || true
        fi
    done
    
    log_success "Cleanup completed"
    exit 0
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Check if port is occupied
check_port() {
    local port=$1
    local service_name=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_error "Port $port is occupied, cannot start $service_name"
        log_info "Please run the following command to clean port usage:"
        log_info "sudo lsof -ti:$port | xargs kill -9"
        return 1
    fi
    return 0
}

# Check dependencies
check_dependencies() {
    log_header "=== Checking Dependencies ==="
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is not installed"
        exit 1
    fi
    log_success "Python3: $(python3 --version)"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed"
        exit 1
    fi
    log_success "Node.js: $(node --version)"
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        log_error "npm is not installed"
        exit 1
    fi
    log_success "npm: $(npm --version)"
    
    # Check lsof (for port checking)
    if ! command -v lsof &> /dev/null; then
        log_warning "lsof is not installed, skipping port check"
    fi
}

# Install backend dependencies
install_backend_deps() {
    log_header "=== Installing Backend Dependencies ==="
    
    if [[ ! -d "$BACKEND_DIR" ]]; then
        log_error "Backend directory does not exist: $BACKEND_DIR"
        exit 1
    fi
    
    cd "$BACKEND_DIR"
    
    # Check if requirements.txt exists
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing backend Python dependencies..."
        pip3 install -r requirements.txt
    else
        log_info "Installing default backend dependencies..."
        pip3 install fastapi uvicorn python-multipart
    fi
    
    log_success "Backend dependencies installation completed"
}

# Install frontend dependencies
install_frontend_deps() {
    log_header "=== Installing Frontend Dependencies ==="
    
    # Install Shopping Store dependencies
    if [[ -d "$SHOPPING_STORE_DIR" ]]; then
        log_info "Installing Shopping Store dependencies..."
        cd "$SHOPPING_STORE_DIR"
        if [[ ! -d "node_modules" ]]; then
            npm install
        else
            log_info "Shopping Store dependencies already exist, skipping installation"
        fi
        log_success "Shopping Store dependencies installation completed"
    else
        log_warning "Shopping Store directory does not exist, skipping installation"
    fi
    
    # Install Meituan dependencies
    if [[ -d "$MEITUAN_DIR" ]]; then
        log_info "Installing Meituan dependencies..."
        cd "$MEITUAN_DIR"
        if [[ ! -d "node_modules" ]]; then
            npm install
        else
            log_info "Meituan dependencies already exist, skipping installation"
        fi
        log_success "Meituan dependencies installation completed"
    else
        log_warning "Meituan directory does not exist, skipping installation"
    fi
}

# Start backend service
start_backend() {
    log_header "=== Starting Backend Service ==="
    
    if ! check_port $BACKEND_PORT "Backend"; then
        exit 1
    fi
    
    cd "$BACKEND_DIR"
    
    # Start FastAPI service
    log_info "Starting backend service on port $BACKEND_PORT..."
    nohup uvicorn main:app --host 0.0.0.0 --port $BACKEND_PORT --reload > "$PROJECT_ROOT/backend.log" 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > "$PID_DIR/backend.pid"
    
    # Wait for service to start
    sleep 3
    
    # Check if service started normally
    if kill -0 $BACKEND_PID 2>/dev/null; then
        log_success "Backend service started successfully (PID: $BACKEND_PID, Port: $BACKEND_PORT)"
        log_info "Backend log: $PROJECT_ROOT/backend.log"
        log_info "Backend access URL: http://localhost:$BACKEND_PORT"
    else
        log_error "Backend service startup failed"
        exit 1
    fi
}

# Start Shopping Store
start_shopping_store() {
    log_header "=== Starting Shopping Store ==="
    
    if [[ ! -d "$SHOPPING_STORE_DIR" ]]; then
        log_warning "Shopping Store directory does not exist, skipping startup"
        return
    fi
    
    if ! check_port $SHOPPING_STORE_PORT "Shopping Store"; then
        exit 1
    fi
    
    cd "$SHOPPING_STORE_DIR"
    
    log_info "Starting Shopping Store on port $SHOPPING_STORE_PORT..."
    nohup npm run dev -- --host > "$PROJECT_ROOT/shopping-store.log" 2>&1 &
    STORE_PID=$!
    echo $STORE_PID > "$PID_DIR/shopping-store.pid"
    
    # Wait for service to start
    sleep 5
    
    # Check if service started normally
    if kill -0 $STORE_PID 2>/dev/null; then
        log_success "Shopping Store started successfully (PID: $STORE_PID, Port: $SHOPPING_STORE_PORT)"
        log_info "Shopping Store log: $PROJECT_ROOT/shopping-store.log"
        log_info "Shopping Store access URL: http://localhost:$SHOPPING_STORE_PORT"
    else
        log_error "Shopping Store startup failed"
        exit 1
    fi
}

# Start Meituan
start_meituan() {
    log_header "=== Starting Meituan ==="
    
    if [[ ! -d "$MEITUAN_DIR" ]]; then
        log_warning "Meituan directory does not exist, skipping startup"
        return
    fi
    
    if ! check_port $MEITUAN_PORT "Meituan"; then
        exit 1
    fi
    
    cd "$MEITUAN_DIR"
    
    log_info "Starting Meituan on port $MEITUAN_PORT..."
    nohup npm run dev -- --host > "$PROJECT_ROOT/meituan.log" 2>&1 &
    MEITUAN_PID=$!
    echo $MEITUAN_PID > "$PID_DIR/meituan.pid"
    
    # Wait for service to start
    sleep 5
    
    # Check if service started successfully
    if kill -0 $MEITUAN_PID 2>/dev/null; then
        log_success "Meituan started successfully (PID: $MEITUAN_PID, Port: $MEITUAN_PORT)"
        log_info "Meituan log: $PROJECT_ROOT/meituan.log"
        log_info "Meituan access URL: http://localhost:$MEITUAN_PORT"
    else
        log_error "Failed to start Meituan"
        exit 1
    fi
}

# Show service status
show_status() {
    log_header "=== 服务状态 ==="
    
    echo -e "${GREEN}🚀 所有服务启动完成！${NC}"
    echo ""
    echo -e "${BLUE}📋 服务列表：${NC}"
    echo -e "  • 后端服务:      http://localhost:$BACKEND_PORT"
    echo -e "  • Shopping Store: http://localhost:$SHOPPING_STORE_PORT"
    echo -e "  • Meituan:       http://localhost:$MEITUAN_PORT"
    echo ""
    echo -e "${BLUE}📝 日志文件：${NC}"
    echo -e "  • 后端日志:      $PROJECT_ROOT/backend.log"
    echo -e "  • Shopping Store: $PROJECT_ROOT/shopping-store.log"
    echo -e "  • Meituan:       $PROJECT_ROOT/meituan.log"
    echo ""
    echo -e "${YELLOW}⚠️  按 Ctrl+C 停止所有服务${NC}"
}

# 主函数
main() {
    log_header "=== Mobile-Selection-Eval 启动脚本 ==="
    
    # 检查依赖
    check_dependencies
    
    # 安装依赖
    install_backend_deps
    install_frontend_deps
    
    # 启动服务
    start_backend
    start_shopping_store
    start_meituan
    
    # 显示状态
    show_status
    
    # 保持脚本运行
    log_info "所有服务正在运行中..."
    log_info "使用 'tail -f *.log' 查看实时日志"
    
    # 等待中断信号
    while true; do
        sleep 1
    done
}

# 处理命令行参数
case "${1:-}" in
    "stop")
        log_info "停止所有服务..."
        cleanup
        ;;
    "status")
        log_header "=== 服务状态检查 ==="
        for port in $BACKEND_PORT $SHOPPING_STORE_PORT $MEITUAN_PORT; do
            if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
                echo -e "${GREEN}✓${NC} 端口 $port 正在使用中"
            else
                echo -e "${RED}✗${NC} 端口 $port 未使用"
            fi
        done
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  start     启动所有服务 (默认)"
        echo "  stop      停止所有服务"
        echo "  status    检查服务状态"
        echo "  help      显示帮助信息"
        echo ""
        echo "服务端口："
        echo "  后端服务:      $BACKEND_PORT"
        echo "  Shopping Store: $SHOPPING_STORE_PORT"
        echo "  Meituan:       $MEITUAN_PORT"
        ;;
    "start"|"")
        main
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' to show help information"
        exit 1
        ;;
esac