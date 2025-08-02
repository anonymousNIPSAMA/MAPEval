# Service Startup Script Usage Guide

## Description

`run.sh` is a one-click startup script used to simultaneously start the following services:

- **Backend Service** (FastAPI): Port 8000
- **Shopping Store** (Vue3 + Vite): Port 5173  
- **Meituan** (Vue2): Port 8080

## Usage

### Start All Services
```bash
./run.sh
# or
./run.sh start
```

### Stop All Services
```bash
./run.sh stop
```

### Check Service Status
```bash
./run.sh status
```

### View Help Information
```bash
./run.sh help
```

## Service Access URLs

After successful startup, you can access each service through the following URLs:

- Backend API: http://localhost:8000
- Shopping Store: http://localhost:5173
- Meituan: http://localhost:8080

## Log Files

The script will generate the following log files in the current directory:

- `backend.log` - Backend service log
- `shopping-store.log` - Shopping Store log
- `meituan.log` - Meituan service log

## Dependencies

The script will automatically check and install the following dependencies:

### System Dependencies
- Python 3.x
- Node.js
- npm
- lsof (optional, for port checking)

### Python Dependencies
- fastapi
- uvicorn
- python-multipart

### Node.js Dependencies
Will automatically install dependencies defined in each project's package.json

## Features

- ✅ **Smart Dependency Check**: Automatically check system dependencies
- ✅ **Auto Install Dependencies**: Automatically install required dependencies on first run
- ✅ **Port Conflict Detection**: Check port usage before startup
- ✅ **Process Management**: Support graceful startup and shutdown
- ✅ **Real-time Logging**: Independent log recording for each service
- ✅ **Status Monitoring**: Real-time service status checking
- ✅ **Error Handling**: Comprehensive error handling and prompts
- ✅ **Colored Output**: User-friendly command line interface

## Troubleshooting

### Port Already in Use
If you encounter port occupation errors, you can:

1. Use the script to stop services: `./run.sh stop`
2. Manually clear ports:
   ```bash
   # Check port usage
   lsof -i :8000
   lsof -i :5173
   lsof -i :8080
   
   # Force kill processes
   sudo lsof -ti:port_number | xargs kill -9
   ```

### Dependency Installation Failed
1. Ensure network connection is normal
2. Manually install Python dependencies:
   ```bash
   cd backend
   pip3 install -r requirements.txt
   ```
3. Manually install Node.js dependencies:
   ```bash
   cd web_app/shopping-store-vue
   npm install
   
   cd ../vue-meituan
   npm install
   ```

### Service Startup Failed
1. Check the corresponding log files
2. Ensure project directory structure is correct
3. Check if configuration files exist

## Advanced Usage

### Run in Background
```bash
nohup ./run.sh > run.log 2>&1 &
```

### View Real-time Logs
```bash
# View all logs
tail -f *.log

# View specific service logs
tail -f backend.log
tail -f shopping-store.log
tail -f meituan.log
```

### Custom Ports
To modify ports, edit the port configuration in the `run.sh` file:
```bash
BACKEND_PORT=8000
SHOPPING_STORE_PORT=5173
MEITUAN_PORT=8080
```

## Notes

1. First run will be slower as dependencies need to be installed
2. Ensure sufficient disk space to store node_modules
3. Recommended to execute script in project root directory
4. Use Ctrl+C to gracefully stop all services
