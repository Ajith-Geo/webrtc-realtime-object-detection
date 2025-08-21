# WebRTC Real-Time Object Detection Demo

A real-time object detection system using WebRTC for video streaming between devices, with YOLOv11 processing either in the browser (WASM) or on a Python server.

## 🎥 Demo Video

[![Demo Video](https://img.shields.io/badge/Watch-Demo%20Video-blue?style=for-the-badge&logo=loom)](https://www.loom.com/share/f5b1039967a8466ea81c81f63e833d30?sid=55400882-a987-4d13-8736-a6147e2addd6)

<div>
    <a href="https://www.loom.com/share/f5b1039967a8466ea81c81f63e833d30?sid=55400882-a987-4d13-8736-a6147e2addd6">
    </a>
</div>

## 📋 Requirements

### System Requirements
#### **NOTE**: PLEASE ENSURE Python, Node.js and Ngrok is INSTALLED before running the install script
- **Python**: 3.12+ 
- **Node.js**: 22+
- **GPU**: NVIDIA GPU with CUDA 11.8+ (for server mode GPU acceleration)
- **RAM**: 16GB recommended
- **OS**: Windows 10/11, macOS, or Linux

### Test Environment
- **CPU**: Intel i7-12700H
- **GPU**: NVIDIA RTX 3060 Mobile (6GB VRAM)
- **RAM**: 16GB
- **OS**: Windows 11
- **Python**: 3.12.9
- **Node.js**: 22.11.0

## 🚀 Quick Start

### One-Command Installation (Windows with Git Bash, Recommended mode of Installation)

```bash
chmod +x setup-and-run.sh
./setup-and-run.sh
```

This script will:
- Verify dependencies (Python, Node.js, ngrok)
- Set up virtual environment
- Install all required packages
- Configure ngrok tunnel
- Launch the application

## 🐛 Troubleshooting

### Common Issues

1. **Video not showing on laptop after scanning QR code**
   - Refresh the page on the phone first
   - If issue persists, refresh laptop page and scan the NEW QR code

2. **Inconsistent metrics results**
   - Stay on the browser for at least 30 seconds after overlay starts
   - This is a known issue that will be fixed in future versions

3. **GPU not detected (Server Mode)**
   - Ensure NVIDIA drivers are installed
   - Verify CUDA 11.8+ installation
   - Check PyTorch CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

### 📊 Viewing Performance Metrics

- **Open Browser Developer Tools** (F12) on the laptop
- **Check the Console tab** to see real-time performance statistics
- **Stay on the browser for at least 30 seconds** after detection starts for accurate metrics
- **Metrics are automatically saved** to `metrics.json` in the root directory

## 📊 Metrics and Benchmarking

The application automatically collects performance metrics including:

- **Latency measurements** (end-to-end, inference, network)
- **Frame rates** (FPS)
- **Processing times** (capture, inference, render)
- **Resource utilization** (CPU, GPU, memory)

Metrics are:
- Displayed in browser console in real-time (only for server mode (expand payload in browser console to check))
- Automatically saved to `metrics.json` after 30 seconds
- Include both summary statistics and raw measurements

## 📱 How to Use

1. **Start the application** using one of the installation methods above
2. **Open the laptop interface** at `https://tomcat-beloved-feline.ngrok-free.app/laptop.html`
3. **Scan the QR code** with your phone to access the mobile interface
4. **Grant camera permissions** on your phone
5. **Watch real-time object detection** on the laptop screen


## ⚙️ Modes

### WASM Mode
- Object detection runs entirely in the browser
- Uses ONNX Runtime Web with Webgpu.... if not, falls back to WebGL, else falls back to WebAssembly(WASM)
- Lower server resource usage
- Suitable for client-side processing when GPU is not available

### Server Mode  
- Object detection runs on Python server
- Supports GPU acceleration (CUDA)
- Higher accuracy and performance potential
- Falls back to CPU if GPU is not available
- CPU and GPU resource consumption capped at 90% and 85% respectively
- Requires more server resources


## 🎯 Features

- **Real-time WebRTC streaming** between devices
- **Dual inference modes:**
  - **WASM Mode**: YOLOv11 runs in browser using ONNX Runtime Web
  - **Server Mode**: YOLOv11 runs on Python server with GPU acceleration
- **QR Code connection** for easy device pairing
- **Live object detection overlay** with bounding boxes
- **Performance metrics collection** (latency, FPS, processing time)
- **Cross-platform support** (Windows, macOS, Linux)

### Manual Installation (Not recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd admybrand
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

3. **Set up Python environment**
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   
   pip install -r python-receiver/requirements.txt
   ```

4. **Start the application**
   
   **WASM Mode (Browser-based inference):**
   ```bash
   npm run start:wasm
   ```
   
   **Server Mode (Python-based inference):**
   ```bash
   # Terminal 1
   npm run start:server
   
   # Terminal 2
   cd python-receiver
   python server.py
   ```




## 🔧 Technical Architecture

### Frontend (Browser)
- **WebRTC** for peer-to-peer video streaming
- **ONNX Runtime Web** for WASM-based inference
- **Canvas API** for real-time overlay rendering
- **WebSocket** signaling for connection establishment

### Backend (Node.js)
- **Express.js** web server
- **WebSocket** signaling server for WebRTC
- **UUID-based room management**
- **Metrics collection and storage**

### Python Server (Server Mode)
- **aiortc** for WebRTC media handling
- **YOLOv11** via Ultralytics for object detection
- **PyTorch** with CUDA support
- **Resource monitoring** and adaptive pipeline








## 🛣️ Further Fixes:

- [ ] Fix 30-second minimum browser time requirement
- [ ] Add support for multiple concurrent connections
- [ ] Implement additional YOLO model variants
- [ ] Enhanced metrics dashboard

## 📝 File Structure

```
├── package.json              # Node.js dependencies and scripts
├── server.js                 # Main WebRTC signaling server
├── setup-and-run.sh         # Automated setup script
├── metrics.json              # Performance metrics (generated)
├── yolo11n.pt               # YOLO model file
├── public/
│   ├── laptop.html          # Laptop/receiver interface
│   ├── phone.html           # Mobile/sender interface
│   └── yolov8n.onnx        # ONNX model for browser
└── python-receiver/
    ├── server.py            # Python WebRTC receiver
    ├── detector.py          # YOLO detection logic
    ├── requirements.txt     # Python dependencies
    └── resource_limiter.py  # Resource monitoring
```

---

**Note**: This is a demonstration project showcasing real-time object detection with WebRTC. For production use, additional security, scalability, and error handling considerations should be implemented.
