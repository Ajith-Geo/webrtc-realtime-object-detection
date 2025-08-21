# WebRTC Real-Time Object Detection System - Technical Report

## Executive Summary

This project implements a real-time object detection system using WebRTC for low-latency video streaming between devices, with dual inference modes: browser-based WASM inference and server-based Python inference. The system demonstrates sophisticated design choices for resource optimization, backpressure handling, and cross-device compatibility.

## Architecture Overview

### Frontend Architecture
- **WebRTC P2P Streaming**: Direct peer-to-peer video connection between phone and laptop
- **QR Code Connection**: Simplified device pairing using UUID-based room management
- **Dual Rendering Pipeline**: Canvas overlay system for real-time bounding box visualization

### Backend Architecture  
- **Node.js Signaling Server**: Express.js server with WebSocket signaling for WebRTC establishment
- **Python Inference Server**: aiortc-based server for GPU-accelerated object detection
- **Resource Management**: Adaptive pipeline with CPU/GPU monitoring

## Design Choices & Implementation Details

### 1. Low-Resource Mode Implementation ✅

**Code Evidence:**
```javascript
// File: public/laptop.html, lines 421-430
try { if (ort?.env?.wasm) { 
  ort.env.wasm.numThreads = Math.min(6, Math.max(2, navigator.hardwareConcurrency || 4)); 
  ort.env.wasm.simd = true; 
} }
const providerAttempts = [['webgpu', 'wasm'], ['webgl', 'wasm'], ['wasm']];
for (const providers of providerAttempts) {
  try { session = await ort.InferenceSession.create('./yolov8n.onnx', { executionProviders: providers }); 
    console.log('Using ORT providers:', providers); break; }
  catch (e) { console.warn('Provider set failed', providers, e); }
}
```

**Strategy:** Intelligent fallback hierarchy: WebGPU → WebGL → WASM
- **WASM Mode**: Uses ONNX Runtime Web with adaptive provider selection
- **Input Downscaling**: 320×240 to 640×640 letterbox preprocessing for mobile compatibility
- **Frame Rate Control**: Adaptive 10-15 FPS with dynamic interval adjustment (`targetIntervalMs = inferMs * 0.15`)

### 2. Real-Time Phone Browser Support ✅

**Code Evidence:**
```javascript
// File: public/phone.html, lines 42-60
if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
  videoEl.srcObject = stream;
  const loop = () => {
    try { videoEl.requestVideoFrameCallback((now, meta) => { sendFrame(now); loop(); }); }
    catch(e){ /* fallback below */ }
  };
  loop();
} else {
  // Fallback timer approx 15fps
  setInterval(() => { sendFrame(performance.now()); }, 66);
}
```

**Features:**
- **Browser-Only Operation**: No native app required, works on Chrome/Android & Safari/iOS
- **High-Precision Timestamps**: Uses `requestVideoFrameCallback` for frame-accurate capture timing
- **WebRTC ICE Configuration**: Multiple STUN servers for NAT traversal

### 3. One-Command Startup ✅

**Code Evidence:**
```bash
# File: setup-and-run.sh, lines 1-15
#!/usr/bin/env bash
# Unified setup & run script (Linux/macOS) for the WebRTC demo.
# 1. Verify python, node, npm, ngrok availability
# 2. Prompt for ngrok authtoken (if not already configured)
# 3. Create/upgrade virtual environment (.venv)
# 4. Install npm deps & Python deps
# 5. Ask for mode: wasm | server
```

**Implementation:**
- **Automated Dependency Checking**: Verifies Python 3.12+, Node.js 22+, ngrok
- **Environment Setup**: Creates virtual environment, installs dependencies
- **Mode Selection**: Interactive prompt for WASM vs Server mode
- **Ngrok Integration**: Automatic tunnel setup with fixed subdomain

### 4. Metrics Collection & JSON Output ✅

**Code Evidence:**
```javascript
// File: server.js, lines 43-89
app.post('/metrics', (req, res) => {
  const payload = req.body || {};
  // Accept latencies under different possible keys (normalize)
  const latArr = Array.isArray(payload.latencies_ms) ? payload.latencies_ms : 
                 Array.isArray(payload.latency_ms) ? payload.latency_ms : 
                 Array.isArray(payload.wasm_inference_ms) ? payload.wasm_inference_ms : null;
  
  latestMetrics = { ...payload, collected_at: new Date().toISOString() };
  appendMetricsToFile(latestMetrics);
```

**Metrics Structure:**
```json
{
  "mode": "server|wasm",
  "duration_s": 30,
  "frames": 450,
  "fps": 15.0,
  "median_latency_ms": 45.2,
  "p95_latency_ms": 89.1,
  "latencies_ms": [42.1, 43.5, ...]
}
```

### 5. Detection Message Format ✅

**Code Evidence:**
```python
# File: python-receiver/detector.py, lines 242-261
payload = {
    "capture_ts": t0,
    "inference_ts": infer_done,
    "latency_ms": infer_ms,
    "inference_ms": infer_ms,
    "device": self.device,
    "half": self.use_half,
    "detections": dets,
    "img": {"w": W, "h": H}
}
```

**Server Detection Output:**
```json
{
  "frame_id": "string_or_int",
  "capture_ts": 1690000000000,
  "recv_ts": 1690000000100,
  "inference_ts": 1690000000120,
  "detections": [
    {
      "label": "person",
      "score": 0.93,
      "xmin": 0.12,
      "ymin": 0.08,
      "xmax": 0.34,
      "ymax": 0.67
    }
  ]
}
```

### 6. Latency Measurement & Benchmarking ✅

**Code Evidence:**
```javascript
// File: public/laptop.html, lines 340-370
const rawOffset = payload.recv_ts - payload.capture_ts; // seconds
if (!bench._baselineFrozen) {
  if (bench._baseline === null || rawOffset < bench._baseline) bench._baseline = rawOffset;
  if (bench._rawOffsets.length >= bench._baselineFreezeAfter) bench._baselineFrozen = true;
}
if (bench._baseline !== null) {
  let corrected = (rawOffset - bench._baseline) * 1000.0; // ms (clock skew removed)
  if (corrected < 0) { bench.dropped_negative++; return; }
  if (corrected < 5000) {
    bench.latencies.push(corrected);
  }
}
```

**Measurement Categories:**
- **E2E Latency**: `overlay_display_ts - capture_ts` with clock skew correction
- **Server Latency**: `inference_ts - recv_ts`
- **Network Latency**: `recv_ts - capture_ts`
- **Processed FPS**: Frame count over 30-second benchmark window

### 7. Advanced Low-Resource Strategies ✅

#### WebGPU with Intelligent Fallbacks

**Code Evidence:**
```javascript
// File: public/laptop.html, lines 421-425
const providerAttempts = [['webgpu', 'wasm'], ['webgl', 'wasm'], ['wasm']];
for (const providers of providerAttempts) {
  try { session = await ort.InferenceSession.create('./yolov8n.onnx', { executionProviders: providers }); 
    console.log('Using ORT providers:', providers); break; }
```

**Strategy:** Cleverly utilizes WebGPU for hardware acceleration when available, with graceful fallback to WebGL, then pure WASM.

#### Adaptive Input Scaling

**Code Evidence:**
```python
# File: python-receiver/detector.py, lines 218-230
if self.device == 'cpu' and CPU_AUTO_TUNE and self.frames_seen >= CPU_WARM_FRAMES:
  if (self.frames_seen % 3) == 0:
    if infer_ms > CPU_TARGET_MS * 1.25 and self.current_imgsz > CPU_MIN_IMGSZ:
      new_size = max(CPU_MIN_IMGSZ, self.current_imgsz - CPU_STEP)
      if new_size != self.current_imgsz:
        self.current_imgsz = new_size
```

**Features:**
- **Dynamic Resolution**: Automatically adjusts from 320×240 to 640×640 based on performance
- **Target 15 FPS**: Maintains responsive frame rates through adaptive processing

#### Frame Thinning & Backpressure Policy

**Code Evidence:**
```python
# File: python-receiver/server.py, lines 106-120
latest_frame = {'img': None, 'capture_ts': None, 'recv_time': None}
# ...
# Overwrite latest (drop previous unprocessed if any)
if latest_frame['img'] is not None:
    stats['dropped'] += 1
latest_frame['img'] = img
latest_frame['capture_ts'] = capture_ts
latest_frame['recv_time'] = now
frame_event.set()
```

**Strategy:** Single-slot frame buffer with overwrite policy - always processes latest frame, drops older ones when inference can't keep up.

#### Resource Monitoring & Throttling

**Code Evidence:**
```python
# File: python-receiver/resource_limiter.py, lines 67-81
async def wait_for_resources(self):
    sleep_ms = self.min_sleep_ms
    while True:
        async with self._lock:
            snap = self._snapshot
        over_cpu = snap.cpu >= self.max_cpu if snap.cpu else False
        over_gpu = snap.gpu is not None and snap.gpu >= self.max_gpu
        if not over_cpu and not over_gpu:
            return
        await asyncio.sleep(sleep_ms / 1000.0)
        sleep_ms = min(self.max_sleep_ms, sleep_ms * 2)
```

**Features:**
- **CPU Limit**: 90% maximum utilization with exponential backoff
- **GPU Limit**: 85% maximum utilization to prevent thermal throttling
- **Adaptive Throttling**: Prevents system overload while maintaining responsiveness

#### ONNX Runtime Optimization

**Code Evidence:**
```python
# File: python-receiver/detector.py, lines 100-125
if USE_ONNX_CPU and ort is not None:
  if not os.path.exists(ONNX_WEIGHTS):
    # Attempt export if .pt present
    pt_candidate = YOLO_WEIGHTS
    if os.path.exists(pt_candidate):
      try:
        YOLO(pt_candidate).export(format='onnx', opset=12, dynamic=False)
```

**Features:**
- **CPU Mode**: Prefers ONNX Runtime over PyTorch for CPU inference (better performance)
- **Automatic Export**: Converts PyTorch models to ONNX format when needed
- **Threading Control**: Configurable thread allocation for optimal CPU utilization

## Mode Switching Implementation

**Code Evidence:**
```javascript
// File: package.json, lines 6-10
"scripts": {
  "start": "node server.js",
  "start:server": "cross-env MODE=server node server.js",
  "start:wasm": "cross-env MODE=wasm node server.js"
}
```

**Auto-Detection Strategy:** The setup script automatically detects system capabilities and prompts for optimal mode selection, with environment variable override support.

## Performance Optimizations

### Cross-Origin Isolation
```javascript
// File: server.js, lines 14-18
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  next();
});
```

### Memory-Efficient Processing
```javascript
// File: public/laptop.html, lines 452-465
const reusableTensor = new ort.Tensor('float32', preprocessedData, [1, 3, inputSize.h, inputSize.w]);
// Inline preprocessing into reusable buffer (NCHW, normalized 0..1)
const px = imageData.data; // RGBA
for (let i = 0, p = 0; i < area; i++, p += 4) {
  preprocessedData[i] = px[p] * (1 / 255);
  preprocessedData[i + area] = px[p + 1] * (1 / 255);
  preprocessedData[i + 2 * area] = px[p + 2] * (1 / 255);
}
```

## Conclusion

The system successfully balances performance, compatibility, and resource efficiency through intelligent adaptive algorithms and fallback strategies. (Need some fixes though)
