const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PORT = process.env.PORT || 3000;
const MODE = process.env.MODE || 'server'; // 'server' or 'wasm'

// Cross-origin isolation headers (enable WASM SIMD + multi-thread if supported)
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  next();
});

// Suppress ngrok free-tier browser warning interstitial (avoids extra refresh + CSP noise)
app.use((req, res, next) => {
  res.setHeader('ngrok-skip-browser-warning', 'true');
  next();
});

// Serve static frontend files from public/
app.use(express.static(path.join(__dirname, 'public')));

// Allow JSON bodies (needed for /metrics collection)
app.use(express.json({ limit: '1mb' }));

// Endpoint to get the current mode
app.get('/mode', (req, res) => {
  res.json({ mode: MODE });
});

// Simple in-memory mapping of roomId -> websocket connections
const rooms = new Map();

// In-memory latest benchmarking metrics (overwritten each run)
let latestMetrics = null; // { mode, summary, raw, collected_at }
const METRICS_FILE = path.join(__dirname, 'metrics.json');

async function appendMetricsToFile(entry) {
  try {
    let arr = [];
    if (fs.existsSync(METRICS_FILE)) {
      try {
        const raw = await fs.promises.readFile(METRICS_FILE, 'utf8');
        if (raw.trim()) {
          const parsed = JSON.parse(raw);
          if (Array.isArray(parsed)) arr = parsed; else if (parsed && typeof parsed === 'object') arr = [parsed];
        }
      } catch (e) {
        console.warn('metrics.json parse error, resetting file', e.message);
      }
    }
    arr.push(entry);
    // Optional cap to avoid unbounded growth
    if (arr.length > 200) arr = arr.slice(-200);
    await fs.promises.writeFile(METRICS_FILE, JSON.stringify(arr, null, 2));
  } catch (e) {
    console.error('Failed writing metrics.json', e);
  }
}

app.post('/metrics', (req, res) => {
  try {
    const payload = req.body || {};
    let reason = null;

    if (!payload || typeof payload !== 'object') reason = 'payload_not_object';
    else if (!payload.mode) reason = 'missing_mode';

    // Accept latencies under different possible keys (normalize)
    const latArr = Array.isArray(payload.latencies_ms)
      ? payload.latencies_ms
      : Array.isArray(payload.latency_ms)
      ? payload.latency_ms
      : Array.isArray(payload.wasm_inference_ms)
      ? payload.wasm_inference_ms
      : null;

    if (!latArr) reason = reason || 'missing_latencies_array';

    if (reason) {
      console.warn('Rejecting metrics:', reason, payload && Object.keys(payload));
      return res.status(400).json({ error: 'Invalid metrics payload', reason });
    }

    // Normalize field name
    payload.latencies_ms = latArr;

    // Backfill median_latency_ms if omitted
    if (typeof payload.median_latency_ms !== 'number' && latArr.length) {
      const sorted = [...latArr].filter(v => Number.isFinite(v)).sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      payload.median_latency_ms =
        sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }

  latestMetrics = { ...payload, collected_at: new Date().toISOString() };
  appendMetricsToFile(latestMetrics); // fire and forget
    console.log('Stored metrics summary', {
      mode: payload.mode,
      frames: payload.frames,
      duration_s: payload.duration_s,
      fps: payload.fps,
      lat_samples: payload.latencies_ms.length,
    });

    return res.json({ status: 'ok' });
  } catch (e) {
    console.error('Failed to store metrics', e);
    return res.status(500).json({ error: 'Server error' });
  }
});

app.get('/metrics', (req, res) => {
  if (!latestMetrics) return res.status(404).json({ error: 'No metrics yet' });
  res.json(latestMetrics);
});

// Proxy QR code generation (to bypass COEP restrictions)
app.get('/qr', async (req, res) => {
  try {
    const data = encodeURIComponent(req.query.data || '');
    if (!data) return res.status(400).send('missing data');

    const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${data}`;
    const response = await fetch(qrUrl);
    if (!response.ok) return res.status(502).send('upstream error');

    res.setHeader('Content-Type', response.headers.get('content-type') || 'image/png');
    const buf = await response.arrayBuffer();
    res.end(Buffer.from(buf));
  } catch (e) {
    console.error('QR proxy error', e);
    res.status(500).send('qr error');
  }
});

wss.on('connection', (ws) => {
  ws.on('message', (msg) => {
    let data;
    try {
      data = JSON.parse(msg);
    } catch (e) {
      console.warn('Invalid JSON', e);
      return;
    }

    // Handle room creation
    if (data.type === 'create') {
      const roomId = uuidv4();
      rooms.set(roomId, { host: ws, guest: null });
      ws.roomId = roomId;
      ws.isHost = true;
      ws.send(JSON.stringify({ type: 'created', roomId }));
      console.log('Room created', roomId);
      return;
    }

    // Handle joining room
    if (data.type === 'join') {
      const room = rooms.get(data.roomId);
      if (room && room.host) {
        room.guest = ws;
        ws.roomId = data.roomId;
        ws.isHost = false;

        // Cross-link peers for forwarding messages
        ws.peer = room.host;
        room.host.peer = ws;

        // Forward offer from guest to host
        room.host.send(JSON.stringify({ type: 'join-request', offer: data.offer }));
        console.log('Forwarded join-request for room', data.roomId);
        return;
      } else {
        ws.send(JSON.stringify({ type: 'error', message: 'Room not found' }));
        return;
      }
    }

    // Handle answer from host
    if (data.type === 'answer') {
      const room = rooms.get(data.roomId);
      if (room && room.guest) {
        room.guest.send(JSON.stringify({ type: 'answer', answer: data.answer }));
        console.log('Forwarded answer for room', data.roomId);
        return;
      }
    }

    // Handle ICE candidates forwarding
    if (data.type === 'ice-candidate') {
      const room = rooms.get(data.roomId);
      if (!room) return;
      if (ws.isHost) {
        if (room.guest) room.guest.send(JSON.stringify({ type: 'ice-candidate', candidate: data.candidate }));
        console.log('Forwarded ICE candidate host->guest for room', data.roomId);
      } else {
        if (room.host) room.host.send(JSON.stringify({ type: 'ice-candidate', candidate: data.candidate }));
        console.log('Forwarded ICE candidate guest->host for room', data.roomId);
      }
      return;
    }
  });

  ws.on('close', () => {
    if (ws.roomId) {
      const room = rooms.get(ws.roomId);
      if (!room) return;

      if (ws.isHost) {
        // Notify guest and close room if host disconnected
        if (room.guest) {
          try { room.guest.send(JSON.stringify({ type: 'error', message: 'Host disconnected' })); } catch (e) {}
          try { room.guest.close(); } catch (e) {}
        }
        rooms.delete(ws.roomId);
        console.log('Host closed, room removed', ws.roomId);
      } else {
        // Guest disconnected, clear guest ref
        room.guest = null;
        if (room.host) room.host.peer = null;
        console.log('Guest disconnected from', ws.roomId);
      }
    }
  });
});

server.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});
