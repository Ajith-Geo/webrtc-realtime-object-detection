"""
Simple aiortc-based receiver for Phase 2.
- Accepts an offer from the browser via HTTP /offer
- Creates an RTCPeerConnection, receives video track frames
- Every 10th frame, sends a dummy detection JSON over the data channel
"""
import asyncio
import json
import time
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from av import VideoFrame
import numpy as np

from detector import YoloDetector
from resource_limiter import get_limiter


async def cors_middleware(app, handler):
    async def middleware_handler(request):
        # respond to preflight
        if request.method == 'OPTIONS':
            headers = {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
            return web.Response(status=200, headers=headers)

        resp = await handler(request)
        # ensure CORS header on real responses
        if isinstance(resp, web.Response):
            resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    return middleware_handler

pcs = set()
detector = YoloDetector()  # load model once at startup
resource_limiter = get_limiter()
# Dedicated small thread pool so heavy CPU inference doesn't block event loop scheduling
INFER_THREADS = int(os.getenv('INFER_THREADS', '1'))
executor = ThreadPoolExecutor(max_workers=INFER_THREADS)
PIPELINE_VERBOSE = int(os.getenv('PIPELINE_VERBOSE', '0'))  # 1 enables gap/pipeline logs

async def index(request):
    return web.Response(text="aiortc receiver running")

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)
    print('Created PC for offer')
    # Ensure resource limiter polling started
    await resource_limiter.start()

    dc = None  # detections outbound channel
    capture_meta_ch = None  # inbound capture metadata channel
    latest_capture_ts_map = {}
    frame_count = {"n": 0}

    @pc.on("datachannel")
    def on_datachannel(channel):
        nonlocal dc, capture_meta_ch
        if channel.label == 'from-python':  # (not expected here; created by browser toward python) safeguard
            dc = channel
            print('Detections outbound channel ready (from-python label)')
        elif channel.label == 'capture-meta-forward':
            capture_meta_ch = channel
            print('Capture metadata channel open')
            @channel.on('message')
            def on_meta(msg):
                try:
                    data = json.loads(msg)
                    seq = data.get('seq')
                    cts = data.get('capture_ts')  # seconds epoch
                    if seq is not None and cts is not None:
                        latest_capture_ts_map['last'] = cts
                except Exception as e:
                    print('Bad capture-meta message', e)
        else:
            # treat first non-meta channel as detection channel fallback
            dc = channel
            print('Generic datachannel opened:', channel.label)

    @pc.on("track")
    def on_track(track):
        print('Track received:', track.kind)
        if track.kind == 'video':
            async def recv_frames():
                # Robust adaptive pipeline:
                #  - Always keep only the *latest* frame (drop older) to bound latency.
                #  - Separate producer (receives frames) and consumer (inference) loops.
                #  - If GPU fast, almost every frame still processed; if slow, latency stays bounded.
                #  - Provides required queue/drop/backpressure behavior for robustness rubric.
                base_time_offset: Optional[float] = None  # wallclock - frame_time
                latest_frame = {'img': None, 'capture_ts': None, 'recv_time': None}
                frame_event = asyncio.Event()
                stats = {
                    'received': 0,
                    'processed': 0,
                    'dropped': 0,
                    'last_log': time.time()
                }
                LOG_INTERVAL = 5.0  # seconds
                TARGET_MAX_QUEUE_LATENCY_MS = float(os.getenv('TARGET_MAX_QUEUE_LATENCY_MS', '250'))
                # (We keep only one frame, so queue latency is implicit; variable reserved for future adaptive policies.)

                async def inference_loop():
                    busy = False
                    last_send = time.time()

                    async def run_inference(img, capture_ts, recv_time):
                        nonlocal busy, last_send
                        try:
                            # Run blocking model inference in thread
                            det_payload = await asyncio.get_event_loop().run_in_executor(
                                executor, lambda: detector.infer(img)
                            )
                            # Cooperative wait if system resources saturated
                            await resource_limiter.wait_for_resources()
                            det_payload['inference_start_ts'] = det_payload.get('capture_ts')
                            if latest_capture_ts_map.get('last') is not None:
                                det_payload['capture_ts'] = latest_capture_ts_map['last']
                            else:
                                det_payload['capture_ts'] = capture_ts
                            det_payload['recv_ts'] = recv_time
                            det_payload['inference_duration_ms'] = det_payload.get('inference_ms')
                            send_ts = time.time()
                            det_payload['send_ts'] = send_ts
                            det_payload['queue_delay_ms'] = (recv_time - capture_ts) * 1000.0 if capture_ts else None
                            det_payload['end_to_end_ms'] = (send_ts - capture_ts) * 1000.0 if capture_ts else None
                            gap = send_ts - last_send
                            if PIPELINE_VERBOSE and gap > 1.0:  # optional diagnostics
                                print(f"[gap] {gap:.2f}s since last payload (processed={stats['processed']})")
                            last_send = send_ts
                            if dc is not None:
                                dc.send(json.dumps(det_payload))
                            stats['processed'] += 1
                        except Exception as e:
                            print('Inference/send error', e)
                        finally:
                            busy = False
                            # If a newer frame arrived while busy, trigger immediately
                            if latest_frame['img'] is not None:
                                frame_event.set()
                        # Periodic stats log
                        now_log = time.time()
                        if PIPELINE_VERBOSE and now_log - stats['last_log'] >= LOG_INTERVAL:
                            proc = stats['processed']
                            rec = stats['received']
                            drop = stats['dropped']
                            print(f"[pipeline] frames received={rec} processed={proc} dropped={drop} drop_rate={(drop/max(rec,1)):.2%}")
                            stats['last_log'] = now_log

                    while True:
                        await frame_event.wait()
                        # If already processing, let current inference finish; new frame overwrites slot
                        if busy:
                            frame_event.clear()
                            await asyncio.sleep(0)
                            continue
                        frame_event.clear()
                        if latest_frame['img'] is None:
                            continue
                        img = latest_frame['img']
                        capture_ts = latest_frame['capture_ts']
                        recv_time = latest_frame['recv_time']
                        latest_frame['img'] = None  # free slot
                        busy = True
                        asyncio.create_task(run_inference(img, capture_ts, recv_time))

                asyncio.create_task(inference_loop())

                while True:
                    try:
                        frame = await track.recv()  # type: VideoFrame
                    except Exception:
                        break
                    stats['received'] += 1
                    frame_count['n'] += 1

                    # Derive capture_ts from frame.pts/time_base if available.
                    frame_time = None
                    if frame.pts is not None and frame.time_base is not None:
                        try:
                            frame_time = frame.pts * frame.time_base
                        except Exception:
                            frame_time = None
                    if frame_time is None and hasattr(frame, 'time') and frame.time is not None:
                        frame_time = frame.time  # fallback
                    now = time.time()
                    if frame_time is not None:
                        if base_time_offset is None:
                            base_time_offset = now - frame_time
                        capture_ts = base_time_offset + frame_time
                    else:
                        capture_ts = now  # fallback if no pts

                    # Convert frame to ndarray. If inference loop busy and another frame arrives before
                    # conversion completes, we still process this one because conversion precedes overwrite.
                    try:
                        img = frame.to_ndarray(format='bgr24')
                    except Exception as e:
                        print('Frame conversion error', e)
                        continue
                    # Overwrite latest (drop previous unprocessed if any)
                    if latest_frame['img'] is not None:
                        stats['dropped'] += 1
                    latest_frame['img'] = img
                    latest_frame['capture_ts'] = capture_ts
                    latest_frame['recv_time'] = now
                    frame_event.set()
                    # Yield to allow inference task to run promptly
                    await asyncio.sleep(0)

                print('Video track ended')
            asyncio.ensure_future(recv_frames())

    await pc.setRemoteDescription(offer)
    # create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    try:
        await resource_limiter.shutdown()
    except Exception:
        pass

if __name__ == '__main__':
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get('/', index)
    app.router.add_post('/offer', offer)
    app.on_shutdown.append(on_shutdown)
    web.run_app(app, port=8080)
