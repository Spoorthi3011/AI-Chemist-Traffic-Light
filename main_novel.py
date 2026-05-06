#!/usr/bin/env python3
import sys
import os
import datetime
import socketserver
import http.server
import io
import json

# Host - http://192.168.0.1:8765/  
# fuser -k /dev/video0
# fuser -k /dev/video1
# fuser -k /dev/video2
# fuser /dev/video0
# fuser /dev/video1
# fuser /dev/video2  
# MUST be set before ANY cv2 import to prevent Qt crash
os.environ['QT_QPA_PLATFORM'] = 'offscreen' 

os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

sys.path.extend(['/home/robot/Code/groupc', '/home/robot/Code/groupc/handlers', '/home/robot/Code/groupc/examples'])

# MAMBAFORGE HYBRID - Uses mambaforge torch + system libs
MAMBAFORGE_PATH = '/home/robot/mambaforge/lib/python3.10/site-packages'
if MAMBAFORGE_PATH not in sys.path:
    sys.path.insert(0, MAMBAFORGE_PATH)

try:
    from ultralytics import YOLO
    yolo_model = YOLO('/home/robot/Code/groupc/best.pt')
    YOLO_AVAILABLE = True
    print("MAMBAFORGE HYBRID YOLO LOADED!")
    print(f"   Model classes: {yolo_model.names}")
except:
    YOLO_AVAILABLE = False
    print("Color-only mode")

import cv2
import time
import threading
import numpy as np

# Robot & Hardware
from handlers.robot_handler import URControl
from examples.utils.robotiq.robotiq_gripper import RobotiqGripper
from handlers.camera_handler_2 import Camera
from handlers.colour_detection import ColourDetector
from handlers.ika_strirrer import IKAStirrer

# AI SYSTEMS
from handlers.ai_chemist import AIChemist
from handlers.digital_twin import DigitalTwin
from handlers.reaction_analyzer import ReactionAnalyzer
from handlers.vlm_monitor import VLMMonitor

HOST = "192.168.0.2"
PORT = 30003

# GLOBAL VARIABLES
stop_event = threading.Event()
digital_twin_global = None

# ─────────────────────────────────────────────
#  COLOUR SEQUENCE:
#  Phase 1: GREEN -> RED    (Oxidation)
#  Phase 2: RED   -> YELLOW (Reduction)
#  Phase 3: YELLOW -> GREEN (Regeneration)
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  WEB STREAM SERVER  (replaces cv2.imshow)
#  Open http://192.168.0.2:8765 in any browser
# ─────────────────────────────────────────────
# Shared state dict — detection thread writes, browser reads
stream_status = {
    "phase": "Phase 1 — Oxidation",
    "colour": "UNKNOWN",
    "pixels": {"RED": 0, "GREEN": 0, "YELLOW": 0},
    "rpm": 0,
    "yolo_conf": 0.0,
    "phase1": "waiting",   # "running" | "done" | "waiting"
    "phase2": "waiting",
    "phase3": "waiting",
    "log": [],
    "images_saved": 0,
    "frame_count": 0,
}

def status_log(msg):
    """Call this instead of print() for messages you want in the browser log."""
    print(msg)
    stream_status["log"].append(msg)
    if len(stream_status["log"]) > 20:
        stream_status["log"].pop(0)


class StreamServer:
    def __init__(self, port=8765):
        self._frame = None
        self._lock = threading.Lock()
        self._port = port
        self._server = None
        self._thread = None

    def update_frame(self, jpeg_bytes):
        with self._lock:
            self._frame = jpeg_bytes

    def get_frame(self):
        with self._lock:
            return self._frame

    def start(self):
        stream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(DASHBOARD_HTML.encode())

                elif self.path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'keep-alive')
                    self.end_headers()
                    try:
                        while True:
                            frame = stream.get_frame()
                            if frame:
                                self.wfile.write(b'--jpgboundary\r\n')
                                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                                self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode())
                                self.wfile.write(frame)
                                self.wfile.write(b'\r\n')
                                self.wfile.flush()
                            time.sleep(0.04)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

                elif self.path == '/status':
                    data = json.dumps(stream_status).encode()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(data)

        socketserver.TCPServer.allow_reuse_address = True
        self._server = socketserver.ThreadingTCPServer(('0.0.0.0', self._port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[Stream] Dashboard at http://{HOST}:{self._port}")

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<title>AI-Chemist Live</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; color: #ccc; font-family: monospace; }
  .topbar { background: #0a0a0a; border-bottom: 1px solid #1a1a1a; padding: 8px 16px;
            display: flex; justify-content: space-between; align-items: center; }
  .topbar-title { color: #00ff41; font-size: 13px; }
  .topbar-right { font-size: 11px; color: #555; }
  .rec { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         background: #E24B4A; margin-right: 4px; animation: blink 1s infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }
  .main { display: grid; grid-template-columns: 1fr 280px; gap: 12px; padding: 12px; }
  .feed-wrap { background: #000; border-radius: 8px; border: 1.5px solid #1e3a1e; overflow: hidden; }
  .feed-wrap img { width: 100%; display: block; }
  .feed-footer { padding: 6px 12px; display: flex; justify-content: space-between;
                 font-size: 11px; color: #444; border-top: 1px solid #1a1a1a; }
  .sidebar { display: flex; flex-direction: column; gap: 10px; }
  .card { background: #141414; border: 0.5px solid #222; border-radius: 10px; padding: 12px 14px; }
  .card-title { font-size: 10px; color: #444; text-transform: uppercase;
                letter-spacing: .08em; margin-bottom: 10px; }
  .phase-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
  .pdot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
  .pname { font-size: 12px; flex: 1; }
  .pstatus { font-size: 11px; font-weight: 500; }
  .pbar-bg { height: 4px; background: #1e1e1e; border-radius: 2px; margin: 0 0 10px 17px; }
  .pbar-fill { height: 4px; border-radius: 2px; transition: width .5s; }
  .colour-swatch { width: 34px; height: 34px; border-radius: 6px; border: 1px solid #2a2a2a;
                   flex-shrink: 0; transition: background .5s; }
  .colour-row { display: flex; align-items: center; gap: 10px; padding: 6px 0 10px; }
  .colour-name { font-size: 15px; font-weight: 500; color: #ddd; }
  .colour-sub { font-size: 11px; color: #555; }
  .pbar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
  .pbar-lbl { font-size: 11px; width: 12px; }
  .pbar-bg2 { flex: 1; height: 5px; background: #1e1e1e; border-radius: 3px; overflow: hidden; }
  .pbar-fill2 { height: 5px; border-radius: 3px; transition: width .4s; }
  .pbar-num { font-size: 10px; color: #555; width: 36px; text-align: right; }
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat-cell { background: #1a1a1a; border-radius: 7px; padding: 8px 10px; }
  .stat-label { font-size: 10px; color: #555; margin-bottom: 2px; }
  .stat-val { font-size: 18px; font-weight: 500; }
  .log-box { background: #0a0a0a; border-radius: 6px; padding: 8px; font-size: 10px;
             max-height: 130px; overflow-y: auto; }
  .log-line { margin-bottom: 3px; color: #555; }
  .log-line.ok  { color: #3B6D11; }
  .log-line.warn{ color: #854F0B; }
  .log-line.info{ color: #185FA5; }
  .outpath { font-size: 10px; color: #333; margin-top: 6px; word-break: break-all; }
</style>
</head>
<body>
<div class="topbar">
  <span class="topbar-title">&#9632; AI-CHEMIST LIVE &nbsp;|&nbsp; GREEN &#8594; RED &#8594; YELLOW &#8594; GREEN</span>
  <span class="topbar-right"><span class="rec"></span>REC &nbsp;|&nbsp; 192.168.0.2:8765</span>
</div>
<div class="main">
  <div class="feed-wrap">
    <img src="/stream" id="feed">
    <div class="feed-footer">
      <span>MJPEG &bull; YOLO + colour detection</span>
      <span id="frame-count">Frame #0</span>
    </div>
  </div>
  <div class="sidebar">

    <div class="card">
      <div class="card-title">Phase progress</div>
      <div class="phase-row">
        <div class="pdot" id="dot1" style="background:#639922"></div>
        <div class="pname" id="pname1">Phase 1 — oxidation</div>
        <div class="pstatus" id="pstat1" style="color:#854F0B">waiting</div>
      </div>
      <div class="pbar-bg"><div class="pbar-fill" id="pbar1" style="width:0%;background:#639922"></div></div>
      <div class="phase-row">
        <div class="pdot" id="dot2" style="background:#E24B4A;opacity:.3"></div>
        <div class="pname" id="pname2" style="color:#444">Phase 2 — reduction</div>
        <div class="pstatus" id="pstat2" style="color:#444">waiting</div>
      </div>
      <div class="pbar-bg"><div class="pbar-fill" id="pbar2" style="width:0%;background:#E24B4A"></div></div>
      <div class="phase-row">
        <div class="pdot" id="dot3" style="background:#639922;opacity:.3"></div>
        <div class="pname" id="pname3" style="color:#444">Phase 3 — regeneration</div>
        <div class="pstatus" id="pstat3" style="color:#444">waiting</div>
      </div>
      <div class="pbar-bg" style="margin-bottom:0">
        <div class="pbar-fill" id="pbar3" style="width:0%;background:#639922"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Colour detection</div>
      <div class="colour-row">
        <div class="colour-swatch" id="swatch"></div>
        <div><div class="colour-name" id="colour-name">SCANNING</div>
             <div class="colour-sub" id="colour-sub">Waiting for detection...</div></div>
      </div>
      <div class="pbar-row">
        <div class="pbar-lbl" style="color:#E24B4A">R</div>
        <div class="pbar-bg2"><div class="pbar-fill2" id="rbar" style="width:0%;background:#E24B4A"></div></div>
        <div class="pbar-num" id="rval">0</div>
      </div>
      <div class="pbar-row">
        <div class="pbar-lbl" style="color:#639922">G</div>
        <div class="pbar-bg2"><div class="pbar-fill2" id="gbar" style="width:0%;background:#639922"></div></div>
        <div class="pbar-num" id="gval">0</div>
      </div>
      <div class="pbar-row">
        <div class="pbar-lbl" style="color:#EF9F27">Y</div>
        <div class="pbar-bg2"><div class="pbar-fill2" id="ybar" style="width:0%;background:#EF9F27"></div></div>
        <div class="pbar-num" id="yval">0</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">System stats</div>
      <div class="stat-grid">
        <div class="stat-cell">
          <div class="stat-label">Stirrer RPM</div>
          <div class="stat-val" id="rpm" style="color:#BA7517">0</div>
        </div>
        <div class="stat-cell">
          <div class="stat-label">YOLO conf</div>
          <div class="stat-val" id="conf" style="color:#3B6D11">—</div>
        </div>
        <div class="stat-cell">
          <div class="stat-label">Images saved</div>
          <div class="stat-val" id="imgs" style="color:#ccc">0</div>
        </div>
        <div class="stat-cell">
          <div class="stat-label">Current phase</div>
          <div class="stat-val" style="font-size:13px;color:#ccc" id="phase-label">P1</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Log</div>
      <div class="log-box" id="logbox"></div>
      <div class="outpath" id="outpath">OUTPUT/...</div>
    </div>

  </div>
</div>
<script>
const SWATCHES = {
  GREEN:'#2d6a1f', RED:'#8b1a1a', YELLOW:'#a07800', UNKNOWN:'#2a2a2a', WAITING_FOR_VIAL:'#1a1a1a'
};
const PHASE_COLORS = { running:'#639922', done:'#185FA5', waiting:'#444' };

function classForLog(line) {
  if (line.includes('confirmed') || line.includes('COMPLETE') || line.includes('✓')) return 'ok';
  if (line.includes('watching') || line.includes('Timeout') || line.includes('timed')) return 'warn';
  if (line.includes('[Stream]') || line.includes('[Robot]') || line.includes('[Camera]')) return 'info';
  return '';
}

function phaseWidth(status) {
  return status === 'done' ? '100%' : status === 'running' ? '55%' : '0%';
}

function update(d) {
  document.getElementById('frame-count').textContent = 'Frame #' + d.frame_count;
  document.getElementById('colour-name').textContent = d.colour;
  document.getElementById('colour-sub').textContent = 'Current state';
  document.getElementById('swatch').style.background = SWATCHES[d.colour] || '#2a2a2a';

  const tot = (d.pixels.RED + d.pixels.GREEN + d.pixels.YELLOW) || 1;
  const rp = Math.round(d.pixels.RED / tot * 100);
  const gp = Math.round(d.pixels.GREEN / tot * 100);
  const yp = Math.round(d.pixels.YELLOW / tot * 100);
  document.getElementById('rbar').style.width = rp + '%';
  document.getElementById('gbar').style.width = gp + '%';
  document.getElementById('ybar').style.width = yp + '%';
  document.getElementById('rval').textContent = d.pixels.RED;
  document.getElementById('gval').textContent = d.pixels.GREEN;
  document.getElementById('yval').textContent = d.pixels.YELLOW;

  document.getElementById('rpm').textContent = d.rpm;
  document.getElementById('conf').textContent = d.yolo_conf > 0 ? d.yolo_conf.toFixed(2) : '—';
  document.getElementById('imgs').textContent = d.images_saved;
  document.getElementById('phase-label').textContent = d.phase || '—';

  [1,2,3].forEach(i => {
    const s = d['phase'+i];
    const col = PHASE_COLORS[s] || '#444';
    document.getElementById('dot'+i).style.opacity = s === 'waiting' ? '0.25' : '1';
    document.getElementById('pstat'+i).textContent = s;
    document.getElementById('pstat'+i).style.color = col;
    document.getElementById('pname'+i).style.color = s === 'waiting' ? '#444' : '#ccc';
    document.getElementById('pbar'+i).style.width = phaseWidth(s);
  });

  const box = document.getElementById('logbox');
  box.innerHTML = d.log.slice(-12).map(l =>
    `<div class="log-line ${classForLog(l)}">${l}</div>`
  ).join('');
  box.scrollTop = box.scrollHeight;
}

async function poll() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    update(d);
  } catch(e) {}
  setTimeout(poll, 1000);
}
poll();
</script>
</body>
</html>"""

def create_output_folder():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"OUTPUT/{timestamp}_AI_CHEMIST"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/videos", exist_ok=True)
    os.makedirs(f"{output_dir}/reports", exist_ok=True)
    print(f"[OUTPUT] Created: {output_dir}")
    return output_dir


def plot_loop():
    # DigitalTwin rendering disabled - Matplotlib cannot run outside main thread
    # Data is still collected and saved to report at end of experiment
    print("[DigitalTwin] Data collection active (live plot disabled - no main thread)")
    while not stop_event.is_set():
        time.sleep(5)
    print("[DigitalTwin] Stopped.")


YOUR_POSITIONS = {
    'start':         [1.680945634841919,  -1.9105240307249964, 2.3388877550708216,
                      -1.9763552151122035, -1.6369965712176722, 0.12794356048107147],
    'pick':          [1.6805652379989624, -1.7611729107298792, 2.4781621138202112,
                      -2.2649728260436,   -1.638026539479391,  0.12882831692695618],
    'up':            [1.680945634841919,  -1.9105240307249964, 2.3388877550708216,
                      -1.9763552151122035, -1.6369965712176722, 0.12794356048107147],
    'above_stirrer': [1.5243818759918213, -1.3093386751464386, 1.5124495665179651,
                      -1.816165109673971,  -1.6351736227618616, 0.12748508155345917],
    'hover':         [1.5242940187454224, -1.248557524090149,  1.6598289648639124,
                      -2.0242706737914027, -1.6358855406390589, 0.12835928797721863],
    'insert':        [1.537716269493103,  -1.51904912412677,   1.6096895376788538,
                      -1.7614737949767054, -1.6358497778521937, 0.12833695113658905]
}


class VideoRecorder:
    def __init__(self, output_dir, cam):
        self.output_dir = output_dir
        self.cam = cam
        self.writer = None
        self.fps = 20
        self.frame_count = 0
        self.recording = False

    def start_recording(self, name):
        if self.writer:
            self.writer.release()

        # Wait for camera to produce a valid frame (up to 5 seconds)
        timeout = 5.0
        start = time.time()
        while self.cam.current_frame is None:
            if time.time() - start > timeout:
                print(f"[VIDEO] WARNING: Camera not ready after {timeout}s - skipping {name}")
                return None
            time.sleep(0.1)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        filename = f"{self.output_dir}/videos/{name}_{timestamp}.avi"
        h, w = self.cam.current_frame.shape[:2]
        self.writer = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))
        self.frame_count = 0
        self.recording = True
        print(f"[VIDEO] Recording started: {filename}")
        return filename

    def write_frame(self, annotated_frame):
        if self.writer and self.recording and annotated_frame is not None:
            self.writer.write(annotated_frame)
            self.frame_count += 1

    def stop_recording(self):
        if self.writer:
            self.writer.release()
            self.writer = None
            self.recording = False
            print(f"[VIDEO] Saved {self.frame_count} frames")


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1: GREEN -> RED  (Oxidation)
# ─────────────────────────────────────────────────────────────────────────────
def phase1_oxidation(robot, gripper, cam, stirrer, ai_chemist, detector, analyzer, output_dir, video_recorder):
    stream_status["phase1"] = "running"
    stream_status["phase"] = "Phase 1"
    stream_status["rpm"] = 1500 
    print("\n" + "="*60)
    print("PHASE 1: Oxidation  GREEN -> RED")
    print("="*60)


    print("[Robot] Moving to start")
    robot.move_joint_list(YOUR_POSITIONS['start'], 0.5, 0.5, 0.02)
    gripper.move(gripper.get_open_position(), 125, 125)
    cam.capture_image(f'{output_dir}/images/phase1_start.jpg')
    stream_status["images_saved"] += 1

    print("[Robot] Moving to pick")
    robot.move_joint_list(YOUR_POSITIONS['pick'], 0.5, 0.5, 0.02)
    gripper.move(255, 255, 255)
    cam.capture_image(f'{output_dir}/images/phase1_pick.jpg')
    stream_status["images_saved"] += 1

    print("[Robot] Moving to up")
    robot.move_joint_list(YOUR_POSITIONS['up'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase1_up.jpg')
    stream_status["images_saved"] += 1
    
    print("[Robot] Moving to above_stirrer")
    robot.move_joint_list(YOUR_POSITIONS['above_stirrer'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase1_above_stirrer.jpg')
    stream_status["images_saved"] += 1

    print("[Robot] Moving to hover")
    robot.move_joint_list(YOUR_POSITIONS['hover'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase1_hover.jpg')
    stream_status["images_saved"] += 1

    rpm = ai_chemist.recommend_rpm('OXIDATION')
    stirrer.set_speed(1500)
    stirrer.start_stirring()
    print(f"[Stirrer] Started at {rpm} RPM - waiting for GREEN -> RED")
    time.sleep(15)
    print("[Robot] Moving to insert")
    robot.move_joint_list(YOUR_POSITIONS['insert'], 0.3, 0.3, 0.01)
    cam.capture_image(f'{output_dir}/images/phase1_inserted.jpg')
    stream_status["images_saved"] += 1

    # First confirm we are seeing GREEN stably before waiting for RED
    print("[Wait] Confirming GREEN is stable before watching for RED...")
    green_confirmed = wait_for_colour(detector, 'GREEN', timeout=10, output_dir=output_dir)
    if green_confirmed:
        print("[Wait] GREEN confirmed - now watching for RED transition...")
    else:
        print("[Wait] GREEN not confirmed in 10s - proceeding anyway...")

    # Stirrer runs until RED is stably detected
    success = wait_for_colour(detector, 'RED', timeout=180, output_dir=output_dir)
    stirrer.stop_stirring()
    print(f"[Stirrer] Stopped - RED {'achieved' if success else 'timed out'}")

    metrics = analyzer.analyze_phase(detector)
    stream_status["phase1"] = "done"
    return success, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2: RED -> YELLOW  (Reduction)
# ─────────────────────────────────────────────────────────────────────────────
def phase2_reduction(robot, cam, stirrer, ai_chemist, detector, analyzer, phase1_metrics, output_dir, video_recorder):
    stream_status["phase2"] = "running"
    stream_status["phase"] = "Phase 2"
    print("\n" + "="*60)
    print("PHASE 2: Reduction  RED -> YELLOW")
    print("="*60)


    print("[Robot] Extract - hover")
    robot.move_joint_list(YOUR_POSITIONS['hover'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase2_extract_hover.jpg')
    stream_status["images_saved"] += 1

    print("[Robot] Extract - above stirrer")
    robot.move_joint_list(YOUR_POSITIONS['above_stirrer'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase2_above_stirrer.jpg')
    stream_status["images_saved"] += 1

    print("[Stirrer] Skipped in Phase 2 (manual/chemical transition only)")

    print("[Robot] Re-hover")
    robot.move_joint_list(YOUR_POSITIONS['hover'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase2_rehover.jpg')
    stream_status["images_saved"] += 1

    print("[Robot] Final insert")
    robot.move_joint_list(YOUR_POSITIONS['insert'], 0.3, 0.3, 0.01)
    cam.capture_image(f'{output_dir}/images/phase2_final_insert.jpg')
    stream_status["images_saved"] += 1

    # Stirrer runs until YELLOW detected
    success = wait_for_colour(detector, 'YELLOW', timeout=120, output_dir=output_dir)
    print(f"[Stirrer] Stopped - YELLOW {'achieved' if success else 'timed out'}")

    metrics = analyzer.analyze_phase(detector)
    stream_status["phase2"] = "done"
    return success, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3: YELLOW -> GREEN  (Regeneration)
# ─────────────────────────────────────────────────────────────────────────────
def phase3_regeneration(robot, cam, stirrer, ai_chemist, detector, analyzer, phase2_metrics, output_dir, video_recorder):
    stream_status["phase3"] = "running"
    stream_status["phase"] = "Phase 3"
    print("\n" + "="*60)
    print("PHASE 3: Regeneration  YELLOW -> GREEN")
    print("="*60)


    print("[Robot] Extract - hover")
    robot.move_joint_list(YOUR_POSITIONS['hover'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase3_extract_hover.jpg')
    stream_status["images_saved"] += 1

    # print("[Robot] Extract - above stirrer")
    # robot.move_joint_list(YOUR_POSITIONS['above_stirrer'], 0.5, 0.5, 0.02)
    # cam.capture_image(f'{output_dir}/images/phase3_above_stirrer.jpg')

    stirrer.set_speed(1500)
    stirrer.start_stirring()
    print("[Stirrer] Started at 1500 RPM for 25 seconds")

    time.sleep(25)

    stirrer.stop_stirring()
    print("[Stirrer] Stopped after 10 seconds")

    print("[Robot] Re-hover")
    robot.move_joint_list(YOUR_POSITIONS['hover'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase3_rehover.jpg')
    stream_status["images_saved"] += 1

    print("[Robot] Final insert")
    robot.move_joint_list(YOUR_POSITIONS['insert'], 0.3, 0.3, 0.01)
    cam.capture_image(f'{output_dir}/images/phase3_final_insert.jpg')
    stream_status["images_saved"] += 1

    # Stirrer runs until GREEN detected (full cycle complete)
    success = wait_for_colour(detector, 'GREEN', timeout=200, output_dir=output_dir)
    print(f"[Stirrer] Stopped - GREEN {'achieved - FULL CYCLE COMPLETE!' if success else 'timed out'}")

    metrics = analyzer.analyze_phase(detector)
    stream_status["phase3"] = "done"
    return success, metrics



def wait_for_colour(detector, target_colour='RED', timeout=180, output_dir=None):
    print(f"[Wait] Waiting for: {target_colour} (timeout={timeout}s)")

    start_time = time.time()
    consecutive_count = 0
    missed_frames = 0

    REQUIRED_CONSECUTIVE = 2
    MAX_MISSED_FRAMES = 5   # allow small flicker noise
    REQUIRED_HOLD_SECS = 1.5

    colour_hold_start = None

    while time.time() - start_time < timeout:

        if detector.current_state == target_colour:
            consecutive_count += 1
            missed_frames = 0

            if consecutive_count == 1:
                colour_hold_start = time.time()

            hold_time = time.time() - colour_hold_start

            if consecutive_count >= REQUIRED_CONSECUTIVE and hold_time >= REQUIRED_HOLD_SECS:
                elapsed = time.time() - start_time
                print(f"[Wait] {target_colour} CONFIRMED after {elapsed:.1f}s")

                if output_dir and hasattr(detector, 'last_frame'):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(
                        f"{output_dir}/images/CONFIRMED_{target_colour}_{timestamp}.jpg",
                        detector.last_frame
                    )

                return True

        else:
            if consecutive_count > 0:
                missed_frames += 1

                if missed_frames <= MAX_MISSED_FRAMES:
                    print(f"[Wait] Small flicker ignored ({missed_frames}/{MAX_MISSED_FRAMES})")
                else:
                    print(f"[Wait] Resetting confirmation for {target_colour}")
                    consecutive_count = 0
                    missed_frames = 0
                    colour_hold_start = None

        time.sleep(0.5)

    print(f"[Wait] Timeout for {target_colour}")
    return False


# FIX 3 + FIX 4 + FIX 5: Replaced enhanced_detection()
def enhanced_detection(stop_event, cam, detector, vlm_monitor,
                       digital_twin_local, video_recorder,
                       output_dir, stream_server):

    # FIX 4: Reuse global YOLO model
    model = yolo_model if YOLO_AVAILABLE else None

    if model:
        print("[Detection] YOLO + Color pipeline")
    else:
        print("[Detection] Color-only pipeline")

    print("[Detection] Vision ready")
    print("[Detection] Live preview available in browser")

    video_recorder.start_recording("FULL_EXPERIMENT")

    save_count = 0

    while not stop_event.is_set():
        with cam._lock:
            if cam.current_frame is None:
                time.sleep(0.05)
                continue

            frame = cam.current_frame.copy()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        detections = []
        annotated = frame.copy()

        # YOLO Detection
        if model:
            try:
                results = model(frame, verbose=False, conf=0.15)
                detections = results[0].boxes if results[0].boxes else []
                annotated = results[0].plot()
            except Exception as e:
                print("[YOLO ERROR]", e)

        # ROI Selection
        if len(detections) > 0:
            box = max(detections, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # FIX 5: Safe clamp for ROI boundaries
            h, w = frame.shape[:2]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                roi = frame
                x1, y1, x2, y2 = 0, 0, w, h
            else:
                roi = frame[y1:y2, x1:x2]

            conf = float(box.conf[0])

        else:
            roi = None
            x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
            conf = 0.0

        # ONLY perform colour detection if vial is detected
        if roi is not None:
            state, pixels = detector.detect_colour(roi)
        else:
            state = "WAITING_FOR_VIAL"
            pixels = {
                'RED': 0,
                'GREEN': 0,
                'YELLOW': 0
    }

        previous_state = getattr(detector, "current_state", "UNKNOWN")

        detector.update_state(state)
        stream_status["colour"] = state
        stream_status["pixels"] = pixels
        stream_status["yolo_conf"] = round(conf, 2)
        stream_status["frame_count"] += 1
        detector.pixel_counts = pixels

        # FIX 3: Safe state_start_time reset on state change
        if not hasattr(detector, "state_start_time"):
            detector.state_start_time = time.time()

        if previous_state != state:
            detector.state_start_time = time.time()

        # Draw overlay
        colours = {
            'GREEN': (0, 255, 0),
            'RED': (0, 0, 255),
            'YELLOW': (0, 255, 255),
            'UNKNOWN': (128, 128, 128),
        }

        color = colours.get(state, (255, 255, 255))
        label_y = max(y1 - 35, 30)

        cv2.putText(
            annotated,
            f"COLOUR: {state}",
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )

        # FIX 1: Correct pixel keys
        pixel_text = (
            f"R:{pixels.get('RED', 0):.0f}  "
            f"G:{pixels.get('GREEN', 0):.0f}  "
            f"Y:{pixels.get('YELLOW', 0):.0f}"
        )

        cv2.putText(
            annotated,
            pixel_text,
            (x1, min(y2 + 25, frame.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        frame = annotated

        frame = vlm_monitor.get_status_overlay(frame)
        digital_twin_local.update(detector)

        vial_status = (
            f"VIAL DETECTED: {conf:.2f}"
            if len(detections) > 0
            else "WAITING FOR VIAL"
        )

        status = (
            f"[AI-CHEMIST] {vial_status} | "
            f"COLOR: {detector.current_state or 'SCANNING'}"
        )

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 60),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            status,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Sequence: GREEN -> RED -> YELLOW -> GREEN",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2
        )

        video_recorder.write_frame(frame)

        try:
            success, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

            if not success:
                print("[STREAM ERROR] JPEG encode failed")
            else:
                stream_server.update_frame(jpeg.tobytes())
        except Exception:
            pass

        save_count += 1

        if save_count % 300 == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(
                f'{output_dir}/images/LIVE_{timestamp}_{state}.jpg',
                frame
            )

    video_recorder.stop_recording()


def main_novel():
    global digital_twin_global, stop_event

    print("AI-CHEMIST SYSTEM")
    print("COLOUR SEQUENCE: GREEN -> RED -> YELLOW -> GREEN")
    print("=" * 80)

    output_dir = create_output_folder()

    # Start web stream server (replaces cv2.imshow, no GTK needed)
    stream_server = StreamServer(port=8765)
    stream_server.start()

    robot = URControl(ip=HOST, port=PORT)
    gripper = RobotiqGripper()
    gripper.connect(HOST, 63352)
    print(f"[Gripper] Current: {gripper.get_current_position()}")
    gripper.move(255, 125, 125)

    cam = Camera(2, f'{output_dir}/images')
    cam.start_recording()
    print("[Camera] Recording started")
    time.sleep(2)

    detector = ColourDetector()

    # FIX 1 + FIX 2: Initialise detector state attributes
    detector.current_state = "UNKNOWN"
    detector.pixel_counts = {
        'RED': 0,
        'GREEN': 0,
        'YELLOW': 0
    }
    detector.state_start_time = time.time()

    ai_chemist = AIChemist()
    digital_twin = DigitalTwin()
    digital_twin_global = digital_twin
    analyzer = ReactionAnalyzer()
    vlm_monitor = VLMMonitor(cam, '/home/robot/Code/groupc/best.pt')

    video_recorder = VideoRecorder(output_dir, cam)

    plot_thread = threading.Thread(target=plot_loop, daemon=True)
    plot_thread.start()

    det_thread = threading.Thread(target=enhanced_detection,
                                  args=(stop_event, cam, detector, vlm_monitor, digital_twin,
                                        video_recorder, output_dir, stream_server),
                                  daemon=True)
    det_thread.start()

    stirrer = IKAStirrer(port='/dev/ttyACM0')

    try:
        cam.capture_image(f'{output_dir}/images/STARTUP.jpg')
        stream_status["images_saved"] += 1

        # PHASE 1: GREEN -> RED
        phase1_success, phase1_metrics = phase1_oxidation(
            robot, gripper, cam, stirrer, ai_chemist, detector, analyzer, output_dir, video_recorder
        )
        if not phase1_success:
            print("Phase 1 incomplete - continuing to Phase 2")

        # PHASE 2: RED -> YELLOW
        phase2_success, phase2_metrics = phase2_reduction(
            robot, cam, stirrer, ai_chemist, detector, analyzer, phase1_metrics, output_dir, video_recorder
        )
        if not phase2_success:
            print("Phase 2 incomplete - continuing to Phase 3")

        # PHASE 3: YELLOW -> GREEN
        phase3_success, phase3_metrics = phase3_regeneration(
            robot, cam, stirrer, ai_chemist, detector, analyzer, phase2_metrics, output_dir, video_recorder
        )
        if phase3_success:
            print("\nFULL CYCLE COMPLETE: GREEN -> RED -> YELLOW -> GREEN")
        else:
            print("\nPhase 3 timed out - cycle incomplete")

        # Return to above stirrer
        print("\nReturning to above_stirrer position")
        robot.move_joint_list(YOUR_POSITIONS['above_stirrer'], 0.5, 0.5, 0.02)
        cam.capture_image(f'{output_dir}/images/final_return.jpg')
        stream_status["images_saved"] += 1

        print("\n" + "=" * 80)
        analyzer.final_report()
        digital_twin.save_report(f'{output_dir}/reports/ai_chemist_report.png')

        summary = f"""
AI-CHEMIST EXPERIMENT SUMMARY
=============================
Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Folder: {output_dir}

Colour Sequence: GREEN -> RED -> YELLOW -> GREEN

Phase 1 (GREEN->RED)    : {'YES' if phase1_success else 'NO (timed out)'}
Phase 1 Metrics: {phase1_metrics}

Phase 2 (RED->YELLOW)   : {'YES' if phase2_success else 'NO (timed out)'}
Phase 2 Metrics: {phase2_metrics}

Phase 3 (YELLOW->GREEN) : {'YES' if phase3_success else 'NO (timed out)'}
Phase 3 Metrics: {phase3_metrics}

Full Cycle Complete: {'YES' if (phase1_success and phase2_success and phase3_success) else 'NO'}

Files Generated:
  Images : {output_dir}/images/
  Videos : {output_dir}/videos/
  Reports: {output_dir}/reports/
=============================
"""
        with open(f'{output_dir}/reports/SUMMARY.txt', 'w') as f:
            f.write(summary)
        print(summary)

    except KeyboardInterrupt:
        print("\nUser interrupt")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        with open(f'{output_dir}/reports/ERROR_LOG.txt', 'w') as f:
            f.write(f"ERROR OCCURRED: {str(e)}\n{traceback.format_exc()}")

    finally:
        try:
            stirrer.stop_stirring()
            print("[Stirrer] Stopped")
        except Exception as e:
            print(f"Detection error: {e}")

        # FIRST stop threads
        stop_event.set()

        det_thread.join(timeout=5)
        plot_thread.join(timeout=5)

        print("[Threads] Detection + Plot stopped")

        # THEN stop camera + writers
        cam.stop_recording()
        print("[Camera] Original recording saved")

        video_recorder.stop_recording()
        print("[Video] All annotated videos saved!")

        stream_server.stop()
        print("[Stream] Web server stopped")

        print("\n" + "=" * 80)
        print("ALL FILES SAVED TO OUTPUT FOLDER!")
        print(f"{output_dir}")
        print("=" * 80)
if __name__ == "__main__":
    main_novel()
    
    
  
