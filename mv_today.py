#!/usr/bin/env python3
import sys
import os
import datetime
import socketserver
import http.server
import io

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
                    self.wfile.write(b"""
                        <html>
                        <head>
                          <title>AI-CHEMIST Live Feed</title>
                          <style>
                            body { background:#111; margin:0; display:flex;
                                   flex-direction:column; align-items:center; }
                            h2   { color:#0f0; font-family:monospace; margin:10px; }
                            img  { width:100%; max-width:900px; border:2px solid #0f0; }
                            p    { color:#888; font-family:monospace; font-size:12px; }
                          </style>
                        </head>
                        <body>
                          <h2>AI-CHEMIST LIVE FEED | GREEN -> RED -> YELLOW -> GREEN</h2>
                          <img id='f' src='/frame'>
                          <p>Auto-refreshes every 100ms | YOLO + Color Detection</p>
                          <script>
                            setInterval(function(){
                              document.getElementById('f').src = '/frame?t=' + Date.now();
                            }, 100);
                          </script>
                        </body>
                        </html>""")

                elif self.path.startswith('/frame'):
                    frame = stream.get_frame()
                    if frame:
                        self.send_response(200)
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Cache-Control', 'no-cache')
                        self.end_headers()
                        self.wfile.write(frame)
                    else:
                        self.send_response(204)
                        self.end_headers()

        # allow_reuse_address BEFORE socket is created
        socketserver.TCPServer.allow_reuse_address = True
        self._server = socketserver.TCPServer(('0.0.0.0', self._port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[Stream] Live preview at http://{HOST}:{self._port}")
        print(f"[Stream] Open that URL in any browser on your network!")

    def stop(self):
        if self._server:
            self._server.shutdown()


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
        filename = f"{self.output_dir}/videos/{name}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    print("\n" + "="*60)
    print("PHASE 1: Oxidation  GREEN -> RED")
    print("="*60)

    video_recorder.start_recording("PHASE1_GREEN_TO_RED")

    print("[Robot] Moving to start")
    robot.move_joint_list(YOUR_POSITIONS['start'], 0.5, 0.5, 0.02)
    gripper.move(gripper.get_open_position(), 125, 125)
    cam.capture_image(f'{output_dir}/images/phase1_start.jpg')

    print("[Robot] Moving to pick")
    robot.move_joint_list(YOUR_POSITIONS['pick'], 0.5, 0.5, 0.02)
    gripper.move(255, 255, 255)
    cam.capture_image(f'{output_dir}/images/phase1_pick.jpg')

    print("[Robot] Moving to up")
    robot.move_joint_list(YOUR_POSITIONS['up'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase1_up.jpg')

    print("[Robot] Moving to above_stirrer")
    robot.move_joint_list(YOUR_POSITIONS['above_stirrer'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase1_above_stirrer.jpg')

    print("[Robot] Moving to hover")
    robot.move_joint_list(YOUR_POSITIONS['hover'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase1_hover.jpg')

    rpm = ai_chemist.recommend_rpm('OXIDATION')
    stirrer.set_speed(rpm)
    stirrer.start_stirring()
    print(f"[Stirrer] Started at {rpm} RPM - waiting for GREEN -> RED")

    print("[Robot] Moving to insert")
    robot.move_joint_list(YOUR_POSITIONS['insert'], 0.3, 0.3, 0.01)
    cam.capture_image(f'{output_dir}/images/phase1_inserted.jpg')

    # First confirm we are seeing GREEN stably before waiting for RED
    print("[Wait] Confirming GREEN is stable before watching for RED...")
    green_confirmed = wait_for_colour(detector, 'GREEN', timeout=30, output_dir=output_dir)
    if green_confirmed:
        print("[Wait] GREEN confirmed - now watching for RED transition...")
    else:
        print("[Wait] GREEN not confirmed in 30s - proceeding anyway...")

    # Stirrer runs until RED is stably detected
    success = wait_for_colour(detector, 'RED', timeout=180, output_dir=output_dir)
    stirrer.stop_stirring()
    print(f"[Stirrer] Stopped - RED {'achieved' if success else 'timed out'}")

    metrics = analyzer.analyze_phase(detector)
    video_recorder.stop_recording()
    return success, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2: RED -> YELLOW  (Reduction)
# ─────────────────────────────────────────────────────────────────────────────
def phase2_reduction(robot, cam, stirrer, ai_chemist, detector, analyzer, phase1_metrics, output_dir, video_recorder):
    print("\n" + "="*60)
    print("PHASE 2: Reduction  RED -> YELLOW")
    print("="*60)

    video_recorder.start_recording("PHASE2_RED_TO_YELLOW")

    print("[Robot] Extract - hover")
    robot.move_joint_list(YOUR_POSITIONS['hover'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase2_extract_hover.jpg')

    print("[Robot] Extract - above stirrer")
    robot.move_joint_list(YOUR_POSITIONS['above_stirrer'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase2_above_stirrer.jpg')

    print("[Stirrer] Skipped in Phase 2 (manual/chemical transition only)")

    print("[Robot] Re-hover")
    robot.move_joint_list(YOUR_POSITIONS['hover'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase2_rehover.jpg')

    print("[Robot] Final insert")
    robot.move_joint_list(YOUR_POSITIONS['insert'], 0.3, 0.3, 0.01)
    cam.capture_image(f'{output_dir}/images/phase2_final_insert.jpg')

    # Stirrer runs until YELLOW detected
    success = wait_for_colour(detector, 'YELLOW', timeout=180, output_dir=output_dir)
    print(f"[Stirrer] Stopped - YELLOW {'achieved' if success else 'timed out'}")

    metrics = analyzer.analyze_phase(detector)
    video_recorder.stop_recording()
    return success, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3: YELLOW -> GREEN  (Regeneration)
# ─────────────────────────────────────────────────────────────────────────────
def phase3_regeneration(robot, cam, stirrer, ai_chemist, detector, analyzer, phase2_metrics, output_dir, video_recorder):
    print("\n" + "="*60)
    print("PHASE 3: Regeneration  YELLOW -> GREEN")
    print("="*60)

    video_recorder.start_recording("PHASE3_YELLOW_TO_GREEN")

    print("[Robot] Extract - hover")
    robot.move_joint_list(YOUR_POSITIONS['hover'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase3_extract_hover.jpg')

    print("[Robot] Extract - above stirrer")
    robot.move_joint_list(YOUR_POSITIONS['above_stirrer'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase3_above_stirrer.jpg')

    rpm3 = ai_chemist.adapt_rpm(800, phase2_metrics.get('yellow_speed', 0))
    stirrer.set_speed(rpm3)
    stirrer.start_stirring()
    print(f"[Stirrer] Started at {rpm3} RPM - waiting for YELLOW -> GREEN")

    print("[Robot] Re-hover")
    robot.move_joint_list(YOUR_POSITIONS['hover'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/phase3_rehover.jpg')

    print("[Robot] Final insert")
    robot.move_joint_list(YOUR_POSITIONS['insert'], 0.3, 0.3, 0.01)
    cam.capture_image(f'{output_dir}/images/phase3_final_insert.jpg')

    # Stirrer runs until GREEN detected (full cycle complete)
    success = wait_for_colour(detector, 'GREEN', timeout=300, output_dir=output_dir)
    stirrer.stop_stirring()
    print(f"[Stirrer] Stopped - GREEN {'achieved - FULL CYCLE COMPLETE!' if success else 'timed out'}")

    metrics = analyzer.analyze_phase(detector)
    video_recorder.stop_recording()
    return success, metrics


def wait_for_colour(detector, target_colour='RED', timeout=180, output_dir=None):
    print(f"[Wait] Waiting for: {target_colour} (timeout={timeout}s)")
    start_time = time.time()
    consecutive_count = 0
    colour_hold_start = None
    REQUIRED_CONSECUTIVE = 5    # must see colour 5 times in a row
    REQUIRED_HOLD_SECS   = 3.0  # colour must hold stable for 3 seconds

    while time.time() - start_time < timeout:
        if detector.current_state == target_colour:
            consecutive_count += 1
            if consecutive_count == 1:
                colour_hold_start = time.time()  # start timing how long it holds

            hold_duration = time.time() - colour_hold_start if colour_hold_start else 0

            if consecutive_count >= REQUIRED_CONSECUTIVE and hold_duration >= REQUIRED_HOLD_SECS:
                elapsed = time.time() - start_time
                print(f"[Wait] {target_colour} CONFIRMED stable for {hold_duration:.1f}s after {elapsed:.0f}s!")
                if output_dir and hasattr(detector, 'last_frame'):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'{output_dir}/images/CONFIRMED_{target_colour}_{timestamp}.jpg', detector.last_frame)
                time.sleep(2)
                return True
        else:
            if consecutive_count > 0:
                print(f"[Wait] {target_colour} flickered ({consecutive_count}x) - ignoring noise, still waiting...")
            consecutive_count = 0
            colour_hold_start = None

        # FIX 6: Reduced sleep interval from 1s to 0.5s
        time.sleep(0.5)

    print(f"[Wait] Timeout - proceeding")
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

        detections = []
        annotated = frame.copy()

        # YOLO Detection
        if model:
            try:
                results = model(frame, verbose=False, conf=0.25)
                detections = results[0].boxes if results[0].boxes else []
                annotated = results[0].plot()
            except Exception:
                pass

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
            roi = frame
            x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
            conf = 0.0

        # Colour Detection
        state, pixels = detector.detect_colour(roi)

        previous_state = getattr(detector, "current_state", "UNKNOWN")

        detector.update_state(state)
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
            'UNKNOWN': (128, 128, 128)
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
            else "FULL FRAME SCAN"
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
            _, jpeg = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
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

    cam = Camera(0, f'{output_dir}/images')
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
        except:
            pass

        cam.stop_recording()
        print("[Camera] Original recording saved")
        video_recorder.stop_recording()
        print("[Video] All annotated videos saved!")
        stream_server.stop()
        print("[Stream] Web server stopped")

        stop_event.set()
        det_thread.join(timeout=3)
        plot_thread.join(timeout=3)

        print("\n" + "=" * 80)
        print("ALL FILES SAVED TO OUTPUT FOLDER!")
        print(f"{output_dir}")
        print("Videos: FULL_EXPERIMENT | PHASE1_GREEN_TO_RED | PHASE2_RED_TO_YELLOW | PHASE3_YELLOW_TO_GREEN")
        print("=" * 80)


if __name__ == "__main__":
    main_novel()
