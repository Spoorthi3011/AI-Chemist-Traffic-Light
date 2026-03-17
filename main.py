#!/usr/bin/env python3
import sys
import os
import datetime
sys.path.extend(['/home/robot/Code/groupc', '/home/robot/Code/groupc/handlers', '/home/robot/Code/groupc/examples'])

MAMBAFORGE_PATH = '/home/robot/mambaforge/lib/python3.10/site-packages'
if MAMBAFORGE_PATH not in sys.path:
    sys.path.insert(0, MAMBAFORGE_PATH)

try:
    from ultralytics import YOLO
    yolo_model = YOLO('/home/robot/Code/groupc/best.pt')
    YOLO_AVAILABLE = True
    print(" YOLO LOADED!")
    print(f"   Model classes: {yolo_model.names}")
except:
    YOLO_AVAILABLE = False
    print(" Color-only mode")

import cv2
import time
import threading
import numpy as np

from handlers.robot_handler import URControl
from examples.utils.robotiq.robotiq_gripper import RobotiqGripper
from handlers.camera_handler_2 import Camera
from handlers.colour_detection import ColourDetector
from handlers.ika_strirrer import IKAStirrer
from handlers.ai_chemist import AIChemist
from handlers.digital_twin import DigitalTwin
from handlers.reaction_analyzer import ReactionAnalyzer
from handlers.vlm_monitor import VLMMonitor

HOST = "192.168.0.2"
PORT = 30003
EXTERNAL_CAM_INDEX = 0
WRIST_CAM_INDEX = 1

stop_event = threading.Event()
digital_twin_global = None
USE_WRIST_CAM_FOR_COLOUR = threading.Event()

# Shared annotation state visible to both camera streams
annotation_state = {
    'phase': 'STARTUP',
    'colour': 'UNKNOWN',
    'vial_detected': False,
    'robot_action': 'IDLE',
    'conf': 0.0,
    'rpm': 0,
    'pixels': {'red': 0, 'green': 0, 'blue': 0}
}
annotation_lock = threading.Lock()


def set_annotation(key, value):
    with annotation_lock:
        annotation_state[key] = value


def get_annotation():
    with annotation_lock:
        return annotation_state.copy()


def create_output_folder():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"OUTPUT/{timestamp}_AI_CHEMIST"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/images/wrist", exist_ok=True)
    os.makedirs(f"{output_dir}/videos", exist_ok=True)
    os.makedirs(f"{output_dir}/reports", exist_ok=True)
    print(f"[OUTPUT] Created: {output_dir}")
    return output_dir


def plot_loop():
    global digital_twin_global
    while not stop_event.is_set() and digital_twin_global is not None:
        try:
            digital_twin_global.render()
        except Exception as e:
            print(f"[DigitalTwin] Render error: {e}")
        time.sleep(2)


YOUR_POSITIONS = {
    'start': [1.680945634841919, -1.9105240307249964, 2.3388877550708216,
              -1.9763552151122035, -1.6369965712176722, 0.12794356048107147],
    'pick': [1.6805652379989624, -1.7611729107298792, 2.4781621138202112,
             -2.2649728260436, -1.638026539479391, 0.12882831692695618],
    'up': [1.680945634841919, -1.9105240307249964, 2.3388877550708216,
           -1.9763552151122035, -1.6369965712176722, 0.12794356048107147],
    'above_stirrer': [1.5243818759918213, -1.3093386751464386, 1.5124495665179651,
                      -1.816165109673971, -1.6351736227618616, 0.12748508155345917],
    'hover': [1.5242940187454224, -1.248557524090149, 1.6598289648639124,
              -2.0242706737914027, -1.6358855406390589, 0.12835928797721863],
    'insert': [1.537716269493103, -1.51904912412677, 1.6096895376788538,
               -1.7614737949767054, -1.6358497778521937, 0.12833695113658905]
}

COLOURS_BGR = {
    'GREEN': (0, 255, 0),
    'RED': (0, 0, 255),
    'YELLOW': (0, 255, 255),
    'UNKNOWN': (128, 128, 128)
}


def draw_overlay(frame, cam_label="EXT"):
    ann = get_annotation()
    h, w = frame.shape[:2]
    colour = ann['colour']
    bgr = COLOURS_BGR.get(colour, (255, 255, 255))

    # Top status bar
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"[{cam_label}] PHASE: {ann['phase']}  ACTION: {ann['robot_action']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"COLOUR: {colour}  CONF: {ann['conf']:.2f}  RPM: {ann['rpm']}  VIAL: {'YES' if ann['vial_detected'] else 'NO'}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

    # Bottom pixel bar
    px = ann['pixels']
    r = px.get('red', 0)
    g = px.get('green', 0)
    b = px.get('blue', 0)
    cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Pixels  R:{r:.0f}  G:{g:.0f}  B:{b:.0f}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Colour indicator box top-right
    cv2.rectangle(frame, (w - 160, 5), (w - 5, 65), bgr, -1)
    cv2.putText(frame, colour, (w - 150, 47), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Wrist cam label
    if USE_WRIST_CAM_FOR_COLOUR.is_set() and cam_label == "WRIST":
        cv2.putText(frame, "PRIMARY COLOUR SOURCE", (10, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


class DualCameraManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.wrist_cap = None
        self.wrist_frame = None
        self.wrist_lock = threading.Lock()
        self.wrist_thread = None
        self.wrist_running = False

    def start_wrist_camera(self):
        self.wrist_cap = cv2.VideoCapture(WRIST_CAM_INDEX)
        if not self.wrist_cap.isOpened():
            print(f"[WristCam] Could not open camera index {WRIST_CAM_INDEX}")
            return False
        self.wrist_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.wrist_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.wrist_running = True
        self.wrist_thread = threading.Thread(target=self._wrist_capture_loop, daemon=True)
        self.wrist_thread.start()
        time.sleep(1)
        print("[WristCam] ✅ Started!")
        return True

    def _wrist_capture_loop(self):
        while self.wrist_running and not stop_event.is_set():
            ret, frame = self.wrist_cap.read()
            if ret:
                with self.wrist_lock:
                    self.wrist_frame = frame.copy()
            time.sleep(0.033)

    def get_wrist_frame(self):
        with self.wrist_lock:
            if self.wrist_frame is not None:
                return self.wrist_frame.copy()
        return None

    def save_wrist_image(self, name):
        frame = self.get_wrist_frame()
        if frame is not None:
            path = f"{self.output_dir}/images/wrist/{name}.jpg"
            cv2.imwrite(path, frame)
            print(f"[WristCam] Saved: {path}")
            return path
        return None

    def stop_wrist_camera(self):
        self.wrist_running = False
        if self.wrist_thread:
            self.wrist_thread.join(timeout=2)
        if self.wrist_cap:
            self.wrist_cap.release()
        print("[WristCam] Stopped")


class VideoRecorder:
    def __init__(self, output_dir, cam):
        self.output_dir = output_dir
        self.cam = cam
        self.writer = None
        self.wrist_writer = None
        self.fps = 20
        self.frame_count = 0
        self.recording = False

    def start_recording(self, name):
        if self.writer:
            self.writer.release()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/videos/{name}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = self.cam.current_frame.shape[:2]
        self.writer = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))
        self.frame_count = 0
        self.recording = True
        print(f"[VIDEO] Recording: {filename}")
        return filename

    def start_wrist_recording(self, name, frame_shape):
        if self.wrist_writer:
            self.wrist_writer.release()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/videos/WRIST_{name}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = frame_shape[:2]
        self.wrist_writer = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))
        print(f"[WristVideo] Recording: {filename}")
        return filename

    def write_frame(self, frame):
        if self.writer and self.recording:
            self.writer.write(frame)
            self.frame_count += 1

    def write_wrist_frame(self, frame):
        if self.wrist_writer:
            self.wrist_writer.write(frame)

    def stop_recording(self):
        if self.writer:
            self.writer.release()
            self.writer = None
            self.recording = False
            print(f"[VIDEO] Saved {self.frame_count} frames")

    def stop_wrist_recording(self):
        if self.wrist_writer:
            self.wrist_writer.release()
            self.wrist_writer = None
            print("[WristVideo] Stopped")


def check_vial_at_start(cam, detector, dual_cam, model, output_dir, timeout=30):
    """
    Look at the start/pick position and confirm the vial is present using YOLO.
    Blocks until vial detected or timeout. Annotates both camera feeds live.
    Returns True if vial found, False if timeout.
    """
    print("[VialCheck] Checking for vial at start position...")
    set_annotation('phase', 'VIAL CHECK')
    set_annotation('robot_action', 'WAITING FOR VIAL')
    start_time = time.time()
    consecutive = 0

    while time.time() - start_time < timeout:
        with cam._lock:
            if cam.current_frame is None:
                time.sleep(0.1)
                continue
            frame = cam.current_frame.copy()

        vial_found = False
        conf = 0.0
        x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]

        if model:
            try:
                results = model(frame, verbose=False, conf=0.4)
                boxes = results[0].boxes
                if boxes and len(boxes) > 0:
                    box = max(boxes, key=lambda b: float(b.conf[0]))
                    conf = float(box.conf[0])
                    if conf >= 0.4:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        vial_found = True
            except:
                pass

        set_annotation('vial_detected', vial_found)
        set_annotation('conf', conf)

        bgr = (0, 255, 0) if vial_found else (0, 0, 255)
        label = f"VIAL DETECTED ({conf:.2f})" if vial_found else f"NO VIAL ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 4)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, bgr, 2)
        annotated = draw_overlay(frame, "EXT - VIAL CHECK")
        cv2.imshow('AI-CHEMIST External Camera', annotated)

        wrist_frame = dual_cam.get_wrist_frame()
        if wrist_frame is not None:
            wrist_ann = draw_overlay(wrist_frame.copy(), "WRIST - VIAL CHECK")
            cv2.imshow('Wrist Camera', wrist_ann)

        cv2.waitKey(1)

        if vial_found:
            consecutive += 1
            print(f"[VialCheck] Vial seen {consecutive}/3 (conf={conf:.2f})")
            if consecutive >= 3:
                elapsed = time.time() - start_time
                print(f"[VialCheck] ✅ Vial confirmed in {elapsed:.1f}s — starting pick!")
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'{output_dir}/images/VIAL_CONFIRMED_{ts}.jpg', annotated)
                if wrist_frame is not None:
                    cv2.imwrite(f'{output_dir}/images/wrist/VIAL_CONFIRMED_{ts}.jpg', wrist_frame)
                return True
        else:
            consecutive = 0

        time.sleep(0.5)

    print("[VialCheck]  Vial not detected - proceeding anyway")
    return False


def wait_for_colour_dual(detector, dual_cam, target_colour='RED', timeout=180, output_dir=None):
    print(f"[Wait] Waiting for: {target_colour} (timeout={timeout}s)")
    set_annotation('robot_action', f'WAITING FOR {target_colour}')
    start_time = time.time()
    consecutive_count = 0

    while time.time() - start_time < timeout:
        wrist_state = None
        wrist_frame = dual_cam.get_wrist_frame()
        if wrist_frame is not None:
            h, w = wrist_frame.shape[:2]
            margin = 40
            roi = wrist_frame[margin:h - margin, margin:w - margin]
            wrist_state, wrist_pixels = detector.detect_colour(roi)
            if isinstance(wrist_pixels, dict):
                set_annotation('pixels', wrist_pixels)

        ext_state = detector.current_state
        needed = 2 if ext_state == target_colour else 3

        if wrist_state == target_colour:
            consecutive_count += 1
            set_annotation('colour', wrist_state)
            print(f"[Wait] {wrist_state} [{consecutive_count}/{needed}]")
            if consecutive_count >= needed:
                duration = time.time() - start_time
                print(f"[Wait] ✅ {target_colour} CONFIRMED in {duration:.1f}s!")
                set_annotation('robot_action', f'{target_colour} CONFIRMED')
                if output_dir:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    if wrist_frame is not None:
                        cv2.imwrite(f'{output_dir}/images/wrist/CONFIRMED_{target_colour}_{ts}.jpg', wrist_frame)
                    if hasattr(detector, 'last_frame') and detector.last_frame is not None:
                        cv2.imwrite(f'{output_dir}/images/CONFIRMED_{target_colour}_{ts}.jpg', detector.last_frame)
                time.sleep(2)
                return True
        else:
            consecutive_count = 0

        time.sleep(1)

    print(f"[Wait] Timeout - proceeding")
    return False


def enhanced_detection(stop_event, cam, detector, vlm_monitor, digital_twin_local,
                       video_recorder, output_dir, dual_cam):
    model = None
    if YOLO_AVAILABLE:
        try:
            model = YOLO('/home/robot/Code/groupc/best.pt')
        except:
            pass

    video_recorder.start_recording("FULL_EXPERIMENT")
    wrist_frame_init = dual_cam.get_wrist_frame()
    if wrist_frame_init is not None:
        video_recorder.start_wrist_recording("FULL_EXPERIMENT_WRIST", wrist_frame_init.shape)

    save_count = 0

    while not stop_event.is_set():
        with cam._lock:
            if cam.current_frame is None:
                time.sleep(0.05)
                continue
            frame = cam.current_frame.copy()

        detections = []
        if model:
            try:
                results = model(frame, verbose=False, conf=0.25)
                detections = results[0].boxes if results[0].boxes else []
            except:
                pass

        if len(detections) > 0:
            box = max(detections, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            roi = frame[y1:y2, x1:x2]
            conf = float(box.conf[0])
            set_annotation('vial_detected', True)
            set_annotation('conf', conf)
            bgr = COLOURS_BGR.get(get_annotation()['colour'], (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 4)
        else:
            roi = frame
            x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
            conf = 0.0
            set_annotation('vial_detected', False)
            set_annotation('conf', 0.0)

        ext_state, ext_pixels = detector.detect_colour(roi)

        wrist_frame = dual_cam.get_wrist_frame()
        if wrist_frame is not None and USE_WRIST_CAM_FOR_COLOUR.is_set():
            h, w = wrist_frame.shape[:2]
            margin = 40
            wrist_roi = wrist_frame[margin:h - margin, margin:w - margin]
            wrist_state, wrist_pixels = detector.detect_colour(wrist_roi)
            final_state = wrist_state
            if isinstance(wrist_pixels, dict):
                set_annotation('pixels', wrist_pixels)
        else:
            final_state = ext_state
            if isinstance(ext_pixels, dict):
                set_annotation('pixels', ext_pixels)
            elif hasattr(ext_pixels, '__len__') and len(ext_pixels) >= 3:
                set_annotation('pixels', {'red': ext_pixels[2], 'green': ext_pixels[1], 'blue': ext_pixels[0]})

        detector.update_state(final_state)
        set_annotation('colour', final_state)

        frame = vlm_monitor.get_status_overlay(frame)
        digital_twin_local.update(detector)

        annotated_ext = draw_overlay(frame, "EXT")
        video_recorder.write_frame(annotated_ext)

        if wrist_frame is not None:
            annotated_wrist = draw_overlay(wrist_frame.copy(), "WRIST")
            video_recorder.write_wrist_frame(annotated_wrist)
            cv2.imshow('Wrist Camera', annotated_wrist)

        save_count += 1
        if save_count % 300 == 0:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f'{output_dir}/images/LIVE_{ts}_{final_state}.jpg', annotated_ext)
            if wrist_frame is not None:
                cv2.imwrite(f'{output_dir}/images/wrist/LIVE_{ts}_{final_state}.jpg', annotated_wrist)

        cv2.imshow('AI-CHEMIST External Camera [Q to quit]', annotated_ext)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_recorder.stop_recording()
    video_recorder.stop_wrist_recording()
    cv2.destroyAllWindows()


def phase1_oxidation(robot, gripper, cam, stirrer, ai_chemist, detector,
                     analyzer, output_dir, video_recorder, dual_cam, model):
    print("\nPHASE 1: Oxidation (Green->Red)")
    set_annotation('phase', 'PHASE 1 - OXIDATION')

    video_recorder.start_recording("PHASE1_OXIDATION")
    wrist_frame = dual_cam.get_wrist_frame()
    if wrist_frame is not None:
        video_recorder.start_wrist_recording("PHASE1_OXIDATION", wrist_frame.shape)

    set_annotation('robot_action', 'MOVING TO START')
    print("[Robot] Moving to start")
    robot.move_joint_list(YOUR_POSITIONS['start'], 0.5, 0.5, 0.02)
    gripper.move(gripper.get_open_position(), 125, 125)
    cam.capture_image(f'{output_dir}/images/ai_phase1_start.jpg')
    dual_cam.save_wrist_image('phase1_start')

    vial_found = check_vial_at_start(cam, detector, dual_cam, model, output_dir, timeout=30)
    if not vial_found:
        print("[Phase1] Vial not confirmed — continuing anyway")

    set_annotation('robot_action', 'PICKING VIAL')
    print("[Robot] Moving to pick")
    robot.move_joint_list(YOUR_POSITIONS['pick'], 0.5, 0.5, 0.02)
    gripper.move(255, 255, 255)
    cam.capture_image(f'{output_dir}/images/ai_phase1_pick.jpg')
    dual_cam.save_wrist_image('phase1_pick')

    set_annotation('robot_action', 'LIFTING')
    print("[Robot] Moving to up")
    robot.move_joint_list(YOUR_POSITIONS['up'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/ai_phase1_up.jpg')

    set_annotation('robot_action', 'MOVING TO STIRRER')
    print("[Robot] Moving to above_stirrer")
    robot.move_joint_list(YOUR_POSITIONS['above_stirrer'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/ai_phase1_above_stirrer.jpg')

    print("[Robot] Moving to hover")
    robot.move_joint_list(YOUR_POSITIONS['hover'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/ai_phase1_hover.jpg')

    rpm = ai_chemist.recommend_rpm('OXIDATION')
    stirrer.set_speed(rpm)
    stirrer.start_stirring()
    set_annotation('rpm', rpm)
    print(f"AI Chemist: Oxidation RPM = {rpm}")

    set_annotation('robot_action', 'INSERTING INTO STIRRER')
    print("[Robot] Moving to insert")
    robot.move_joint_list(YOUR_POSITIONS['insert'], 0.3, 0.3, 0.01)
    USE_WRIST_CAM_FOR_COLOUR.set()
    cam.capture_image(f'{output_dir}/images/ai_phase1_inserted.jpg')
    dual_cam.save_wrist_image('phase1_inserted')

    success = wait_for_colour_dual(detector, dual_cam, 'RED', timeout=180, output_dir=output_dir)
    metrics = analyzer.analyze_phase(detector)

    USE_WRIST_CAM_FOR_COLOUR.clear()
    video_recorder.stop_recording()
    video_recorder.stop_wrist_recording()
    return success, metrics


def phase2_reduction(robot, cam, stirrer, ai_chemist, detector, analyzer,
                     phase1_metrics, output_dir, video_recorder, dual_cam):
    print("\nPHASE 2: Reduction (Red->Yellow)")
    set_annotation('phase', 'PHASE 2 - REDUCTION')

    video_recorder.start_recording("PHASE2_REDUCTION")
    wrist_frame = dual_cam.get_wrist_frame()
    if wrist_frame is not None:
        video_recorder.start_wrist_recording("PHASE2_REDUCTION", wrist_frame.shape)

    set_annotation('robot_action', 'EXTRACTING FROM STIRRER')
    print("[Robot] Extract - hover")
    robot.move_joint_list(YOUR_POSITIONS['hover'], 0.5, 0.5, 0.02)
    USE_WRIST_CAM_FOR_COLOUR.clear()
    cam.capture_image(f'{output_dir}/images/ai_phase2_extract_hover.jpg')

    print("[Robot] Extract - above stirrer")
    robot.move_joint_list(YOUR_POSITIONS['above_stirrer'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/ai_phase2_above_stirrer.jpg')

    rpm2 = ai_chemist.adapt_rpm(1200, phase1_metrics.get('red_speed', 0))
    stirrer.set_speed(rpm2)
    set_annotation('rpm', rpm2)
    print(f"AI Chemist: Reduction RPM = {rpm2}")

    set_annotation('robot_action', 'RE-INSERTING')
    print("[Robot] Re-hover")
    robot.move_joint_list(YOUR_POSITIONS['hover'], 0.5, 0.5, 0.02)
    cam.capture_image(f'{output_dir}/images/ai_phase2_rehover.jpg')

    print("[Robot] Final insert")
    robot.move_joint_list(YOUR_POSITIONS['insert'], 0.3, 0.3, 0.01)
    USE_WRIST_CAM_FOR_COLOUR.set()
    cam.capture_image(f'{output_dir}/images/ai_phase2_final_insert.jpg')
    dual_cam.save_wrist_image('phase2_final_insert')

    success = wait_for_colour_dual(detector, dual_cam, 'YELLOW', timeout=180, output_dir=output_dir)
    metrics = analyzer.analyze_phase(detector)

    USE_WRIST_CAM_FOR_COLOUR.clear()
    video_recorder.stop_recording()
    video_recorder.stop_wrist_recording()
    return success, metrics


def main_novel():
    global digital_twin_global, stop_event

    print("AI-CHEMIST DUAL CAMERA SYSTEM")
    print("=" * 80)

    output_dir = create_output_folder()

    robot = URControl(ip=HOST, port=PORT)
    gripper = RobotiqGripper()
    gripper.connect(HOST, 63352)
    print(f"[Gripper] Current: {gripper.get_current_position()}")
    gripper.move(255, 125, 125)

    cam = Camera(EXTERNAL_CAM_INDEX, f'{output_dir}/images')
    cam.start_recording()
    print("[Camera] Recording started")
    time.sleep(2)

    dual_cam = DualCameraManager(output_dir)
    wrist_ok = dual_cam.start_wrist_camera()
    if not wrist_ok:
        print("[WristCam] Running without wrist camera")

    detector = ColourDetector()
    ai_chemist = AIChemist()
    digital_twin = DigitalTwin()
    digital_twin_global = digital_twin
    analyzer = ReactionAnalyzer()
    vlm_monitor = VLMMonitor(cam, '/home/robot/Code/groupc/best.pt')
    video_recorder = VideoRecorder(output_dir, cam)

    model = None
    if YOLO_AVAILABLE:
        try:
            model = YOLO('/home/robot/Code/groupc/best.pt')
        except:
            pass

    plot_thread = threading.Thread(target=plot_loop, daemon=True)
    plot_thread.start()

    det_thread = threading.Thread(
        target=enhanced_detection,
        args=(stop_event, cam, detector, vlm_monitor, digital_twin,
              video_recorder, output_dir, dual_cam),
        daemon=True
    )
    det_thread.start()

    stirrer = IKAStirrer(port='/dev/ttyACM0')

    try:
        set_annotation('phase', 'STARTUP')
        set_annotation('robot_action', 'INITIALISING')
        cam.capture_image(f'{output_dir}/images/STARTUP.jpg')
        dual_cam.save_wrist_image('STARTUP_wrist')

        phase1_success, phase1_metrics = phase1_oxidation(
            robot, gripper, cam, stirrer, ai_chemist, detector,
            analyzer, output_dir, video_recorder, dual_cam, model
        )

        if not phase1_success:
            print("Phase 1 incomplete - continuing")

        phase2_success, phase2_metrics = phase2_reduction(
            robot, cam, stirrer, ai_chemist, detector, analyzer,
            phase1_metrics, output_dir, video_recorder, dual_cam
        )

        set_annotation('phase', 'COMPLETE')
        set_annotation('robot_action', 'RETURNING TO START')
        print("\nSafe return to start position")
        USE_WRIST_CAM_FOR_COLOUR.clear()
        robot.move_joint_list(YOUR_POSITIONS['above_stirrer'], 0.5, 0.5, 0.02)
        gripper.move(255, 255, 255)
        robot.move_joint_list(YOUR_POSITIONS['start'], 0.5, 0.5, 0.02)
        gripper.move(gripper.get_open_position(), 125, 125)
        cam.capture_image(f'{output_dir}/images/ai_final_return.jpg')
        dual_cam.save_wrist_image('final_return')

        set_annotation('robot_action', 'DONE')
        analyzer.final_report()
        digital_twin.save_report(f'{output_dir}/reports/ai_chemist_report.png')

        summary = f"""
AI-CHEMIST DUAL CAMERA SUMMARY
================================
Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Folder: {output_dir}
Wrist Camera: {'YES' if wrist_ok else 'NO'}
Phase 1 Success: {'YES' if phase1_success else 'NO'}
Phase 1 Metrics: {phase1_metrics}
Phase 2 Success: {'YES' if phase2_success else 'NO'}
Phase 2 Metrics: {phase2_metrics}
================================
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
            f.write(f"ERROR: {str(e)}\n{traceback.format_exc()}")

    finally:
        try:
            stirrer.stop_stirring()
        except:
            pass
        USE_WRIST_CAM_FOR_COLOUR.clear()
        cam.stop_recording()
        dual_cam.stop_wrist_camera()
        video_recorder.stop_recording()
        video_recorder.stop_wrist_recording()
        stop_event.set()
        det_thread.join(timeout=3)
        plot_thread.join(timeout=3)
        print(f"\nDone. Files saved to: {output_dir}")


if __name__ == "__main__":
    main_novel()
