import sys
import os

# Fix 1: Use mambaforge torch (system Python doesn't have it)
MAMBAFORGE_PATH = '/home/robot/mambaforge/lib/python3.10/site-packages'
if MAMBAFORGE_PATH not in sys.path:
    sys.path.insert(0, MAMBAFORGE_PATH)

# Fix 2: Prevent Qt crash (we have no display)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from examples.utils.UR_Functions import URfunctions as URControl
from examples.utils.robotiq.robotiq_gripper import RobotiqGripper
from handlers.camera_handler_2 import Camera
from handlers.colour_detection import ColourDetector
from ultralytics import YOLO
import cv2
import math
import time
import threading
import numpy as np

from handlers.ika_strirrer import IKAStirrer

HOST = "192.168.0.2"
PORT = 30003



# ── Combined YOLO + Colour Detection Thread ──────────────────────────────────
def run_detection(stop_event, cam):
    model    = YOLO('/home/robot/Code/groupc/best.pt')
    detector = ColourDetector()
    print("[Detection] ✅ YOLO + Colour ready. Waiting for camera...")

    start = time.time()
    while cam.current_frame is None:
        if time.time() - start > 10:
            print("[Detection] ❌ Timed out waiting for camera frame.")
            return
        time.sleep(0.1)

    print("[Detection] 🎥 Running. Press 'q' to quit.")

    # Smoothing — vial must be missing N frames before declared lost
    miss_counter    = 0
    MISS_THRESHOLD  = 8
    last_known_box  = None
    stable_detected = False

    colour_map = {
        'GREEN':   (0, 255, 0),
        'RED':     (0, 0, 255),
        'YELLOW':  (0, 255, 255),
        'UNKNOWN': (128, 128, 128)
    }

    while not stop_event.is_set():
        with cam._lock:
            if cam.current_frame is None:
                time.sleep(0.05)
                continue
            frame = cam.current_frame.copy()

        # ── Step 1: YOLO ─────────────────────────────────────
        results    = model(frame, verbose=False, conf=0.25)
        detections = results[0].boxes
        raw_count  = len(detections) if detections is not None else 0
        annotated  = results[0].plot()

        # Smoothing
        if raw_count > 0:
            miss_counter    = 0
            stable_detected = True
            last_known_box  = max(detections, key=lambda b: float(b.conf[0]))
        else:
            miss_counter += 1
            if miss_counter >= MISS_THRESHOLD:
                stable_detected = False
                last_known_box  = None

        colour_state = "UNKNOWN"
        pixel_counts = {}

        # ── Step 2: Colour from YOLO bbox ────────────────────
        if stable_detected and last_known_box is not None:
            x1, y1, x2, y2 = map(int, last_known_box.xyxy[0].tolist())
            roi = frame[y1:y2, x1:x2]
            colour_state, pixel_counts = detector.detect_colour(roi)
            detector.update_state(colour_state)

            label_colour = colour_map.get(colour_state, (255, 255, 255))
            cv2.putText(annotated, f"Colour: {colour_state}",
                        (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, label_colour, 2)
            y = 60
            for c, count in pixel_counts.items():
                cv2.putText(annotated, f"{c}: {count}px", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_map.get(c, (255,255,255)), 1)
                y += 20

        # ── Step 3: Status bar ───────────────────────────────
        status = f"Vial: {'DETECTED' if stable_detected else 'NOT FOUND'} | Colour: {colour_state}"
        cv2.putText(annotated, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if detector.state_start_time:
            elapsed = time.time() - detector.state_start_time
            cv2.putText(annotated, f"Duration: {elapsed:.1f}s",
                        (10, annotated.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Vial Detection + Colour', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    detector.print_summary()
    cv2.destroyAllWindows()
    print("[Detection] 🛑 Stopped.")


# ── Main Robot Routine ───────────────────────────────────────
def main():
    robot   = URControl(ip=HOST, port=PORT)
    gripper = RobotiqGripper()
    gripper.connect(HOST, 63352)

    print(f"[Gripper] Current position: {gripper.get_current_position()}")
    gripper.move(255, 125, 125)

    cam = Camera(0, '/home/robot/Code/groupc/images/robot_vial_recording')
    cam.start_recording()
    print("[Camera] 🎥 Recording started.")
    time.sleep(2)

    stop_event       = threading.Event()
    detection_thread = threading.Thread(target=run_detection, args=(stop_event, cam), daemon=True)
    detection_thread.start()

    
    stirrer = IKAStirrer(port='/dev/ttyACM0')
    

    try:
        # ── Position 1: Start position, open gripper ──────────
        print("[Robot] Moving to Position 1 - Start")
        joint_state = [1.680945634841919, -1.9105240307249964, 2.3388877550708216,
                       -1.9763552151122035, -1.6369965712176722, 0.12794356048107147]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        gripper.move(gripper.get_open_position(), 125, 125)
        cam.capture_image('/home/robot/Code/groupc/images/position_1_start.jpg')

        # ── Position 2: Pick position, close gripper ──────────
        print("[Robot] Moving to Position 2 - Pick")
        joint_state = [1.6805652379989624, -1.7611729107298792, 2.4781621138202112,
                       -2.2649728260436, -1.638026539479391, 0.12882831692695618]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        gripper.move(255, 255, 255)
        cam.capture_image('/home/robot/Code/groupc/images/position_2_pick.jpg')
        
        # ── Position 3: Move directly up ──────────────────────
        print("[Robot] Moving to Position 3 - Up")
        joint_state = [1.680945634841919, -1.9105240307249964, 2.3388877550708216,
                       -1.9763552151122035, -1.6369965712176722, 0.12794356048107147]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        cam.capture_image('/home/robot/Code/groupc/images/position_3_up.jpg')

        # ── Position 4: Move vial above stirrer plate ─────────
        print("[Robot] Moving to Position 4 - Above Stirrer")
        joint_state = [1.5243818759918213, -1.3093386751464386, 1.5124495665179651,
                       -1.816165109673971, -1.6351736227618616, 0.12748508155345917]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        gripper.move(255, 255, 255)
        cam.capture_image('/home/robot/Code/groupc/images/position_4_above_stirrer.jpg')

        # ── Position 5: Hover vial above stirrer ──────────────
        print("[Robot] Moving to Position 5 - Hover")
        joint_state = [1.5242940187454224, -1.248557524090149, 1.6598289648639124,
                       -2.0242706737914027, -1.6358855406390589, 0.12835928797721863]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        cam.capture_image('/home/robot/Code/groupc/images/position_5_hover.jpg')

        stirrer.set_speed(500)
        stirrer.start_stirring()
        time.sleep(5)

        # ── Move down ─────────────────────────────────────────
        print("[Robot] Moving to Position - Down")
        joint_state = [1.537716269493103, -1.51904912412677, 1.6096895376788538,
                       -1.7614737949767054, -1.6358497778521937, 0.12833695113658905]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        cam.capture_image('/home/robot/Code/groupc/images/position_new_05_down.jpg')

        print("[Robot] Waiting 90 seconds...")
        time.sleep(250)

        # ── Move back above stirrer ───────────────────────────
        print("[Robot] Moving back above stirrer")
        joint_state = [1.5243818759918213, -1.3093386751464386, 1.5124495665179651,
                       -1.816165109673971, -1.6351736227618616, 0.12748508155345917]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        gripper.move(255, 255, 255)
        cam.capture_image('/home/robot/Code/groupc/images/position_4_above_stirrer_2.jpg')

        stirrer.set_speed(1500)
        stirrer.start_stirring()

        # ── Position 5 again ──────────────────────────────────
        print("[Robot] Moving to Position 5 - Hover again")
        joint_state = [1.5242940187454224, -1.248557524090149, 1.6598289648639124,
                       -2.0242706737914027, -1.6358855406390589, 0.12835928797721863]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        cam.capture_image('/home/robot/Code/groupc/images/position_5_hover_2.jpg')

        print("[Robot] Waiting 60 seconds...")
        time.sleep(30)

        # ── Final position ────────────────────────────────────
        print("[Robot] Moving to Final Position")
        joint_state = [1.537716269493103, -1.51904912412677, 1.6096895376788538,
                       -1.7614737949767054, -1.6358497778521937, 0.12833695113658905]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        time.sleep(10)
        cam.capture_image('/home/robot/Code/groupc/images/position_final.jpg')
        print("[Robot] ✅ Routine complete.")

    except Exception as e:
        print(f"[ERROR] ❌ {e}")

    finally:
        stirrer.stop_stirring()
        print("[Camera] Stopping recording...")
        cam.stop_recording()
        print("[Detection] Stopping...")
        stop_event.set()
        detection_thread.join(timeout=3)
        print("[Done] ✅ All systems stopped.")


def degreestorad(lst):
    return [x * (math.pi / 180) for x in lst]


if __name__ == "__main__":
    main()
