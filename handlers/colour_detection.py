import cv2
import numpy as np
import time
from datetime import datetime

class ColourDetector:
    def __init__(self, port=0):
        self.port = port
        self.current_state = None
        self.previous_state = None
        self.state_start_time = None
        self.transitions = []
        self.colour_ranges = {
            # Two ranges for GREEN:
            #   1) vivid green  — normal saturation/value
            #   2) pale green   — low saturation, high brightness (washed-out/light vials)
            'GREEN':  [
                {'lower': np.array([35, 40, 40]),   'upper': np.array([85, 255, 255])},  # vivid
                {'lower': np.array([35, 15, 160]),  'upper': np.array([85,  60, 255])}   # pale
            ],
            'RED':    [
                {'lower': np.array([0,   50, 50]),  'upper': np.array([10,  255, 255])},  # lower hue
                {'lower': np.array([170, 50, 50]),  'upper': np.array([180, 255, 255])}   # upper hue
            ],
            'YELLOW': [
                {'lower': np.array([20, 50, 50]),   'upper': np.array([35,  255, 255])}
            ]
        }

    def detect_vial(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=50, param2=30, minRadius=20, maxRadius=150)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            largest = max(circles, key=lambda c: c[2])
            x, y, r = largest
            cv2.circle(frame, (x, y), r, (255, 255, 0), 2)
            cv2.putText(frame, "Vial detected", (x-40, y-r-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            roi = frame[max(0, y-r):min(frame.shape[0], y+r),
                        max(0, x-r):min(frame.shape[1], x+r)]
            return roi, (x, y, r), frame
        cv2.putText(frame, "Vial NOT detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return None, None, frame

    def detect_colour(self, roi):
        if roi is None or roi.size == 0:
            return 'UNKNOWN', {}

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        pixel_counts = {}

        # All colours now use list-of-ranges — combine masks with bitwise OR
        for colour, ranges in self.colour_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for r in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, r['lower'], r['upper']))
            pixel_counts[colour] = cv2.countNonZero(mask)

        dominant = max(pixel_counts, key=pixel_counts.get)
        return (dominant, pixel_counts) if pixel_counts[dominant] > 300 else ('UNKNOWN', pixel_counts)

    def update_state(self, new_state):
        if new_state != self.current_state and new_state != 'UNKNOWN':
            if self.current_state is not None and self.state_start_time is not None:
                duration = time.time() - self.state_start_time
                t = {
                    'from_state': self.current_state,
                    'to_state':   new_state,
                    'duration':   round(duration, 2),
                    'timestamp':  datetime.now().strftime('%H:%M:%S')
                }
                self.transitions.append(t)
                print(f"[{t['timestamp']}] {self.current_state} → {new_state} "
                      f"(lasted {duration:.2f}s)")
            self.previous_state  = self.current_state
            self.current_state   = new_state
            self.state_start_time = time.time()

    def draw_overlay(self, frame, state, pixel_counts, vial_found):
        colour_map = {
            'GREEN':   (0, 255, 0),
            'RED':     (0, 0, 255),
            'YELLOW':  (0, 255, 255),
            'UNKNOWN': (128, 128, 128)
        }
        cv2.putText(frame, f"Vial: {'FOUND' if vial_found else 'NOT FOUND'}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if vial_found else (0, 0, 255), 2)
        cv2.putText(frame, f"State: {state}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    colour_map.get(state, (255, 255, 255)), 2)
        y = 110
        for c, count in pixel_counts.items():
            cv2.putText(frame, f"{c}: {count}px", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour_map[c], 1)
            y += 22
        if self.state_start_time:
            cv2.putText(frame, f"Duration: {time.time()-self.state_start_time:.1f}s",
                        (10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def print_summary(self):
        print("\n===== COLOUR TRANSITION SUMMARY =====")
        for t in self.transitions:
            print(f"[{t['timestamp']}] {t['from_state']} → {t['to_state']} "
                  f"| Duration: {t['duration']}s")
        print("=====================================\n")
