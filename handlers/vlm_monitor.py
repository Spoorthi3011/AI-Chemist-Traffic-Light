import cv2
import time
import threading
import numpy as np

# SAFE YOLO IMPORT
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ VLMMonitor: YOLO loaded")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  VLMMonitor: YOLO unavailable - color-only mode")
    class YOLO:
        def __init__(self, path): pass
        def __call__(self, *args, **kwargs): 
            class DummyResults:
                boxes = None
            return [DummyResults()]

class VLMMonitor:
    def __init__(self, cam, yolo_path):
        self.cam = cam
        self.yolo_path = yolo_path
        self.failure_detected = False
        self.alerts = []
        self.last_check = 0
        
        if YOLO_AVAILABLE:
            try:
                self.yolo = YOLO(yolo_path)
                print(f"✅ VLM: YOLO model loaded: {yolo_path}")
            except Exception as e:
                print(f"⚠️  VLM: Model load failed: {e}")
                self.yolo = None
        else:
            self.yolo = None
    
    def check_failures(self):
        """🧪 Vial presence + safety monitoring"""
        current_time = time.time()
        if current_time - self.last_check < 1.0:  # 1Hz check
            return self.failure_detected
        
        self.last_check = current_time
        
        with self.cam._lock:
            if self.cam.current_frame is None:
                return False
            frame = self.cam.current_frame.copy()
        
        # YOLO Detection (if available)
        if YOLO_AVAILABLE and self.yolo:
            try:
                results = self.yolo(frame, conf=0.3, verbose=False)
                vial_count = len(results[0].boxes) if results[0].boxes is not None else 0
            except:
                vial_count = 0  # Fallback
        else:
            vial_count = 1  # Assume OK without YOLO
        
        # Safety Logic
        if vial_count == 0:
            self.failure_detected = True
            self.alerts.append({
                'type': 'MISSING_VIAL', 
                'time': current_time,
                'confidence': 0.0
            })
            print(f"🚨 VLM ALERT: NO VIAL DETECTED! (count={vial_count})")
            return True
        
        # Clear alerts if vial found
        if self.failure_detected:
            print("✅ VLM: Vial recovered")
        self.failure_detected = False
        return False
    
    def get_status_overlay(self, frame):
        """🎨 Status overlay for live video"""
        status = "🟢 VIAL OK" if not self.failure_detected else "🔴 NO VIAL!"
        color = (0, 255, 0) if not self.failure_detected else (0, 0, 255)
        
        # Status background
        overlay = frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (10, h-90), (400, h-10), (20, 20, 20), -1)
        
        # Main status
        cv2.putText(overlay, status, (25, h-35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # YOLO status
        yolo_status = "YOLO✓" if YOLO_AVAILABLE and self.yolo else "YOLO✗"
        cv2.putText(overlay, f"YOLO: {yolo_status}", (25, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        
        # Alert count
        if self.alerts:
            cv2.putText(overlay, f"Alerts: {len(self.alerts)}", (250, h-35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
        
        return overlay
    
    def get_alerts(self):
        """Get recent alerts for AI Chemist"""
        recent = [a for a in self.alerts if time.time() - a['time'] < 60]
        self.alerts = [a for a in self.alerts if time.time() - a['time'] < 300]  # Keep 5min
        return recent

# Test function
def test_vlm():
    print("🧪 VLMMonitor Test")
    cam = type('Camera', (), {'_lock': threading.Lock(), 'current_frame': np.zeros((480,640,3), dtype=np.uint8)})()
    vlm = VLMMonitor(cam, '/home/robot/Code/groupc/best.pt')
    print("✅ VLMMonitor ready!")
