import cv2
import time
import threading

class Camera:
    def __init__(self, port: int, sample_name: str):
        self.finish: bool = False
        self.name = sample_name
        self.port = port
        self._thread = None
        self.current_frame = None
        self._lock = threading.Lock()

    def start_recording(self):
        self.finish = False
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        print("Recording started...")

    def stop_recording(self):
        self.finish = True
        if self._thread is not None:
            self._thread.join()
        print("Recording stopped.")

    def _record_loop(self):
        cap = cv2.VideoCapture(self.port)
        if not cap.isOpened():
            print("Error opening camera")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {width}x{height}")

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(self.name + '.avi', fourcc, 5.0, (width, height))

        while not self.finish:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 0)
                with self._lock:
                    self.current_frame = frame.copy()
                out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def capture_image(self, filename):
        with self._lock:
            if self.current_frame is not None:
                cv2.imwrite(filename, self.current_frame)
                print(f"Image saved: {filename}")
            else:
                print(f"No frame available for {filename}")
